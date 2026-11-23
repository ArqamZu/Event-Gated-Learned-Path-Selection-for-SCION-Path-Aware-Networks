"""Steps 5-9 - Proposed model, baselines, policy evaluation, metrics and statistics."""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_panel import make_labels
from config import BLEND_K, EMBARGO, PROC, SPLIT, SWITCH_BUDGET, TAB

warnings.filterwarnings("ignore")
# optional guard on the gate threshold (variant "guarded", not the Phase 2 rule): a threshold is only
# accepted if it removes MIN_GAIN validation failures without raising mean regret by more than REG_TOL
MIN_GAIN = 2
REG_TOL = 0.25    # ms
STALE_OBS = 12    # a path with fewer observed intervals in its current run (< 6 h of data) is "stale"

# --------------------------------------------------------------------------------------
# Feature groups (Phase 2, Step 4). Path fingerprints and vantage identifiers are never used.
# --------------------------------------------------------------------------------------
W = (1, 3, 6, 12)
GROUPS = {
    "current": ["last_rtt", "last_loss", "last_jit", "delta1"],
    "rolling": [f"rtt_mean_w{k}" for k in W] + [f"loss_mean_w{k}" for k in W] + [f"jit_mean_w{k}" for k in W]
               + [f"rtt_{s}_w{k}" for s in ("std", "min", "max") for k in W[1:]]
               + [f"rtt_slope_w{k}" for k in W[1:]] + ["rtt_ema6"],
    "stability": [f"{s}_w{k}" for s in ("spikes", "cpts", "lossev") for k in W[1:]],
    "structure": ["hop_count", "n_as", "n_isd"],
    "run_age": ["runage_h", "n_appear", "obs_in_run", "present", "timeout_now"],
    "observation": [f"nobs_w{k}" for k in W] + ["since_obs", "skipped_w12", "ctx_n_probed", "ctx_n_present"],
    "relative": ["gap_last", "gap_ema", "n_within10", "rank_last", "ctx_margin2"],
    "temporal": ["hod_sin", "hod_cos", "dow_sin", "dow_cos"],
    # added in Phase 3 (error-analysis feedback, Step 11 -> Step 4)
    "volatility": ["ctx_prev_regret", "ctx_prev_fail", "ctx_bestchg_w6", "ctx_bestchg_w12",
                   "ctx_fail_w6", "ctx_fail_w12"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
# short-history ranker of the stale-history blend (Phase 2, Step 5): path structure plus
# short-window features only (last observation and the 1-interval window)
SHORT_FEATURES = (GROUPS["structure"] + GROUPS["current"]
                  + ["rtt_mean_w1", "loss_mean_w1", "jit_mean_w1", "nobs_w1", "since_obs"])


def features(drop_groups=(), max_window=12):
    keep = [f for g, fs in GROUPS.items() if g not in drop_groups for f in fs]
    return [f for f in keep if not any(f.endswith(f"_w{k}") for k in W if k > max_window)]


def load():
    C = pd.read_parquet(PROC / "candidates.parquet")
    C["day"] = C.slot.dt.floor("D")
    C["src_v"] = C.pair.str.split("->").str[0]
    C["dst_v"] = C.pair.str.split("->").str[1]
    C.attrs["label"] = "fail20ms_loss5"
    # memory incumbent (Phase 3 change): the candidate with the lowest *last observed* RTT,
    # considering observations up to 12 intervals old. Unlike the Phase 2 incumbent (best of the
    # previous interval only) it is not fooled when the best path was simply not sampled at t-1.
    key = C.last_rtt.where(C.since_obs <= 12).fillna(np.inf) - C.is_inc * 1e-6
    C["is_inc_mem"] = False
    C.loc[key.groupby(C.decision_id).idxmin(), "is_inc_mem"] = True
    return C


def relabel(C, ms, loss):
    """Copy of C whose `fail` and `rel` columns use another (regret margin, loss) threshold."""
    D = C.copy()
    D["fail"], D["rel"] = make_labels(D, ms=ms, loss=loss)
    D.attrs["label"] = f"fail{ms:g}ms_loss{loss:g}"
    return D


def chrono_split(C, frac=SPLIT):
    """70/15/15 chronological split of decisions within each directed pair."""
    d = C.groupby("decision_id")[["pair", "t"]].first().sort_values(["pair", "t"])
    r = d.groupby("pair").cumcount() / d.groupby("pair").t.transform("size")
    lab = np.select([r < frac[0], r < frac[0] + frac[1]], ["train", "val"], "test")
    return C.decision_id.map(pd.Series(lab, index=d.index))


def purged_folds(C, n_blocks=6, embargo=EMBARGO):
    """Expanding-window CV: block k is tested on a model trained on blocks < k, minus an
    embargo of `embargo` intervals before the test block (per pair). Yields (train_ids, test_ids)."""
    d = C.groupby("decision_id")[["pair", "t"]].first()
    r = d.groupby("pair").t.rank(method="first") - 1
    blk = (r / d.groupby("pair").t.transform("size") * n_blocks).astype(int).clip(upper=n_blocks - 1)
    for k in range(1, n_blocks):
        start = d[blk == k].groupby("pair").t.min()
        tr = d[(blk < k) & (d.t < d.pair.map(start) - embargo)].index
        te = d[blk == k].index
        yield k, tr, te


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
GATE_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
RANK_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
_tuned = TAB / "tuning.json"
if _tuned.exists():  # parameters selected by tune.py (purged CV inside the training split)
    _t = json.loads(_tuned.read_text())
    GATE_PARAMS.update(_t["gate"]); RANK_PARAMS.update(_t["ranker"])


def inner_val(ids, C, frac=0.15):
    """Last `frac` of each pair's training decisions, used for early stopping / threshold tuning."""
    d = C[C.decision_id.isin(ids)].groupby("decision_id")[["pair", "t"]].first()
    r = d.groupby("pair").t.rank(pct=True)
    return d.index[r <= 1 - frac], d.index[r > 1 - frac]


class Gate:
    """Stage 1: predicts whether the incumbent path fails in the next interval."""

    def __init__(self, kind="lgbm", feats=None, weighting="class", seed=0, params=None, label="fail"):
        self.kind, self.feats, self.weighting, self.seed, self.label = kind, feats or ALL_FEATURES, weighting, seed, label
        self.params = {**GATE_PARAMS, **(params or {})}

    def fit(self, tr, va):
        X, y = tr[self.feats], tr[self.label].astype(int).to_numpy()
        pos = max(y.sum(), 1)
        if self.weighting == "smote" and pos >= 2:
            from imblearn.over_sampling import SMOTE
            Xi = X.fillna(X.median()).fillna(0)
            X, y = SMOTE(random_state=self.seed, k_neighbors=min(5, pos - 1)).fit_resample(Xi, y)
            X = pd.DataFrame(X, columns=self.feats)
        self.constant = None
        if y.min() == y.max():  # no failure event (or only events) in the training rows: constant gate
            self.constant = float(y[0])
            return self
        w = np.where(y == 1, (len(y) - y.sum()) / max(y.sum(), 1), 1.0) if self.weighting == "class" else None
        if self.kind == "lgbm":
            self.m = lgb.LGBMClassifier(random_state=self.seed, **self.params)
            yv = va[self.label].astype(int)
            if 0 < yv.sum() < len(yv):
                self.m.fit(X, y, sample_weight=w, eval_set=[(va[self.feats], yv)], eval_metric="average_precision",
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            else:  # no validation event: early stopping is undefined, use a fixed number of trees
                self.m.set_params(n_estimators=200).fit(X, y, sample_weight=w)
        elif self.kind == "rf":
            self.m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                   RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                                                          class_weight="balanced" if self.weighting == "class" else None,
                                                          random_state=self.seed))
            self.m.fit(X, y)
        elif self.kind == "lr":
            self.m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                   LogisticRegression(C=0.1, max_iter=2000,
                                                      class_weight="balanced" if self.weighting == "class" else None))
            self.m.fit(X, y)
        return self

    def predict(self, df):
        if self.kind == "jump":
            return df["delta1"].fillna(0).to_numpy()
        if getattr(self, "constant", None) is not None:
            return np.full(len(df), self.constant)
        return self.m.predict_proba(df[self.feats])[:, 1]


class Ranker:
    """Stage 2: LambdaRank model ordering the candidate paths of a decision."""

    def __init__(self, feats=None, seed=0, params=None):
        self.feats, self.seed = feats or ALL_FEATURES, seed
        self.params = {**RANK_PARAMS, **(params or {})}

    @staticmethod
    def _grp(df):
        df = df.sort_values(["decision_id"])
        return df, df.groupby("decision_id", sort=False).size().to_numpy()

    def fit(self, tr, va):
        tr, gt = self._grp(tr[tr.n_cand >= 2]); va, gv = self._grp(va[va.n_cand >= 2])
        self.m = lgb.LGBMRanker(random_state=self.seed, **self.params)
        self.m.fit(tr[self.feats], tr.rel, group=gt, eval_set=[(va[self.feats], va.rel)], eval_group=[gv],
                   eval_at=[1], callbacks=[lgb.early_stopping(100, verbose=False)])
        return self

    def predict(self, df):
        return self.m.predict(df[self.feats])


def zscore_within(df, col):
    g = df.groupby("decision_id")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0)


def blend_weight(n, k=BLEND_K):
    """Weight of the full-history score: n / (n + k), n = observed intervals in the current run."""
    n = np.asarray(n, float)
    return np.nan_to_num(n, nan=0.0) / (np.nan_to_num(n, nan=0.0) + k)


def fit_rankers(tr, va, feats, seed, blend, params=None):
    rk = {"full": Ranker(feats, seed, params).fit(tr, va)}
    if blend:
        sf = [f for f in SHORT_FEATURES if f in feats] or SHORT_FEATURES
        rk["short"] = Ranker(sf, seed, params).fit(tr, va)
    return rk


def rank_score(df, rk, blend_k=BLEND_K):
    """Ranking score; with a short-history ranker, the two z-scored scores are blended."""
    full = pd.Series(rk["full"].predict(df), index=df.index)
    if "short" not in rk:
        return full
    short = pd.Series(rk["short"].predict(df), index=df.index)
    w = blend_weight(df.obs_in_run, blend_k)
    z = df[["decision_id"]].assign(f=full, s=short)
    return w * zscore_within(z, "f") + (1 - w) * zscore_within(z, "s")


def top_flag(df, score):
    flag = pd.Series(False, index=df.index)
    flag.loc[score.groupby(df.decision_id).idxmax()] = True
    return flag


def time_blocks(df, n_blocks):
    d = df.groupby("decision_id")[["pair", "t"]].first()
    r = d.groupby("pair").t.rank(method="first") - 1
    blk = (r / d.groupby("pair").t.transform("size") * n_blocks).astype(int).clip(upper=n_blocks - 1)
    return d, blk


_CACHE = {}


def _key(*parts):
    if len(_CACHE) > 400:  # bound memory during long ablation runs
        _CACHE.clear()
    return hash(tuple(parts))


def clear_cache():
    _CACHE.clear()


def oof_rank_top(tr, feats, seed, blend, blend_k, params=None, n_blocks=4, embargo=EMBARGO, tag=None):
    """Cross-fitted ranker top path on training decisions. Training decisions are cut into
    `n_blocks` time blocks per pair; each block is scored by rankers fitted on the other blocks,
    with an embargo of `embargo` intervals on both sides. The gate is then trained on the path the
    ranker would actually have chosen, without in-sample optimism."""
    key = _key("oof", tag, tuple(feats), seed, blend, blend_k,
               tuple(sorted((params or {}).items())), n_blocks, embargo, tuple(np.sort(tr.decision_id.unique())))
    if key in _CACHE:
        return _CACHE[key].reindex(tr.index, fill_value=False)
    d, blk = time_blocks(tr, n_blocks)
    flag = pd.Series(False, index=tr.index)
    for b in range(n_blocks):
        inb = blk == b
        lo, hi = d[inb].groupby("pair").t.min(), d[inb].groupby("pair").t.max()
        far = (d.t < d.pair.map(lo) - embargo) | (d.t > d.pair.map(hi) + embargo)
        a, v = inner_val(d.index[~inb & far], tr)
        rk = fit_rankers(tr[tr.decision_id.isin(a)], tr[tr.decision_id.isin(v)], feats, seed, blend, params)
        part = tr[tr.decision_id.isin(d.index[inb])]
        flag |= top_flag(part, rank_score(part, rk, blend_k)).reindex(tr.index, fill_value=False)
    _CACHE[key] = flag
    return flag


class Switcher:
    """Event-gated switcher: keep the incumbent unless the gate fires, then take the
    top-ranked alternative. Variants: ranker-only, gate-only, no blend."""

    def __init__(self, feats=None, gate_kind="lgbm", weighting="class", blend=True, mode="gate+ranker",
                 seed=0, budget=SWITCH_BUDGET, label="fail", incumbent="ranker", guard=False, blend_k=BLEND_K,
                 gate_params=None, rank_params=None):
        self.feats = feats or ALL_FEATURES
        self.gate_kind, self.weighting, self.blend, self.mode = gate_kind, weighting, blend, mode
        self.seed, self.budget, self.label, self.guard, self.blend_k = seed, budget, label, guard, blend_k
        self.gate_params, self.rank_params = gate_params, rank_params
        # incumbent definition:
        #   "phase2": best path of the previous interval (Phase 2 design)
        #   "memory": lowest last-observed RTT within 6 h (last-known-best)
        #   "ranker": top-ranked path of the stage-2 ranker (Phase 3 final design). The gate is
        #             trained on exactly these rows: on training data the ranker top is cross-fitted
        #             (oof_rank_top), on validation / test it is the fitted ranker's top path.
        self.incumbent = incumbent
        self.inc = {"phase2": "is_inc", "memory": "is_inc_mem", "ranker": "_inc_rank"}[incumbent]
        if incumbent == "ranker" and mode == "gate":
            raise ValueError("gate-only mode needs a fixed incumbent")

    def fit(self, C, train_ids, val_ids=None):
        t0 = time.perf_counter()
        if val_ids is None:
            train_ids, val_ids = inner_val(train_ids, C)
        tr, va = C[C.decision_id.isin(train_ids)], C[C.decision_id.isin(val_ids)]
        if self.mode != "gate":
            key = _key("rk", C.attrs.get("label"), tuple(self.feats), self.seed, self.blend,
                       tuple(sorted((self.rank_params or {}).items())), tuple(np.sort(tr.decision_id.unique())),
                       tuple(np.sort(va.decision_id.unique())))
            if key not in _CACHE:
                _CACHE[key] = fit_rankers(tr, va, self.feats, self.seed, self.blend, self.rank_params)
            self.rk = _CACHE[key]
            self.ranker, self.short = self.rk["full"], self.rk.get("short")
        if self.mode != "ranker":
            if self.incumbent == "ranker":
                tr = tr.assign(_inc_rank=oof_rank_top(tr, self.feats, self.seed, self.blend, self.blend_k,
                                                      self.rank_params, tag=C.attrs.get("label")))
                va = va.assign(_inc_rank=top_flag(va, rank_score(va, self.rk, self.blend_k)))
            gtr, gva = tr[tr[self.inc]], va[va[self.inc]]
            self.gate_train_index, self.gate_val_index = gtr.index, gva.index  # for audits (error analysis)
            self.gate = (Gate(kind="jump") if self.gate_kind == "jump" else
                         Gate(self.gate_kind, self.feats, self.weighting, self.seed, self.gate_params,
                              label=self.label).fit(gtr, gva))
        self.fit_seconds = time.perf_counter() - t0
        self.tau = -np.inf
        if self.mode != "ranker":
            self.tau = self.tune_threshold(va)
        return self

    def score(self, df):
        df = df.copy()
        if self.mode != "gate":
            df["score"] = rank_score(df, self.rk, self.blend_k)
        else:  # gate-only: runner-up by last observed RTT (recent observations only)
            df["score"] = -df.last_rtt.where(df.since_obs <= 12).fillna(1e9)
        if self.incumbent == "ranker":
            df["_inc_rank"] = top_flag(df, df.score)
        if self.mode != "ranker":
            m = df[self.inc]
            df["p_gate"] = df.decision_id.map(pd.Series(self.gate.predict(df[m]), index=df.loc[m, "decision_id"].to_numpy()))
        return df

    def choose(self, df, tau, mode):
        """Return one chosen row per evaluable decision (Phase 2 evaluable set: the
        previous-interval incumbent was probed at t), so every policy is scored on the same decisions."""
        df = df[df.decision_id.isin(df.loc[df.is_inc, "decision_id"])]
        if mode == "ranker":
            return df.loc[df.groupby("decision_id").score.idxmax()]
        alt = df[~df[self.inc]]
        top_alt = alt.loc[alt.groupby("decision_id").score.idxmax()].set_index("decision_id")
        inc = df[df[self.inc]].set_index("decision_id")
        fire = (inc.p_gate > tau) & inc.index.isin(top_alt.index)
        out = pd.concat([inc[~fire], top_alt.loc[inc.index[fire]]]).reset_index()
        out["fired"] = out.decision_id.isin(inc.index[fire])
        return out

    def tune_threshold(self, va):
        """Phase 2, Step 5: the threshold that minimizes validation failure share subject to the
        switching budget (share of decisions on which the gate fires). Ties are broken by mean
        regret, then by firing less. With guard=True a threshold must in addition avoid at least
        MIN_GAIN validation failures and keep mean regret within REG_TOL ms of never firing."""
        s = self.score(va)
        ps = s.loc[s[self.inc], "p_gate"].to_numpy()
        grid = np.unique(np.quantile(ps, np.linspace(0.5, 1.0, 101)))
        never = self.choose(s, np.inf, self.mode)
        best, tau_best = (never.fail.sum(), never.regret.mean()), np.inf
        self.val_curve = [dict(tau=np.inf, fired=0.0, n_fail=int(best[0]), regret=best[1])]
        for tau in grid[::-1]:  # descending: fewer firings first, so ties keep the higher threshold
            ch = self.choose(s, tau, self.mode)
            fired = ch.fired.mean()
            key = (ch.fail.sum(), ch.regret.mean())
            self.val_curve.append(dict(tau=tau, fired=fired, n_fail=int(key[0]), regret=key[1]))
            if fired > self.budget:
                continue
            if self.guard and not (key[0] <= never.fail.sum() - MIN_GAIN and key[1] <= never.regret.mean() + REG_TOL):
                continue
            if key < best:
                best, tau_best = key, tau
        return tau_best

    def select(self, df):
        t0 = time.perf_counter()
        s = self.score(df)
        ch = self.choose(s, self.tau, self.mode)
        self.infer_ms = (time.perf_counter() - t0) * 1000 / max(len(ch), 1)
        self.scored = s
        return ch

    def decide(self, df):
        """Deployment-style inference: one row per decision with the chosen path, the incumbent and
        the gate probability. Unlike select() it needs no ground truth and keeps every decision."""
        s = self.score(df)
        if self.mode == "ranker":
            out = s.loc[s.groupby("decision_id").score.idxmax()]
            return out.assign(incumbent_fp=out.fp, p_gate=np.nan, fired=False)
        alt = s[~s[self.inc]]
        top_alt = alt.loc[alt.groupby("decision_id").score.idxmax()].set_index("decision_id")
        inc = s[s[self.inc]].set_index("decision_id")
        fire = (inc.p_gate > self.tau) & inc.index.isin(top_alt.index)
        out = pd.concat([inc[~fire], top_alt.loc[inc.index[fire]]])
        out["incumbent_fp"] = inc.fp.reindex(out.index)
        out["p_gate"] = inc.p_gate.reindex(out.index)
        out["fired"] = out.index.isin(inc.index[fire])
        return out.reset_index()


# --------------------------------------------------------------------------------------
# Baselines (Phase 2, Step 6). Each returns one chosen candidate row per evaluable decision.
# --------------------------------------------------------------------------------------
def _ev(df):
    return df[df.decision_id.isin(df.loc[df.is_inc, "decision_id"])]


def _argmin(df, col):
    df = df.assign(_k=df[col].fillna(np.inf))
    return df.loc[df.groupby("decision_id")._k.idxmin()].drop(columns="_k")


def mm(df, col):
    g = df.groupby("decision_id")[col]
    lo, hi = g.transform("min"), g.transform("max")
    return ((df[col] - lo) / (hi - lo).replace(0, np.nan)).fillna(0)


def baseline(df, name, seed=0, loss_limit=5.0):
    """loss_limit: the oracle picks the lowest RTT among paths with loss <= loss_limit (the label's limit)."""
    df = _ev(df)
    if name == "Random":
        rng = np.random.default_rng(seed)
        return df.loc[df.assign(_r=rng.random(len(df))).groupby("decision_id")._r.idxmax()]
    if name == "Persistence":
        return df[df.is_inc]
    if name == "Last-known-best":
        return df[df.is_inc_mem]
    if name == "Nowcast min-RTT*":
        return _argmin(df, "rtt_t")
    if name == "EMA-6 RTT":
        return _argmin(df, "rtt_ema6")
    if name == "ScionPathML heuristic (nowcast)*":  # a dead path (no RTT) gets the worst RTT score
        return _argmin(df.assign(h=0.5 * mm(df, "rtt_t").where(df.rtt_t.notna(), 1) + 0.5 * mm(df, "loss_t")), "h")
    if name == "ScionPathML heuristic (forecast)":
        x = df.assign(lr=df.last_rtt.where(df.since_obs <= 1), ll=df.last_loss.where(df.since_obs <= 1))
        return _argmin(x.assign(h=0.5 * mm(x, "lr").where(x.lr.notna(), 1) + 0.5 * mm(x, "ll").where(x.ll.notna(), 1)), "h")
    if name == "Epsilon-greedy (0.1)":
        rng = np.random.default_rng(seed)
        greedy = _argmin(df, "cummean_rtt").set_index("decision_id")
        rnd = df.loc[df.assign(_r=rng.random(len(df))).groupby("decision_id")._r.idxmax()].set_index("decision_id")
        explore = rng.random(len(greedy)) < 0.1
        return pd.concat([greedy[~explore], rnd.loc[greedy.index[explore]]]).reset_index()
    if name == "Oracle*":
        x = df.assign(_o=np.where(df.loss_t <= loss_limit, df.rtt_t, df.rtt_t.fillna(1e6) + 1e6))
        return _argmin(x, "_o")
    raise ValueError(name)


BASELINES = ["Random", "Persistence", "Last-known-best", "Nowcast min-RTT*", "EMA-6 RTT", "ScionPathML heuristic (nowcast)*",
             "ScionPathML heuristic (forecast)", "Epsilon-greedy (0.1)", "Oracle*"]


# --------------------------------------------------------------------------------------
# Metrics (Phase 2, Step 8)
# --------------------------------------------------------------------------------------
def switches(ch):
    ch = ch.sort_values(["pair", "t"])
    prev_fp = ch.groupby("pair").fp.shift()
    prev_t = ch.groupby("pair").t.shift()
    return ((ch.fp != prev_fp) & (ch.t - prev_t == 1)).to_numpy()


def policy_metrics(ch):
    """Decision quality, failure share and stability. Switch rates:
       switch_per_pair_day : switches per 48 evaluated decisions of one directed pair (one pair-day)
       switch_per_day      : switches of all pairs together per calendar day covered"""
    ch = ch.sort_values(["pair", "t"])
    sw = switches(ch)
    pair_days = len(ch) / 48
    cal_days = ch.slot.dt.floor("D").nunique() if "slot" in ch else np.nan
    return dict(n=len(ch), fail_pct=100 * ch.fail.mean(), fail50_pct=100 * ch.fail50.mean(),
                n_fail=int(ch.fail.sum()), regret_mean=ch.regret.mean(), regret_p90=ch.regret.quantile(0.9),
                regret_p99=ch.regret.quantile(0.99), switch_per_dec=sw.mean(),
                switch_per_pair_day=sw.sum() / pair_days, switch_per_day=sw.sum() / cal_days,
                fired_pct=100 * ch.fired.mean() if "fired" in ch else np.nan)


def gate_metrics(y, p, tau):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yhat = p > tau
    two = 0 < y.sum() < len(y)
    return dict(auprc=average_precision_score(y, p) if two else np.nan,
                auroc=roc_auc_score(y, p) if two else np.nan, prevalence=y.mean(),
                precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0), mcc=matthews_corrcoef(y, yhat) if two else np.nan,
                bal_acc=balanced_accuracy_score(y, yhat) if two else np.nan, n_events=int(y.sum()),
                fired_pct=100 * yhat.mean())


# --------------------------------------------------------------------------------------
# Statistics (Phase 2, Step 9)
# --------------------------------------------------------------------------------------
def boot_ci(ch, ref=None, B=4000, seed=0):
    """95 % CIs by resampling whole days (consecutive decisions are autocorrelated).
    Returns CIs for failure share (%) and mean regret, and - if `ref` (another policy on the
    same decisions) is given - for the paired differences policy minus ref."""
    rng = np.random.default_rng(seed)
    a = ch.set_index("decision_id")[["day", "fail", "regret"]]
    if ref is not None:
        a = a.join(ref.set_index("decision_id")[["fail", "regret"]], rsuffix="_r", how="inner")
    g = a.groupby("day")
    n = g.size().to_numpy(); f = g.fail.sum().to_numpy(); r = g.regret.sum().to_numpy()
    idx = rng.integers(0, len(n), (B, len(n)))
    N = n[idx].sum(1)
    out = dict(fail_ci=np.percentile(100 * f[idx].sum(1) / N, [2.5, 97.5]),
               regret_ci=np.percentile(r[idx].sum(1) / N, [2.5, 97.5]))
    if ref is not None:
        fr = g.fail_r.sum().to_numpy(); rr = g.regret_r.sum().to_numpy()
        out["dfail_ci"] = np.percentile(100 * (f[idx].sum(1) - fr[idx].sum(1)) / N, [2.5, 97.5])
        out["dregret_ci"] = np.percentile((r[idx].sum(1) - rr[idx].sum(1)) / N, [2.5, 97.5])
    return out


def paired(ch_a, ch_b):
    """Align two policies' choices on the same decisions."""
    a = ch_a.set_index("decision_id")[["fail", "regret", "day"]]
    b = ch_b.set_index("decision_id")[["fail", "regret"]]
    return a.join(b, rsuffix="_b", how="inner")


def mcnemar(ch_a, ch_b):
    j = paired(ch_a, ch_b)
    n01 = int((j.fail & ~j.fail_b).sum()); n10 = int((~j.fail & j.fail_b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only=n01, b_only=n10, p=p, odds_ratio=(n01 + 0.5) / (n10 + 0.5))


def wilcoxon_days(ch_a, ch_b):
    j = paired(ch_a, ch_b)
    d = j.groupby("day").apply(lambda g: g.regret.mean() - g.regret_b.mean())
    if (d != 0).sum() < 2:
        return dict(p=1.0, rank_biserial=0.0, n_days=len(d))
    res = wilcoxon(d)
    nz = d[d != 0]
    r = pd.Series(np.abs(nz)).rank().to_numpy()
    rb = (r[nz.to_numpy() > 0].sum() - r[nz.to_numpy() < 0].sum()) / r.sum()
    return dict(p=res.pvalue, rank_biserial=rb, n_days=len(d))


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(run, 1.0)
    return adj
