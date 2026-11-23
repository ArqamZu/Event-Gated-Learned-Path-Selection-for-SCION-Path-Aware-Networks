"""Step 7 - Experimental design (A-G), Step 9 statistics, Step 10 ablations, Step 11 error analysis.

Two evaluation protocols are used throughout:
  * TEST : chronological 70/15/15 split per pair (model fit on train, threshold on val).
  * CV   : purged expanding-window CV over the whole timeline (6 blocks -> 5 folds,
           12-interval embargo); out-of-fold decisions are pooled. CV is the primary
           protocol because the 15 % test split holds only ~20 failure events.

Usage:  python experiments.py [main|ablation|robust|error|all]
"""
import json
import resource
import sys
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from config import EMBARGO, LABEL_GRID, PINGS_PER_PROBE, PROC, QOE_PHASE2, QOE_PREPRINT, SEEDS, TAB
from core import (ALL_FEATURES, BASELINES, GROUPS, STALE_OBS, Gate, Switcher, baseline, boot_ci, chrono_split,
                  clear_cache, features, gate_metrics, holm, load, mcnemar, policy_metrics, purged_folds, relabel,
                  switches, wilcoxon_days)

C = load()
C["split"] = chrono_split(C)
FOLDS = list(purged_folds(C))
EV = set(C.loc[C.is_inc, "decision_id"])  # evaluable decisions
OUT = PROC / "runs"
OUT.mkdir(exist_ok=True)
GATE_COLS = ["decision_id", "pair", "fail", "fail50", "p_gate", "fp", "regret", "loss_t", "dead", "obs_in_run",
             "since_obs"]


def gate_rows(m):
    s = m.scored
    return s[s[m.inc] & s.decision_id.isin(EV)]


# decisions where the best path at t had not been probed at t-1 (sampling artifact for persistence)
_best = C[C.decision_id.isin(EV)].loc[lambda d: d.groupby("decision_id").rtt_t.idxmin()]
ARTIFACT = set(_best.loc[(_best.since_obs > 0) | _best.last_rtt.isna(), "decision_id"])


def ids(split, D=None):
    D = C if D is None else D
    return D.loc[D.split == split, "decision_id"].unique()


def run_test(seed=0, data=None, **cfg):
    D = C if data is None else data
    m = Switcher(seed=seed, **cfg).fit(D, ids("train", D), ids("val", D))
    ch = m.select(D[D.split == "test"])
    return m, ch


def run_cv(seed=0, folds=None, train_mask=None, data=None, **cfg):
    """Pooled out-of-fold choices and gate scores. `train_mask(df)->bool` drops training rows;
    `data` replaces C (e.g. a relabelled copy for threshold sensitivity)."""
    D0 = C if data is None else data
    chs, gates, fits = [], [], []
    for k, tr, te in folds or FOLDS:
        D = D0 if train_mask is None else D0[~(D0.decision_id.isin(tr) & train_mask(D0))]
        m = Switcher(seed=seed, **cfg).fit(D, tr)
        ch = m.select(D0[D0.decision_id.isin(te)]).assign(fold=k)
        chs.append(ch)
        if m.mode != "ranker":
            s = gate_rows(m)
            gates.append(s[GATE_COLS].assign(tau=m.tau, fold=k))
        fits.append(dict(fold=k, fit_s=m.fit_seconds, infer_ms=m.infer_ms, tau=m.tau))
    return (pd.concat(chs, ignore_index=True), pd.concat(gates, ignore_index=True) if gates else None,
            pd.DataFrame(fits))


def cv_baselines(seed=0, data=None, loss_limit=5.0):
    D = C if data is None else data
    te = D[D.decision_id.isin(np.concatenate([f[2] for f in FOLDS]))]
    return {b: baseline(te, b, seed, loss_limit) for b in BASELINES}


def summarize(chs):
    rows = []
    for name, ch in chs.items():
        m = policy_metrics(ch)
        ci = boot_ci(ch, chs["Persistence"]); cl = boot_ci(ch, chs["Last-known-best"])
        m.update(fail_lo=ci["fail_ci"][0], fail_hi=ci["fail_ci"][1], reg_lo=ci["regret_ci"][0],
                 reg_hi=ci["regret_ci"][1], dfail_lo=ci["dfail_ci"][0], dfail_hi=ci["dfail_ci"][1],
                 dreg_lo=ci["dregret_ci"][0], dreg_hi=ci["dregret_ci"][1],
                 dfail_lkb_lo=cl["dfail_ci"][0], dfail_lkb_hi=cl["dfail_ci"][1],
                 dreg_lkb_lo=cl["dregret_ci"][0], dreg_lkb_hi=cl["dregret_ci"][1])
        rows.append(dict(policy=name, **m))
    return pd.DataFrame(rows)


def stats_vs(chs, target="Proposed"):
    rows = []
    for name, ch in chs.items():
        if name == target:
            continue
        mc = mcnemar(chs[target], ch); wx = wilcoxon_days(chs[target], ch)
        rows.append(dict(comparison=f"{target} vs {name}", proposed_only_fail=mc["a_only"],
                         other_only_fail=mc["b_only"], mcnemar_p=mc["p"], odds_ratio=mc["odds_ratio"],
                         wilcoxon_p=wx["p"], rank_biserial=wx["rank_biserial"], n_days=wx["n_days"]))
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = holm(df.mcnemar_p); df["wilcoxon_p_holm"] = holm(df.wilcoxon_p)
    return df


def with_switch_flag(ch):
    """Chosen rows with a per-decision switch flag computed on the full, ordered sequence."""
    ch = ch.sort_values(["pair", "t"]).copy()
    ch["switched"] = switches(ch)
    return ch


# ======================================================================================
# A + B + G: main comparison, failure subset, computational cost
# ======================================================================================
def main():
    # ---------------- TEST protocol, 5 seeds ----------------
    seed_rows = []
    for s in SEEDS:
        m, ch = run_test(seed=s)
        pm = policy_metrics(ch); pm["seed"] = s
        inc = gate_rows(m)
        pm.update({f"gate_{k}": v for k, v in gate_metrics(inc.fail.to_numpy(), inc.p_gate.to_numpy(), m.tau).items()})
        seed_rows.append(pm)
        if s == 0:
            ch.to_parquet(OUT / "test_proposed.parquet")
    pd.DataFrame(seed_rows).to_csv(TAB / "test_proposed_seeds.csv", index=False)
    te = C[C.split == "test"]
    chs = {"Proposed": pd.read_parquet(OUT / "test_proposed.parquet")}
    chs.update({b: baseline(te, b) for b in BASELINES})
    for lab, cfg in (("Ranker only", dict(mode="ranker")), ("Gate only", dict(mode="gate", incumbent="memory")),
                     ("Proposed, Phase 2 incumbent", dict(incumbent="phase2"))):
        chs[lab] = run_test(**cfg)[1]
    summarize(chs).to_csv(TAB / "main_test.csv", index=False)
    stats_vs(chs).to_csv(TAB / "stats_test.csv", index=False)

    # ---------------- CV protocol, 5 seeds ----------------
    seed_rows = []
    for s in SEEDS:
        ch, g, fits = run_cv(seed=s)
        pm = policy_metrics(ch); pm["seed"] = s
        pm.update({f"gate_{k}": v for k, v in gate_metrics(g.fail.to_numpy(), g.p_gate.to_numpy(), g.tau.to_numpy()).items()})
        pm.update(fit_s=fits.fit_s.mean(), infer_ms=fits.infer_ms.mean())
        seed_rows.append(pm)
        ch.to_parquet(OUT / f"cv_proposed_s{s}.parquet"); g.to_parquet(OUT / f"cv_gate_s{s}.parquet")
        fits.to_csv(OUT / f"cv_fits_s{s}.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(TAB / "cv_proposed_seeds.csv", index=False)
    chs = {"Proposed": pd.read_parquet(OUT / "cv_proposed_s0.parquet")}
    base = cv_baselines()
    chs.update(base)
    for lab, cfg, f in (("Ranker only", dict(mode="ranker"), "ranker_only"),
                        ("Gate only", dict(mode="gate", incumbent="memory"), "gate_only"),
                        ("Proposed, Phase 2 incumbent", dict(incumbent="phase2"), "phase2_incumbent"),
                        ("Proposed, guarded threshold", dict(guard=True), "guarded")):
        chs[lab] = run_cv(**cfg)[0]
        chs[lab].to_parquet(OUT / f"cv_{f}.parquet")
    for name in ("Random", "Epsilon-greedy (0.1)"):  # average stochastic baselines over seeds
        vals = pd.DataFrame([policy_metrics(cv_baselines(s)[name]) for s in SEEDS])
        vals.to_csv(TAB / f"cv_{name.split()[0].lower()}_seeds.csv", index=False)
    S = summarize(chs)
    S.to_csv(TAB / "main_cv.csv", index=False)
    stats_vs(chs).to_csv(TAB / "stats_cv.csv", index=False)

    # gate quality vs alternative gates (same folds, same incumbent = ranker top), incl. simple rules
    gq = []
    for kind, lab in (("lgbm", "LightGBM gate"), ("rf", "Random-forest gate"), ("lr", "Logistic-regression gate"),
                      ("jump", "RTT-jump rule")):
        if kind == "lgbm":
            ch, g = chs["Proposed"], pd.read_parquet(OUT / "cv_gate_s0.parquet")
        else:
            ch, g, _ = run_cv(gate_kind=kind)
        g.to_parquet(OUT / f"cv_gate_{kind}.parquet"); ch.to_parquet(OUT / f"cv_policy_gate_{kind}.parquet")
        gm = gate_metrics(g.fail.to_numpy(), g.p_gate.to_numpy(), g.tau.to_numpy())
        gq.append(dict(gate=lab, **gm, policy_fail_pct=100 * ch.fail.mean(), policy_regret=ch.regret.mean()))
    g0 = pd.read_parquet(OUT / "cv_gate_s0.parquet")
    ctx = C.groupby("decision_id")[["ctx_prev_fail", "ctx_fail_w12"]].first()
    y = g0.fail.to_numpy()
    for col, lab in (("ctx_prev_fail", "Previous-interval-failed rule"), ("ctx_fail_w12", "Recent-failure-count rule")):
        p = g0.decision_id.map(ctx[col]).fillna(0).to_numpy()
        gq.append(dict(gate=lab, auprc=average_precision_score(y, p), prevalence=y.mean(), n_events=int(y.sum())))
    pd.DataFrame(gq).to_csv(TAB / "gate_quality_cv.csv", index=False)

    # B: failure subset (decisions where persistence fails)
    fs = []
    pers = chs["Persistence"].set_index("decision_id")
    lkb = chs["Last-known-best"].set_index("decision_id")
    subsets = {"persistence fails (all)": set(pers.index[pers.fail]),
               "persistence fails: sampling artifact": set(pers.index[pers.fail]) & ARTIFACT,
               "persistence fails: genuine": set(pers.index[pers.fail]) - ARTIFACT,
               "last-known-best fails": set(lkb.index[lkb.fail])}
    for sub, idset in subsets.items():
        for name, ch in chs.items():
            x = ch[ch.decision_id.isin(idset)]
            fs.append(dict(subset=sub, policy=name, n=len(x), fail_pct=100 * x.fail.mean(), regret_mean=x.regret.mean(),
                           regret_p90=x.regret.quantile(.9), avoided=int((~x.fail).sum())))
    pd.DataFrame(fs).to_csv(TAB / "failure_subset_cv.csv", index=False)

    # G: computational cost (fresh fit, no cached rankers)
    fits = pd.read_csv(OUT / "cv_fits_s0.csv")
    clear_cache()
    m = Switcher().fit(C, ids("train"), ids("val"))
    t0 = time.perf_counter(); m.select(C[C.split == "test"]); dt = time.perf_counter() - t0
    nd = C[C.split == "test"].decision_id.nunique()
    import pickle
    cost = dict(fit_seconds_full_train=m.fit_seconds, fit_seconds_cv_mean=fits.fit_s.mean(),
                infer_ms_per_decision=1000 * dt / nd, decision_interval_ms=30 * 60 * 1000,
                model_size_kb=len(pickle.dumps((m.gate.m, m.ranker.m, m.short.m))) / 1024,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                n_train_candidate_rows=int(C.split.eq("train").sum()))
    (TAB / "cost.json").write_text(json.dumps(cost, indent=2))
    print(S[["policy", "n", "fail_pct", "fail_lo", "fail_hi", "regret_mean", "switch_per_pair_day", "fired_pct"]]
          .round(3).to_string())


# ======================================================================================
# Step 10: ablations (CV protocol, all five seeds) and threshold sensitivity
# ======================================================================================
ABLATIONS = {
    "Full model": {},
    "Ranker only (no gate)": dict(mode="ranker"),
    "Gate only (runner-up switch)": dict(mode="gate", incumbent="memory"),
    "Last-known-best incumbent": dict(incumbent="memory"),
    "Phase 2 incumbent (previous-interval best)": dict(incumbent="phase2"),
    "Guarded threshold (min. gain 2 failures)": dict(guard=True),
    "No stale-history blend": dict(blend=False),
    "Blend k = 3": dict(blend_k=3),
    "Blend k = 12": dict(blend_k=12),
    "No class weighting": dict(weighting="none"),
    "SMOTE instead of class weights": dict(weighting="smote"),
    "Gate: random forest": dict(gate_kind="rf"),
    "Gate: logistic regression": dict(gate_kind="lr"),
    "Window <= 3 intervals": dict(feats=features(max_window=3)),
    "Window <= 6 intervals": dict(feats=features(max_window=6)),
    **{f"- {g} features": dict(feats=features(drop_groups=(g,))) for g in GROUPS},
}


def ablation(seeds=SEEDS):
    rows = []
    pers = cv_baselines()["Persistence"]
    for name, cfg in ABLATIONS.items():
        for s in seeds:
            if name == "Full model":
                ch = pd.read_parquet(OUT / f"cv_proposed_s{s}.parquet"); g = pd.read_parquet(OUT / f"cv_gate_s{s}.parquet")
            else:
                ch, g, _ = run_cv(seed=s, **cfg)
            pm = policy_metrics(ch)
            ci = boot_ci(ch, pers)
            pm.update(dfail_lo=ci["dfail_ci"][0], dfail_hi=ci["dfail_ci"][1],
                      auprc=average_precision_score(g.fail, g.p_gate) if g is not None else np.nan,
                      gate_prevalence=g.fail.mean() if g is not None else np.nan)
            rows.append(dict(ablation=name, seed=s, **pm))
            print(name, s, round(pm["fail_pct"], 3), round(pm["regret_mean"], 3), round(pm["auprc"], 3), flush=True)
        pd.DataFrame(rows).to_csv(TAB / "ablations_cv.csv", index=False)

    # sensitivity 1: switching budget
    sens = []
    for budget in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        ch, g, _ = run_cv(budget=budget)
        sens.append(dict(kind="budget", value=budget, **policy_metrics(ch)))
    pd.DataFrame(sens).to_csv(TAB / "sensitivity_budget_cv.csv", index=False)
    label_sensitivity()


def label_sensitivity():
    # sensitivity 2: failure-label thresholds (Phase 2, Step 10: 20 ms / 50 ms margin, loss limit).
    # Each probe has 15 pings, so loss > 5 % = >= 1 lost ping, > 10 % = >= 2, > 20 % = >= 4.
    # The model is retrained and evaluated on the relabelled data; baselines are re-scored.
    lab = []
    for ms, loss in LABEL_GRID:
        D = relabel(C, ms, loss)
        ch, g, _ = run_cv(data=D)
        b = cv_baselines(data=D, loss_limit=loss)  # oracle filters at the label's loss limit
        row = dict(margin_ms=ms, loss_pct=loss, min_lost_pings=int(np.floor(loss / 100 * PINGS_PER_PROBE)) + 1,
                   **{f"proposed_{k}": v for k, v in policy_metrics(ch).items()},
                   gate_auprc=average_precision_score(g.fail, g.p_gate), gate_prevalence=g.fail.mean(),
                   gate_events=int(g.fail.sum()))
        for bn, short in (("Persistence", "pers"), ("Last-known-best", "lkb"), ("EMA-6 RTT", "ema"), ("Oracle*", "oracle")):
            bm = policy_metrics(b[bn]); row.update({f"{short}_fail_pct": bm["fail_pct"], f"{short}_n_fail": bm["n_fail"]})
        bc = boot_ci(ch, b["Persistence"])
        row.update(dfail_pers_lo=bc["dfail_ci"][0], dfail_pers_hi=bc["dfail_ci"][1])
        lab.append(row)
        print("label", ms, loss, round(row["proposed_fail_pct"], 3), round(row["pers_fail_pct"], 3), flush=True)
    pd.DataFrame(lab).to_csv(TAB / "sensitivity_labels_cv.csv", index=False)


# ======================================================================================
# C, D, E, F: returning paths, drift, cross-vantage transfer, QoE profiles
# ======================================================================================
def runage_table(pols, gate):
    """Phase 2, Step 8: all metrics stratified by run age. Strata = age of the Phase 2 incumbent's
    current presence run, so every policy is scored on the same decisions in each stratum."""
    age = C[C.is_inc].set_index("decision_id").runage_now_h
    bins, labels = [-1, 2, 6, 24, 1e9], ["<2 h", "2-6 h", "6-24 h", ">=24 h"]
    rows = []
    for name, ch in pols.items():
        x = with_switch_flag(ch).assign(age=lambda d: pd.cut(d.decision_id.map(age), bins, labels=labels))
        for a, g in x.groupby("age", observed=True):
            r = dict(policy=name, run_age=a, n=len(g), n_fail=int(g.fail.sum()), fail_pct=100 * g.fail.mean(),
                     fail50_pct=100 * g.fail50.mean(), regret_mean=g.regret.mean(), regret_p90=g.regret.quantile(.9),
                     regret_p99=g.regret.quantile(.99), switch_per_dec=g.switched.mean(),
                     switch_per_pair_day=48 * g.switched.mean(),
                     fired_pct=100 * g.fired.mean() if "fired" in g else np.nan)
            if name == "Proposed":
                gg = gate[gate.decision_id.isin(g.decision_id)]
                r.update({f"gate_{k}": v for k, v in gate_metrics(gg.fail, gg.p_gate, gg.tau.to_numpy()).items()})
            rows.append(r)
    return pd.DataFrame(rows)


def robust():
    prop = pd.read_parquet(OUT / "cv_proposed_s0.parquet")
    gate0 = pd.read_parquet(OUT / "cv_gate_s0.parquet")
    base = cv_baselines()
    noblend = run_cv(blend=False)[0]
    noblend.to_parquet(OUT / "cv_noblend.parquet")

    # ---- C1: run-age strata, all metrics ----
    runage_table({"Persistence": base["Persistence"], "Last-known-best": base["Last-known-best"], "Proposed": prop,
                  "Proposed, no blend": noblend, "EMA-6 RTT": base["EMA-6 RTT"], "Oracle*": base["Oracle*"]},
                 gate0).to_csv(TAB / "runage_cv.csv", index=False)

    # ---- C2: synthetic cold start: 20 % of each pair's paths removed from training, 5 random draws ----
    # Three evaluation subsets per draw (pooled over draws):
    #   best-held-out : the best path at t (lowest RTT, loss <= 5 %) is a held-out path
    #   lkb-held-out  : the last-known-best path is a held-out path
    #   any-held-out  : at least one held-out path is a candidate
    oracle = base["Oracle*"].set_index("decision_id").fp
    lkbfp = C[C.is_inc_mem].set_index("decision_id").fp
    pair_of = C.groupby("decision_id").pair.first()
    key_all = pd.Series(list(zip(C.pair, C.fp)), index=C.index)
    cold = []
    for draw in SEEDS:
        rng = np.random.default_rng(100 + draw)
        held = set()
        for p, g in C.groupby("pair"):
            fps = sorted(g.fp.unique())
            held |= {(p, f) for f in rng.choice(fps, max(1, len(fps) // 5), replace=False)}
        hmask = key_all.isin(held)
        subs = {"best path is held out": {d for d in oracle.index if (pair_of[d], oracle[d]) in held},
                "last-known-best is held out": {d for d in lkbfp.index if (pair_of[d], lkbfp[d]) in held},
                "any candidate is held out": set(C.loc[hmask, "decision_id"])}
        runs = {}
        for blend in (True, False):
            runs[("Proposed" + ("" if blend else ", no blend"), "held-out paths unseen in training")] = \
                run_cv(blend=blend, train_mask=lambda df: hmask.loc[df.index])[0]
            runs[("Proposed" + ("" if blend else ", no blend"), "trained on all paths")] = prop if blend else noblend
        for b in ("Persistence", "Last-known-best", "EMA-6 RTT", "Oracle*"):
            runs[(b, "no training")] = base[b]
        for sub, sid in subs.items():
            for (pol, trn), ch in runs.items():
                x = ch[ch.decision_id.isin(sid)]
                cold.append(dict(draw=draw, subset=sub, policy=pol, training=trn, n=len(x), n_fail=int(x.fail.sum()),
                                 regret_sum=x.regret.sum(), picked_heldout=int(sum((p, f) in held for p, f in zip(x.pair, x.fp)))))
        print("cold-start draw", draw, flush=True)
    cd = pd.DataFrame(cold)
    cd.to_csv(TAB / "coldstart_draws_cv.csv", index=False)
    agg = cd.groupby(["subset", "policy", "training"], sort=False)[["n", "n_fail", "regret_sum", "picked_heldout"]].sum()
    agg["fail_pct"] = 100 * agg.n_fail / agg.n
    agg["regret_mean"] = agg.regret_sum / agg.n
    agg["picked_heldout_pct"] = 100 * agg.picked_heldout / agg.n
    agg.drop(columns=["regret_sum"]).reset_index().to_csv(TAB / "coldstart_cv.csv", index=False)

    # ---- D: drift. Calendar weeks from 16 Jul; week 4 = days 21-23 only. Days 24-26 (from 9 Aug) are a
    # separate segment: the Switzerland node is gone and only the two EU<->NA pairs remain. ----
    t0 = C.slot.min().floor("D")
    day = (C.slot - t0).dt.days
    seg = pd.Series(np.select([day < 7, day < 14, day < 21, day < 24], ["week 1", "week 2", "week 3", "week 4 (days 21-23)"],
                              "days 24-26 (CH offline, EU<->NA only)"), index=C.index)
    C["segment"] = seg
    order = ["week 1", "week 2", "week 3", "week 4 (days 21-23)", "days 24-26 (CH offline, EU<->NA only)"]
    seg_ids = {s: C.loc[C.segment == s, "decision_id"].unique() for s in order}
    dstart = C.groupby("decision_id")[["pair", "t", "segment"]].first()

    def before(s):  # training decisions strictly before segment s, minus an embargo per pair
        start = dstart[dstart.segment == s].groupby("pair").t.min()
        prev = dstart[dstart.segment.isin(order[:order.index(s)])]
        cut = prev.pair.map(start).fillna(np.inf)
        return prev.index[prev.t < cut - EMBARGO]

    static = Switcher().fit(C, seg_ids["week 1"])
    drift = []
    for s in order[1:]:
        te = C[C.decision_id.isin(seg_ids[s])]
        lkb = baseline(te, "Last-known-best")
        pers = baseline(te, "Persistence")
        retr = Switcher().fit(C, before(s))
        for lab, m in (("static (week 1)", static), ("retrained on all earlier data", retr)):
            ch = m.select(te); g = gate_rows(m)
            gm = gate_metrics(g.fail, g.p_gate, m.tau)
            drift.append(dict(segment=s, model=lab, n=len(ch), n_pairs=te.pair.nunique(),
                              cand_mean=te.groupby("decision_id").n_cand.first().mean(), n_events=gm["n_events"],
                              auprc=gm["auprc"], prevalence=gm["prevalence"], fail_pct=100 * ch.fail.mean(),
                              lkb_fail_pct=100 * lkb.fail.mean(), pers_fail_pct=100 * pers.fail.mean(),
                              regret_mean=ch.regret.mean(), lkb_regret=lkb.regret.mean(), fired_pct=100 * ch.fired.mean()))
    pd.DataFrame(drift).to_csv(TAB / "drift.csv", index=False)

    # ---- E: leave-one-vantage-out (time-respecting: fold k trains on blocks < k). Test set: decisions
    # whose source is v. Two training restrictions:
    #   source held out  : drop training decisions whose source is v (Phase 2 Step 7 wording)
    #   vantage unseen   : drop every training decision that involves v as source or destination,
    #                      so v never appears in training at all ----
    lovo = []
    for v in ("EU", "CH", "NA"):
        tgt = set(C.loc[C.src_v == v, "decision_id"])
        runs = {"in-vantage (all pairs)": (prop, gate0)}
        ch_s, g_s, _ = run_cv(train_mask=lambda df, v=v: df.src_v == v)
        ch_u, g_u, _ = run_cv(train_mask=lambda df, v=v: (df.src_v == v) | (df.dst_v == v))
        runs["source held out"] = (ch_s, g_s); runs["vantage unseen (src and dst)"] = (ch_u, g_u)
        pb = base["Last-known-best"]; pb = pb[pb.decision_id.isin(tgt)]
        for lab, (ch, g) in runs.items():
            x = ch[ch.decision_id.isin(tgt)]; gg = g[g.decision_id.isin(tgt)]
            lovo.append(dict(held_out=v, model=lab, n=len(x), n_events=int(gg.fail.sum()),
                             auprc=average_precision_score(gg.fail, gg.p_gate) if gg.fail.any() else np.nan,
                             prevalence=gg.fail.mean(), fail_pct=100 * x.fail.mean(), lkb_fail_pct=100 * pb.fail.mean(),
                             regret_mean=x.regret.mean(), lkb_regret=pb.regret.mean()))
    pd.DataFrame(lovo).to_csv(TAB / "lovo.csv", index=False)

    # ---- F: QoE profile satisfaction (bandwidth criterion dropped, see audit) ----
    ev = C[C.decision_id.isin(prop.decision_id)]
    # relative thresholds: quantiles of each pair's best-path RTT in the training split
    train_rtt = C[C.split == "train"].groupby(["pair", "decision_id"]).rtt_t.min().groupby("pair")
    profiles = {f"{k} (Phase 2: {r} ms, {l} %)": ("abs", r, l, "phase2") for k, (r, l) in QOE_PHASE2.items()}
    profiles.update({f"{k} (preprint: {r} ms, {l} %)": ("abs", r, l, "preprint") for k, (r, l) in QOE_PREPRINT.items()})
    profiles.update({"video_conf, relative (pair best-RTT q75, 7 %)": ("rel", 0.75, 7, "relative"),
                     "gaming, relative (pair best-RTT q50, 4 %)": ("rel", 0.50, 4, "relative")})
    pols = {"Proposed": prop, **{b: base[b] for b in BASELINES}}
    q = []
    for pname, (kind, r, l, src) in profiles.items():
        thr = (ev.pair.map(train_rtt.quantile(r)) if kind == "rel" else pd.Series(r, index=ev.index))
        sat = (ev.rtt_t <= thr) & (ev.loss_t <= l)
        ceiling = sat.groupby(ev.decision_id).any()
        satmap = pd.Series(sat.to_numpy(), index=pd.MultiIndex.from_arrays([ev.decision_id, ev.fp]))
        for name, ch in pols.items():
            s = satmap.reindex(pd.MultiIndex.from_arrays([ch.decision_id, ch.fp])).fillna(False).to_numpy()
            q.append(dict(profile=pname, source=src, policy=name, satisfaction_pct=100 * s.mean(),
                          oracle_ceiling_pct=100 * ceiling.mean(),
                          share_of_ceiling=(s.mean() / ceiling.mean()) if ceiling.mean() > 0 else np.nan))
    pd.DataFrame(q).to_csv(TAB / "qoe_cv.csv", index=False)


# ======================================================================================
# Step 11: error analysis, SHAP and permutation importance, failure taxonomy
# ======================================================================================
def taxonomy(prop, gate):
    """Taxonomy over *all* decisions of the final policy (ranker top + gate), not only gate firings.
    Part 1 - gate outcome per decision (mutually exclusive). Part 2 - cause of every failed final
    decision (mutually exclusive, first matching rule). Part 3 - overlapping attributes."""
    p = prop.set_index("decision_id")
    g = gate.set_index("decision_id")
    j = p[["fail", "regret", "loss_t", "dead", "fired", "fp", "obs_in_run", "since_obs"]].join(
        g[["fail", "fp", "p_gate", "tau"]], rsuffix="_inc")
    inc_f, fired, fail = j.fail_inc.astype(bool), j.fired.astype(bool), j.fail.astype(bool)
    part1 = {
        "incumbent ok, gate silent (correct keep)": int((~inc_f & ~fired).sum()),
        "incumbent fails, gate silent (missed / late detection)": int((inc_f & ~fired).sum()),
        "incumbent fails, gate fired, switch avoided the failure": int((inc_f & fired & ~fail).sum()),
        "incumbent fails, gate fired, alternative also failed": int((inc_f & fired & fail).sum()),
        "false alarm, harmless switch": int((~inc_f & fired & ~fail).sum()),
        "false alarm, switch caused a failure": int((~inc_f & fired & fail).sum()),
    }
    one_ping = 100 / PINGS_PER_PROBE + 0.5
    f = j[fail].copy()
    rules = [("dead path chosen (100 % loss)", f.dead.astype(bool)),
             ("stale-history pick (chosen path < 12 observed intervals in run)", f.obs_in_run < STALE_OBS),
             ("probe-coverage gap (chosen path not observed at t-1)", f.since_obs.fillna(99) > 0),
             ("RTT ranking error (regret > 20 ms)", f.regret > 20),
             ("single lost ping (1 of 15, regret <= 20 ms)", f.loss_t <= one_ping),
             ("loss burst (>= 2 lost pings, regret <= 20 ms)", pd.Series(True, index=f.index))]
    cause, left = {}, pd.Series(True, index=f.index)
    for name, m in rules:
        m = m.astype(bool) & left
        cause[name] = int(m.sum()); left &= ~m
    # was any non-failing candidate available? (if not, the failure was unavoidable for any policy)
    ok_alt = C[C.decision_id.isin(f.index)].groupby("decision_id").fail.apply(lambda s: (~s).any())
    part3 = {"failures with no non-failing candidate (unavoidable)": int((~ok_alt.reindex(f.index).fillna(False)).sum()),
             "failures in decisions where the gate fired": int(f.fired.sum()),
             "failures that are loss-only (regret <= 20 ms)": int((f.regret <= 20).sum()),
             "failures that are single lost pings": int(((f.regret <= 20) & (f.loss_t <= one_ping)).sum()),
             "stale chosen path (any failure)": int((f.obs_in_run < STALE_OBS).sum()),
             "chosen path not observed at t-1 (any failure)": int((f.since_obs.fillna(99) > 0).sum())}
    return dict(n_decisions=len(j), n_failures=int(fail.sum()), gate_outcome=part1, failure_cause=cause,
                failure_attributes=part3,
                decisions_not_evaluable=int((~C.groupby("decision_id").inc_probed.first()).sum())), j


def error():
    prop = pd.read_parquet(OUT / "cv_proposed_s0.parquet")
    gate = pd.read_parquet(OUT / "cv_gate_s0.parquet")
    pers = cv_baselines()["Persistence"].set_index("decision_id")
    tax, j = taxonomy(prop, gate)
    (TAB / "failure_taxonomy_cv.json").write_text(json.dumps(tax, indent=2))

    # breakdown by vantage point, destination, pair, hour, run age, severity (Phase 2, Step 11)
    d = C.groupby("decision_id")[["src_v", "dst_v", "pair", "slot"]].first()
    age = C[C.is_inc].set_index("decision_id").runage_now_h
    k = j.join(d).join(pers[["fail", "regret"]], rsuffix="_pers")
    k["hour_bin"] = pd.cut(k.slot.dt.hour, [-1, 5, 11, 17, 23], labels=["00-06", "06-12", "12-18", "18-24"])
    k["run_age"] = pd.cut(k.index.map(age), [-1, 2, 6, 24, 1e9], labels=["<2 h", "2-6 h", "6-24 h", ">=24 h"])
    k["severity"] = pd.cut(k.regret_pers, [-1, 20, 50, 100, 1e9], labels=["<=20 ms", "20-50", "50-100", ">100"])
    by = []
    for col in ("src_v", "dst_v", "pair", "hour_bin", "run_age", "severity"):
        for key, g in k.groupby(col, observed=True):
            by.append(dict(by=col, group=str(key), n=len(g), pers_fail_pct=100 * g.fail_pers.mean(),
                           prop_fail_pct=100 * g.fail.mean(), pers_regret=g.regret_pers.mean(), prop_regret=g.regret.mean(),
                           prop_n_fail=int(g.fail.sum()), fired_pct=100 * g.fired.mean()))
    pd.DataFrame(by).to_csv(TAB / "error_breakdown_cv.csv", index=False)
    worst = j[j.fail].join(d).sort_values("regret", ascending=False).head(15)
    worst.to_csv(TAB / "worst_decisions_cv.csv")

    # SHAP + permutation importance for the final gate (incumbent = ranker top) of the last CV fold
    import shap
    k_, tr, te = FOLDS[-1]
    m = Switcher().fit(C, tr)
    m.select(C[C.decision_id.isin(te)])
    X = gate_rows(m)
    sv = shap.TreeExplainer(m.gate.m).shap_values(X[ALL_FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": ALL_FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
    grp = {f: g for g, fs in GROUPS.items() for f in fs}
    imp["group"] = imp.feature.map(grp)
    pi = permutation_importance(m.gate.m, X[ALL_FEATURES], X.fail.astype(int), scoring="average_precision",
                                n_repeats=20, random_state=0)
    imp["perm_auprc_drop"] = pi.importances_mean; imp["perm_sd"] = pi.importances_std
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(TAB / "importance_gate.csv", index=False)
    np.save(OUT / "shap_values.npy", sv); X[ALL_FEATURES].to_parquet(OUT / "shap_X.parquet")
    # shortcut audit: a gate that only sees time-of-day / day-of-week, trained on the same rows
    Gt = Gate(feats=GROUPS["temporal"]).fit(C.loc[m.gate_train_index], C.loc[m.gate_val_index])
    sc = dict(full_gate_auprc=average_precision_score(X.fail, m.gate.predict(X)),
              temporal_only_auprc=average_precision_score(X.fail, Gt.predict(X)), prevalence=X.fail.mean(),
              n_events=int(X.fail.sum()))
    (TAB / "shortcut_audit.json").write_text(json.dumps(sc, indent=2))
    print(json.dumps(tax, indent=2)); print(imp.sort_values("mean_abs_shap", ascending=False).head(12).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for step, fn in (("main", main), ("ablation", ablation), ("robust", robust), ("error", error), ("labels", label_sensitivity)):
        if what in (step, "all"):
            t0 = time.time(); fn(); print(f"[{step}] done in {time.time() - t0:.0f} s", flush=True)
