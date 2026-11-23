"""Phase 2 design analysed in more depth: out-of-fold gate scores and the full threshold sweep
(switching budget vs failure share) on the pooled purged-CV folds."""
import numpy as np
import pandas as pd

from config import TAB
from experiments import C, EV, FOLDS, OUT
from core import Switcher

scored = []
for k, tr, te in FOLDS:
    m = Switcher(incumbent="phase2").fit(C, tr)
    s = m.score(C[C.decision_id.isin(te)]).assign(fold=k)
    scored.append((m, s))
g = pd.concat([s[s.is_inc & s.decision_id.isin(EV)][["decision_id", "pair", "fail", "fail50", "p_gate"]] for _, s in scored])
g.to_parquet(OUT / "cv_gate_phase2.parquet")

taus = np.unique(np.quantile(g.p_gate, np.r_[np.linspace(0, 0.9, 46), np.linspace(0.9, 1, 51)]))
rows = []
for tau in list(taus) + [np.inf]:
    ch = pd.concat([m.choose(s, tau, m.mode) for m, s in scored])
    rows.append(dict(tau=tau, fired=ch.fired.mean(), fail_pct=100 * ch.fail.mean(), regret_mean=ch.regret.mean()))
pd.DataFrame(rows).sort_values("fired").to_csv(TAB / "threshold_sweep_phase2.csv", index=False)
print(pd.DataFrame(rows).sort_values("fired").iloc[::10].round(3).to_string())

# SHAP for the Phase 2 gate (the variant whose gate is predictive), on the CV fold with most failure events
import shap
from core import ALL_FEATURES, GROUPS
m, s = max(scored, key=lambda ms: ms[1][ms[1].is_inc & ms[1].decision_id.isin(EV)].fail.sum())
print("SHAP fold", int(s.fold.iloc[0]), "events", int(s[s.is_inc & s.decision_id.isin(EV)].fail.sum()))
X = s[s.is_inc & s.decision_id.isin(EV)]
sv = shap.TreeExplainer(m.gate.m).shap_values(X[ALL_FEATURES])
sv = sv[1] if isinstance(sv, list) else sv
grp = {f: g for g, fs in GROUPS.items() for f in fs}
imp = pd.DataFrame({"feature": ALL_FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
imp["group"] = imp.feature.map(grp)
from sklearn.inspection import permutation_importance
pi = permutation_importance(m.gate.m, X[ALL_FEATURES], X.fail.astype(int), scoring="average_precision", n_repeats=20, random_state=0)
imp["perm_auprc_drop"] = pi.importances_mean; imp["perm_sd"] = pi.importances_std
imp.sort_values("mean_abs_shap", ascending=False).to_csv(TAB / "importance_gate_phase2.csv", index=False)
print(imp.sort_values("mean_abs_shap", ascending=False).head(12).round(4).to_string())
print(imp.groupby("group").mean_abs_shap.sum().sort_values(ascending=False).round(3))

# Shortcut audit of the Phase 2 gate: does AUPRC survive without temporal / volatility features?
import json
from sklearn.metrics import average_precision_score as _ap
from experiments import run_cv
from core import features
sc = {}
for lab, fe in [("all features", None), ("without temporal", features(drop_groups=("temporal",))),
                ("without temporal and volatility", features(drop_groups=("temporal", "volatility"))),
                ("history only (rolling+current)", GROUPS["current"] + GROUPS["rolling"])]:
    ch, gg, _ = run_cv(incumbent="phase2", feats=fe)
    sc[lab] = dict(auprc=round(float(_ap(gg.fail, gg.p_gate)), 3), prevalence=round(float(gg.fail.mean()), 3),
                   policy_fail_pct=round(float(100 * ch.fail.mean()), 3), regret=round(float(ch.regret.mean()), 3))
    print(lab, sc[lab])
(TAB / "shortcut_phase2_gate.json").write_text(json.dumps(sc, indent=2))

# One-step RTT forecast references (context for the ScionPathML forecasting task)
from sklearn.linear_model import LinearRegression
from experiments import C
x = C[C.since_obs == 0]
fc = {}
for col, lab in [("last_rtt", "last value (naive)"), ("rtt_ema6", "EMA-6"), ("rtt_mean_w12", "mean of last 12 intervals")]:
    e = (x.rtt_t - x[col]).abs()
    fc[lab] = dict(mae=round(float(e.mean()), 3), median_ae=round(float(e.median()), 3), n=int(e.notna().sum()))
fl = ["last_rtt", "rtt_mean_w3", "rtt_mean_w6", "rtt_mean_w12", "rtt_ema6", "rtt_min_w12", "rtt_max_w12"]
d = x.dropna(subset=fl + ["rtt_t"]); cut = d.slot.quantile(0.7)
tr, te = d[d.slot < cut], d[d.slot >= cut]
e = (te.rtt_t - LinearRegression().fit(tr[fl], tr.rtt_t).predict(te[fl])).abs()
fc["linear regression on window statistics (70/30 time split)"] = dict(mae=round(float(e.mean()), 3), median_ae=round(float(e.median()), 3), n=len(te))
e2 = (te.rtt_t - te.last_rtt).abs()
fc["last value on the same 30 % test rows"] = dict(mae=round(float(e2.mean()), 3), median_ae=round(float(e2.median()), 3), n=len(te))
(TAB / "forecast_reference.json").write_text(json.dumps(fc, indent=2))
print(json.dumps(fc, indent=2))
