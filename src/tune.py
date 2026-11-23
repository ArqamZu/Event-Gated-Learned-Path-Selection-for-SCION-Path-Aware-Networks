"""Hyperparameter tuning with purged expanding-window time-series CV (training data only).

Only folds whose test block lies inside the training portion (first 70 % of every pair)
are used, so the held-out test split is never touched during tuning.

Two stages, each on the target the component is used for in the final configuration:
  1. ranker: NDCG@1 of the LambdaRank model over the candidates of each decision;
  2. gate:   AUPRC for the failure of the *ranker's top path* (the final gate's incumbent). The
             ranker uses the stage-1 parameters; its top path on the training rows is cross-fitted
             exactly as in core.Switcher.fit, and on the fold's test rows it is the fitted ranker's top.
Writes results/tables/tuning.json with the selected gate and ranker parameters.
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score

from config import TAB
from core import Ranker, Switcher, chrono_split, inner_val, load, purged_folds

C = load()
C["split"] = chrono_split(C)
T = C[C.split == "train"].copy()
folds = list(purged_folds(T, n_blocks=5))
EVAL = set(T.loc[T.is_inc, "decision_id"])  # Phase 2 evaluable decisions

grid = [dict(num_leaves=nl, min_child_samples=mc, learning_rate=lr)
        for nl, mc, lr in itertools.product([7, 15, 31], [20, 50, 100], [0.03, 0.1])]
keys = ["num_leaves", "min_child_samples", "learning_rate"]


def clean(d):
    return {k: (int(d[k]) if k != "learning_rate" else float(d[k])) for k in keys}


# ---- stage 1: ranker ---------------------------------------------------------------------
res_r = []
for prm in grid:
    nd = []
    for k, tr, te in folds:
        a, b = inner_val(tr, T)
        r = Ranker(params=prm).fit(T[T.decision_id.isin(a)], T[T.decision_id.isin(b)])
        e = T[T.decision_id.isin(te) & (T.n_cand >= 2)].assign(s=lambda d: r.predict(d))
        nd.append(np.mean([ndcg_score([g_.rel.to_numpy()], [g_.s.to_numpy()], k=1) for _, g_ in e.groupby("decision_id")]))
    res_r.append({**prm, "oof_ndcg1": float(np.mean(nd))})
    print("ranker", prm, f"NDCG@1={np.mean(nd):.4f}")
R = pd.DataFrame(res_r).sort_values("oof_ndcg1", ascending=False)
best_r = clean(R.iloc[0])

# ---- stage 2: gate, target = failure of the ranker's top path -------------------------------
res_g = []
for prm in grid:
    ys, ps = [], []
    for k, tr, te in folds:
        m = Switcher(rank_params=best_r, gate_params=prm).fit(T, tr)
        s = m.score(T[T.decision_id.isin(te)])
        x = s[s._inc_rank & s.decision_id.isin(EVAL)]
        ys.append(x.fail.to_numpy()); ps.append(x.p_gate.to_numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    res_g.append({**prm, "oof_auprc": average_precision_score(y, p), "prevalence": y.mean(), "n_events": int(y.sum())})
    print("gate", prm, f"AUPRC={res_g[-1]['oof_auprc']:.4f} (prevalence {y.mean():.4f}, {int(y.sum())} events)")
G = pd.DataFrame(res_g).sort_values("oof_auprc", ascending=False)

out = dict(gate=clean(G.iloc[0]), ranker=best_r, gate_grid=G.to_dict("records"), ranker_grid=R.to_dict("records"),
           gate_target="failure of the ranker's top path (cross-fitted on training rows), Phase 2 evaluable decisions",
           ranker_target="NDCG@1 of graded next-interval relevance",
           note="purged expanding-window CV (5 blocks, 4 folds) inside the 70% training portion")
(TAB / "tuning.json").write_text(json.dumps(out, indent=2))
print("selected:", out["gate"], out["ranker"])
