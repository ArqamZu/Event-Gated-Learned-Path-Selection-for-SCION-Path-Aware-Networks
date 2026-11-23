"""Step 12 - train and save the final event-gated switching model.

Two models are written to models/:
  switcher_trainsplit.pkl  fitted on the 70 % training split, threshold tuned on the 15 % validation
                           split. predict.py reproduces the reported TEST-protocol numbers with it.
  switcher_alldata.pkl     fitted on all decisions (threshold on the last 15 % of each pair), the
                           model one would deploy.
Each pickle holds the core.Switcher object (LightGBM gate + LambdaRank rankers + threshold). The
LightGBM boosters are also exported as plain text, and models/model_card.json records the features,
threshold, hyperparameters, training range, dataset provenance and package versions.

Usage:  python train_final.py [--seed 0]
"""
import argparse
import hashlib
import json
import pickle
import platform

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from config import MODELS, PROC, SWITCH_BUDGET
from core import GATE_PARAMS, RANK_PARAMS, Switcher, chrono_split, gate_metrics, load, policy_metrics


def save(m, name, C, train_ids, note):
    m.scored = None
    path = MODELS / f"{name}.pkl"
    path.write_bytes(pickle.dumps(m))
    m.gate.m.booster_.save_model(str(MODELS / f"{name}_gate.txt"))
    m.ranker.m.booster_.save_model(str(MODELS / f"{name}_ranker_full.txt"))
    if m.short is not None:
        m.short.m.booster_.save_model(str(MODELS / f"{name}_ranker_short.txt"))
    tr = C[C.decision_id.isin(train_ids)]
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), note=note,
                tau=float(m.tau), switch_budget=SWITCH_BUDGET, blend_k=m.blend_k, incumbent=m.incumbent,
                n_train_decisions=int(tr.decision_id.nunique()),
                train_range=[str(tr.slot.min()), str(tr.slot.max())],
                gate_trees=int(m.gate.m.booster_.num_trees()), ranker_trees=int(m.ranker.m.booster_.num_trees()),
                fit_seconds=round(m.fit_seconds, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    C = load()
    C["split"] = chrono_split(C)
    ids = {s: C.loc[C.split == s, "decision_id"].unique() for s in ("train", "val", "test")}

    m = Switcher(seed=seed).fit(C, ids["train"], ids["val"])
    te = C[C.split == "test"]
    ch = m.select(te)
    s = m.scored
    g = s[s._inc_rank & s.decision_id.isin(te.loc[te.is_inc, "decision_id"])]
    test = {**{k: float(v) for k, v in policy_metrics(ch).items()},
            **{f"gate_{k}": float(v) for k, v in gate_metrics(g.fail, g.p_gate, m.tau).items()}}
    card_split = save(m, "switcher_trainsplit", C, ids["train"],
                      "fit on the 70 % training split, threshold on validation; evaluated on the 15 % test split")
    card_split["test_metrics"] = test

    m_all = Switcher(seed=seed).fit(C, C.decision_id.unique())
    card_all = save(m_all, "switcher_alldata", C, C.decision_id.unique(),
                    "fit on all decisions (threshold on the last 15 % of each pair); for deployment")

    prov = json.loads((PROC / "provenance.json").read_text())
    card = dict(model="event-gated SCION path switcher (LightGBM failure gate + LambdaRank ranker, stale-history blend)",
                seed=seed, features=m.feats, short_ranker_features=m.short.feats if m.short is not None else None,
                gate_params=GATE_PARAMS, ranker_params=RANK_PARAMS,
                models=[card_split, card_all],
                dataset=dict(repo=prov["dataset_repo"], commit=prov["dataset_commit"], parser_version=prov["parser_version"]),
                versions=dict(python=platform.python_version(), lightgbm=lightgbm.__version__,
                              sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__),
                usage="python predict.py --model ../models/switcher_trainsplit.pkl --split test")
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    print(json.dumps({k: round(v, 3) for k, v in test.items()}, indent=1))
    print("saved:", card_split["file"], card_all["file"])


if __name__ == "__main__":
    main()
