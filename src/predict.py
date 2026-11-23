"""Inference with a saved switching model (Step 12).

Loads a pickled core.Switcher (written by train_final.py) and, for every decision in a candidates
table (built by build_panel.py from raw ScionPathML JSON), outputs the path to use in the next
30-minute interval, the incumbent (ranker top) and the gate's failure probability for it.
If the table carries ground truth (rtt_t, loss_t, fail), the realized metrics are printed too.

Examples:
  python predict.py --model ../models/switcher_trainsplit.pkl --split test
  python predict.py --model ../models/switcher_alldata.pkl --pair EU->NA --last 5
  python predict.py --model ../models/switcher_alldata.pkl --candidates other_candidates.parquet --out choices.csv
"""
import argparse
import pickle
import time

import pandas as pd

from config import PROC
from core import chrono_split, gate_metrics, load, policy_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/switcher_trainsplit.pkl")
    ap.add_argument("--candidates", default=None, help="candidates parquet (default: data/processed/candidates.parquet)")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--pair", default=None, help="e.g. EU->NA")
    ap.add_argument("--last", type=int, default=None, help="only the last N decisions per pair")
    ap.add_argument("--out", default=None, help="write choices to this CSV")
    a = ap.parse_args()

    m = pickle.loads(open(a.model, "rb").read())
    if a.candidates:
        C = pd.read_parquet(a.candidates)
        C["day"] = C.slot.dt.floor("D")
    else:
        C = load()
    if a.split != "all":
        C = C[chrono_split(C) == a.split]
    if a.pair:
        C = C[C.pair == a.pair]
    if a.last:
        keep = C.groupby("pair").t.transform(lambda t: t >= sorted(t.unique())[-a.last])
        C = C[keep]

    t0 = time.perf_counter()
    out = m.decide(C)
    ms = (time.perf_counter() - t0) * 1000 / max(len(out), 1)
    cols = ["pair", "slot", "decision_id", "fp", "incumbent_fp", "p_gate", "fired"]
    show = out[cols].rename(columns={"fp": "chosen_fp"}).sort_values(["pair", "slot"])
    print(f"{len(show)} decisions, threshold tau = {m.tau:.4f}, gate fired on {show.fired.sum()}, "
          f"{ms:.3f} ms per decision")
    print(show.tail(12).to_string(index=False))
    if a.out:
        show.to_csv(a.out, index=False)
        print("written:", a.out)

    if "fail" in C:  # ground truth available: score on the Phase 2 evaluable decisions
        ch = m.select(C)
        s = m.scored
        g = s[s[m.inc] & s.decision_id.isin(C.loc[C.is_inc, "decision_id"])]
        pm = policy_metrics(ch)
        gm = gate_metrics(g.fail, g.p_gate, m.tau)
        print(f"evaluable decisions {pm['n']}: failure share {pm['fail_pct']:.3f} %, mean regret "
              f"{pm['regret_mean']:.3f} ms, switches per pair-day {pm['switch_per_pair_day']:.3f}, "
              f"gate AUPRC {gm['auprc']:.3f} (prevalence {gm['prevalence']:.3f})")


if __name__ == "__main__":
    main()
