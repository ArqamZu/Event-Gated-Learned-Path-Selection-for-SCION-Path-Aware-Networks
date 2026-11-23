"""Data audit: re-derives every quantitative dataset claim made in Phase 2 from the raw tables.

Writes results/tables/audit.json and prints a Phase-2-vs-Phase-3 comparison.
"""
import json

import numpy as np
import pandas as pd

from config import FAIL_LOSS, PINGS_PER_PROBE, PROC, QOE_PHASE2, QOE_PREPRINT, TAB, VANTAGE

pings = pd.read_parquet(PROC / "pings.parquet")
sets = pd.read_parquet(PROC / "pathsets.parquet")
bw = pd.read_parquet(PROC / "bandwidth.parquet")
C = pd.read_parquet(PROC / "candidates.parquet")
A = {}

# ---- provenance (Phase 2, Step 2): dataset commit, archive hashes, parser version ------------
prov = json.loads((PROC / "provenance.json").read_text())
A["dataset_repo"] = prov["dataset_repo"]
A["dataset_commit"] = prov["dataset_commit"]
A["archives_added_in_commit"] = prov["archives_added_in_commit"]
A["archives_match_commit"] = prov["all_archives_match"]
A["archive_sha256"] = {k: v["sha256"] for k, v in prov["archives"].items()}
A["parser"] = prov["parser"]
A["parser_version"] = prov["parser_version"]
A["parser_sha256"] = prov["parser_sha256"]
A["table_row_counts"] = prov["row_counts"]

# ---- volume --------------------------------------------------------------------------
A["ping_rows"] = len(pings)
A["ping_status"] = pings.status.value_counts().to_dict()
A["ping_ok_but_total_loss"] = int(((pings.status == "ok") & pings.rtt_avg.isna()).sum())
A["dead_path_candidate_rows"] = int(C.dead.sum())
A["dead_path_incumbents"] = int((C.dead & C.is_inc).sum())
okp = pings[pings.status == "ok"]
A["pings_per_probe"] = sorted(okp.sent.dropna().astype(int).unique().tolist())
A["loss_values_pct_top"] = okp.loss.value_counts().sort_index().head(8).to_dict()
A["loss_quantum_pct"] = round(100 / PINGS_PER_PROBE, 2)
A["hop_count_equals_2x_links"] = bool((okp.n_hop_ifaces == 2 * (okp.n_as - 1)).all())
A["hop_count_range"] = [int(pings.hop_count.min()), int(pings.hop_count.max())]
A["skipped_share_per_lab"] = pings.groupby("lab").status.apply(lambda s: round((s == "skipped").mean() * 100, 2)).to_dict()
A["paths_per_pair"] = {f"{VANTAGE[s]}->{VANTAGE[d]}": int(n) for (s, d), n in pings.groupby(["src", "dst"]).fp.nunique().items()}
A["distinct_pair_paths"] = int(sum(A["paths_per_pair"].values()))
span = pings.groupby(["src", "dst"]).slot.agg(["min", "max", "nunique"])
span["days"] = (span["max"] - span["min"]).dt.total_seconds() / 86400 + 1 / 48
span["coverage"] = span["nunique"] / (span["days"] * 48)
A["pair_span_days"] = {f"{VANTAGE[s]}->{VANTAGE[d]}": round(r.days, 1) for (s, d), r in span.iterrows()}
A["pair_coverage_pct"] = {f"{VANTAGE[s]}->{VANTAGE[d]}": round(r.coverage * 100, 1) for (s, d), r in span.iterrows()}
A["bandwidth_rows"] = len(bw)
A["bandwidth_tiers"] = sorted(bw.tier.dropna().unique().tolist())
tests = bw.groupby(["lab", "file_ts", "tier", "fp"]).ok.all()
A["bandwidth_failed_pct_per_lab"] = (100 - tests.groupby("lab").mean() * 100).round(1).to_dict()
A["bandwidth_ping_fp_overlap"] = len(set(bw.fp.dropna()) & set(pings.fp))

# ---- churn / runs --------------------------------------------------------------------
ps = sets.dropna(subset=["fp"])
# an empty discovery result (no paths at all) counts as absence of every path
snap = sets.groupby(["src", "dst", "slot"]).fp.apply(lambda x: frozenset(x.dropna()))
chg = []
for _, g in snap.groupby(level=[0, 1]):
    g = g.droplevel([0, 1]).sort_index()
    chg += [a != b for a, b in zip(g.iloc[:-1], g.iloc[1:])]
A["snapshot_change_pct"] = round(np.mean(chg) * 100, 1)
A["empty_snapshots"] = int((sets.status == "none").sum())
A["empty_snapshot_pct"] = round((snap.apply(len) == 0).mean() * 100, 1)
# continuous-presence runs of probed paths, measured on the path-set snapshots
runs = []
for (s, d), g in ps[ps.fp.isin(set(pings.fp))].groupby(["src", "dst"]):
    grid = pd.date_range(g.slot.min(), g.slot.max(), freq="30min")
    have = set(sets[(sets.src == s) & (sets.dst == d)].slot.unique())
    m = g.assign(v=1).pivot_table(index="slot", columns="fp", values="v", aggfunc="first").reindex(grid)
    m = m[[x in have for x in grid]].notna()  # only slots with a snapshot
    for fp in m.columns:
        v = m[fp].to_numpy().astype(int)
        edges = np.diff(np.r_[0, v, 0])
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs += [(s, d, fp, (e - b) * 0.5) for b, e in zip(starts, ends)]
R = pd.DataFrame(runs, columns=["src", "dst", "fp", "hours"])
A["runs"] = len(R)
A["runs_per_path_mean"] = round(R.groupby(["src", "dst", "fp"]).size().mean(), 1)
A["run_median_h"] = float(R.hours.median())
A["runs_under_24h_pct"] = round((R.hours < 24).mean() * 100, 1)
cand_rows = C  # each probe row appears once as a candidate at its slot
young = C.runage_now_h <= 6
A["probe_rows_first6h_pct"] = round(young.mean() * 100, 1)
A["rtt_mean_first6h_vs_older"] = [round(C.rtt_t[young].mean(), 1), round(C.rtt_t[~young].mean(), 1)]
A["loss_mean_first6h_vs_older"] = [round(C.loss_t[young].mean(), 2), round(C.loss_t[~young].mean(), 2)]
test_start = C.groupby("pair").slot.quantile(0.85)
late = C[C.slot >= C.pair.map(test_start)]
early_fps = set(zip(C[C.slot < C.pair.map(test_start)].pair, C[C.slot < C.pair.map(test_start)].fp))
A["unseen_paths_in_last15pct"] = int(len(set(zip(late.pair, late.fp)) - early_fps))

# ---- decisions / persistence ----------------------------------------------------------
D = C.groupby("decision_id").first()
inc = C[C.is_inc]
A["decisions_consecutive"] = len(D)
A["decisions_evaluable"] = int(D.inc_probed.sum())
A["evaluable_pct"] = round(D.inc_probed.mean() * 100, 1)
A["persistence_fail20"] = int(inc.fail.sum())
A["persistence_fail20_pct"] = round(inc.fail.mean() * 100, 2)
A["persistence_fail50_pct"] = round(inc.fail50.mean() * 100, 2)
A["persistence_mean_regret"] = round(inc.regret.mean(), 2)
A["persistence_p99_regret_all"] = round(inc.regret.quantile(0.99), 1)
A["persistence_p99_regret_in_failures"] = round(inc.regret[inc.fail].quantile(0.99), 1)
ev = C[C.decision_id.isin(inc.decision_id)]
best = ev.loc[ev.groupby("decision_id").rtt_t.idxmin()].set_index("decision_id")
artifact = (best.since_obs > 0) | best.last_rtt.isna()
pf = inc.set_index("decision_id")
A["persistence_fail_artifact"] = int((pf.fail & pf.index.map(artifact)).sum())
A["persistence_fail_genuine"] = int((pf.fail & ~pf.index.map(artifact).astype(bool)).sum())
A["genuine_loss_driven"] = int((pf.fail & ~pf.index.map(artifact).astype(bool) & (pf.loss_t > 5)).sum())
# label quantization: with 15 pings, loss > 5 % means at least one lost ping
lossfail = pf[pf.fail & (pf.loss_t > FAIL_LOSS)]
A["persistence_loss_failures"] = len(lossfail)
A["persistence_loss_failures_single_ping"] = int((lossfail.loss_t <= 100 / PINGS_PER_PROBE + 0.5).sum())
A["persistence_loss_failures_regret_le_20"] = int((lossfail.regret <= 20).sum())
A["random_mean_regret"] = round(ev.groupby("decision_id").regret.mean().mean(), 1)
A["candidates_per_decision_mean"] = round(D.n_cand.mean(), 1)

# ---- QoE feasibility (oracle ceiling) --------------------------------------------------
for tag, prof in (("phase2", QOE_PHASE2), ("preprint", QOE_PREPRINT)):
    best_ok = {}
    for name, (r, l) in prof.items():
        sat = (C.rtt_t <= r) & (C.loss_t <= l)
        best_ok[f"{name} ({r} ms, {l} %)"] = round(sat.groupby(C.decision_id).any().mean() * 100, 1)
    A[f"qoe_oracle_ceiling_pct_{tag}_thresholds"] = best_ok
A["min_rtt_per_pair_ms"] = C.groupby("pair").rtt_t.min().round(1).to_dict()

(TAB / "audit.json").write_text(json.dumps(A, indent=2, default=str))
print(json.dumps(A, indent=2, default=str))
