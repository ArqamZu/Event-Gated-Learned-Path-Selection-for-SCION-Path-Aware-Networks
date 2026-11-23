"""Auxiliary bandwidth analysis (Phase 2, Step 3: bandwidth is excluded from per-path decisions
because bandwidth tests share no path fingerprint with ping tests, and retained for this analysis).

Bandwidth tests run from each vantage point to two core-AS servers (18-ffaa:0:1201, 19-ffaa:0:1303)
at target rates of 10, 50 and 100 Mbps, in both directions, over 1-2 paths per test.
Outputs (results/tables):
  bw_overview.csv     tests, failure and degradation share, achieved/target ratio, loss per
                      source x server x tier x direction
  bw_path_effect.csv  how much of the throughput variation is explained by the path (between-path
                      share of variance, eta^2) and by path length; gap between the two paths tested
                      at the same time (headroom from path choice)
  bw_hourly.csv       hour-of-day profile (100 Mbps tier)
  bw_weekly.csv       drift by the same calendar segments as the RTT drift experiment
  bw_ping_link.json   source-level temporal link between ping degradation and bandwidth degradation
                      in the same 30-minute slot (Spearman), the only link possible without path overlap
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROC, TAB, VANTAGE

DEGRADED = 0.9  # achieved below 90 % of the target rate (test completed) = degraded

bw = pd.read_parquet(PROC / "bandwidth.parquet")
bw["src_v"] = bw.src.map(VANTAGE)
bw["ratio"] = bw.achieved_mbps / bw.tier
bw["failed"] = ~bw.ok
bw["degraded"] = bw.ok & (bw.ratio < DEGRADED)
bw["n_as"] = bw.sequence.fillna("").str.count("->") + 1
t0 = bw.slot.min().floor("D")
day = (bw.slot - t0).dt.days
bw["segment"] = np.select([day < 7, day < 14, day < 21, day < 24], ["week 1", "week 2", "week 3", "week 4 (days 21-23)"],
                          "days 24-26")
ok = bw[bw.ok]

# ---- 1. overview --------------------------------------------------------------------------
ov = bw.groupby(["src_v", "server", "tier", "direction"]).agg(
    n_tests=("ok", "size"), n_paths=("fp", "nunique"), failed_pct=("failed", "mean"), degraded_pct=("degraded", "mean"),
    ratio_median=("ratio", "median"), ratio_p10=("ratio", lambda s: s.quantile(0.1)), loss_median=("loss", "median"))
ov[["failed_pct", "degraded_pct"]] *= 100
ov.reset_index().to_csv(TAB / "bw_overview.csv", index=False)
lab_summary = bw.groupby("src_v").agg(tests=("ok", "size"), failed_pct=("failed", "mean"), degraded_pct=("degraded", "mean"))
lab_summary[["failed_pct", "degraded_pct"]] *= 100

# ---- 2. path effect -------------------------------------------------------------------------
rows = []
for (v, srv, tier, dr), g in ok.groupby(["src_v", "server", "tier", "direction"]):
    cnt = g.fp.value_counts()
    g = g[g.fp.isin(cnt.index[cnt >= 20])]
    if g.fp.nunique() < 2:
        continue
    grand = g.ratio.mean()
    between = g.groupby("fp").ratio.agg(["mean", "size"]).pipe(lambda d: (d["size"] * (d["mean"] - grand) ** 2).sum())
    total = ((g.ratio - grand) ** 2).sum()
    # gap between the two paths tested in the same test (same time, same server/tier/direction)
    pairs = g.groupby("file_ts").ratio.agg(["max", "min", "size"])
    pairs = pairs[pairs["size"] == 2]
    gap = pairs["max"] - pairs["min"]
    rho, p = spearmanr(g.n_as, g.ratio) if g.n_as.nunique() > 1 else (np.nan, np.nan)
    rows.append(dict(src_v=v, server=srv, tier=tier, direction=dr, n_tests=len(g), n_paths=g.fp.nunique(),
                     between_path_var_share=between / total if total > 0 else np.nan,
                     path_mean_ratio_min=g.groupby("fp").ratio.mean().min(), path_mean_ratio_max=g.groupby("fp").ratio.mean().max(),
                     paired_tests=len(gap), paired_gap_median=gap.median(), paired_gap_over_10pct=(gap > 0.10).mean() * 100,
                     spearman_len_vs_ratio=rho, spearman_p=p))
PE = pd.DataFrame(rows)
PE.to_csv(TAB / "bw_path_effect.csv", index=False)

# ---- 3. hour of day, 4. weekly drift -------------------------------------------------------
h = bw[bw.tier == 100].assign(hour=lambda d: d.slot.dt.hour)
h.groupby(["src_v", "hour"]).agg(ratio_mean=("ratio", "mean"), failed_pct=("failed", "mean"), n=("ok", "size")) \
    .assign(failed_pct=lambda d: 100 * d.failed_pct).reset_index().to_csv(TAB / "bw_hourly.csv", index=False)
wk = bw.groupby(["segment", "tier"]).agg(n=("ok", "size"), failed_pct=("failed", "mean"), degraded_pct=("degraded", "mean"),
                                          ratio_median=("ratio", "median"))
wk[["failed_pct", "degraded_pct"]] *= 100
wk.reset_index().to_csv(TAB / "bw_weekly.csv", index=False)

# ---- 5. temporal link with ping measurements from the same source --------------------------
pings = pd.read_parquet(PROC / "pings.parquet")
pings = pings[pings.status == "ok"].assign(src_v=lambda d: d.src.map(VANTAGE), lossy=lambda d: d.loss > 0)
pslot = pings.groupby(["src_v", "slot"]).agg(ping_lossy_share=("lossy", "mean"), ping_rtt_median=("rtt_avg", "median"))
bslot = bw[bw.tier == 100].groupby(["src_v", "slot"]).agg(bw_ratio_mean=("ratio", "mean"), bw_failed_share=("failed", "mean"))
J = pslot.join(bslot, how="inner")
link = {}
for v, g in J.groupby(level=0):
    r1 = spearmanr(g.ping_lossy_share, g.bw_ratio_mean, nan_policy="omit")
    r2 = spearmanr(g.ping_rtt_median, g.bw_ratio_mean, nan_policy="omit")
    r3 = spearmanr(g.ping_lossy_share, g.bw_failed_share, nan_policy="omit")
    link[v] = dict(n_slots=len(g),
                   rho_ping_lossy_vs_bw_ratio=float(r1.statistic), p1=float(r1.pvalue),
                   rho_ping_rtt_vs_bw_ratio=float(r2.statistic), p2=float(r2.pvalue),
                   rho_ping_lossy_vs_bw_failed=float(r3.statistic), p3=float(r3.pvalue))
summary = dict(rows=len(bw), tests_by_source=lab_summary.round(2).to_dict("index"), degraded_definition=
               f"test completed but achieved < {DEGRADED:.0%} of target", ping_fp_overlap=len(set(bw.fp) & set(pings.fp)),
               ratio_median_by_tier=ok.groupby("tier").ratio.median().round(3).to_dict(),
               link_by_source=link)
(TAB / "bw_ping_link.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
print(PE.round(3).to_string())
