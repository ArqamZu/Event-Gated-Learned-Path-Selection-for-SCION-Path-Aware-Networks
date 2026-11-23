"""Graphical results: every figure in the report is produced here from results/tables and the saved runs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from config import FIG, PROC, TAB

RUNS = PROC / "runs"
# reference categorical palette (light mode), fixed order; grey = context, ink = text
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, LIGHT, INK, INK2 = "#9a9893", "#d9d8d4", "#0b0b0b", "#52514e"
POL = {"Proposed": BLUE, "Persistence": ORANGE, "Last-known-best": AQUA, "EMA-6 RTT": YELLOW,
       "ScionPathML heuristic (forecast)": VIOLET, "Oracle*": INK2}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig_data_overview():
    C = pd.read_parquet(PROC / "candidates.parquet")
    sets = pd.read_parquet(PROC / "pathsets.parquet")
    inc = C[C.is_inc]
    best = C.loc[C.groupby("decision_id").rtt_t.idxmin()].set_index("decision_id")
    art = (best.since_obs > 0) | best.last_rtt.isna()
    inc = inc.assign(kind=np.where(~inc.fail, "ok", np.where(inc.decision_id.map(art), "artifact", "genuine")))
    daily = inc[inc.fail].groupby([inc.slot.dt.floor("D"), "kind"]).size().unstack(fill_value=0)
    empty = sets[sets.status == "none"].groupby(sets.slot.dt.floor("D")).size()
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.35))
    x = daily.index
    ax[0].bar(x, daily.get("artifact", 0), width=0.8, color=ORANGE, label="sampling artifact (best path not probed at t-1)")
    ax[0].bar(x, daily.get("genuine", 0), width=0.8, bottom=daily.get("artifact", 0), color=BLUE,
              label="genuine (loss > 5 % on the incumbent)", edgecolor="white", linewidth=0.5)
    ax[0].set_ylabel("persistence failures / day"); ax[0].set_ylim(0, 31)
    ax[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    ax[0].set_title("Persistence failures per day, by cause (all six pairs)")
    ax[1].bar(empty.index, empty.values, width=0.8, color=GREY)
    ax[1].set_ylabel("empty path-set\nsnapshots / day")
    ax[1].set_title("Total path-set outages (path discovery returned no path)")
    fig.autofmt_xdate()
    save(fig, "fig01_data_overview")


def fig_main():
    S = pd.read_csv(TAB / "main_cv.csv")
    keep = ["Proposed", "Ranker only", "Proposed, guarded threshold", "Last-known-best", "EMA-6 RTT", "Nowcast min-RTT*",
            "ScionPathML heuristic (forecast)", "Persistence", "Proposed, Phase 2 incumbent",
            "Epsilon-greedy (0.1)", "Oracle*"]
    rnd = S.set_index("policy").loc["Random"]
    n_dec = int(S.set_index("policy").loc["Proposed", "n"])
    S = S.set_index("policy").loc[keep].reset_index().iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    y = np.arange(len(S))
    col = [BLUE if p == "Proposed" else (INK2 if p.endswith("*") else GREY) for p in S.policy]
    for a, (m, lo, hi, lab) in zip(ax, (("fail_pct", "fail_lo", "fail_hi", "failure share (%)"),
                                       ("regret_mean", "reg_lo", "reg_hi", "mean RTT regret (ms)"))):
        a.barh(y, S[m], color=col, height=0.6)
        a.errorbar(S[m], y, xerr=[S[m] - S[lo], S[hi] - S[m]], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for yi, v, h in zip(y, S[m], S[hi]):
            a.text(h + S[hi].max() * 0.025, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        a.set_xlim(0, S[hi].max() * 1.15)
        a.set_xlabel(lab); a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, S.policy)
    ax[0].set_title("Failure share, 95 % day-block bootstrap CI")
    ax[1].set_title("Mean regret vs best path in hindsight")
    fig.text(0.01, -0.06, f"Pooled purged expanding-window CV, {n_dec:,} decisions. * = not deployable (uses the interval's own "
             f"measurements).\nRandom: {rnd.fail_pct:.1f} % failures, {rnd.regret_mean:.1f} ms regret (off scale).",
             fontsize=7.5, color=INK2)
    save(fig, "fig02_main_comparison")


def fig_regret_ccdf():
    C = pd.read_parquet(PROC / "candidates.parquet")
    prop = pd.read_parquet(RUNS / "cv_proposed_s0.parquet")
    import sys; sys.path.insert(0, ".")
    from core import baseline, load
    CC = load(); te = CC[CC.decision_id.isin(prop.decision_id)]
    pols = {"Proposed": prop, "Persistence": baseline(te, "Persistence"), "Last-known-best": baseline(te, "Last-known-best"),
            "EMA-6 RTT": baseline(te, "EMA-6 RTT"), "ScionPathML heuristic (forecast)": baseline(te, "ScionPathML heuristic (forecast)")}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.linspace(0, 120, 400)
    for name, ch in pols.items():
        r = ch.regret.to_numpy()
        ax.plot(xs, [(r > x).mean() * 100 for x in xs], color=POL[name], label=name,
                linestyle="--" if name == "Last-known-best" else "-")
    ax.axvline(20, color=GREY, linewidth=1, linestyle=":"); ax.text(21, 30, "20 ms failure margin", color=INK2, fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(0.02, 40)
    ax.set_xlabel("RTT regret x (ms)"); ax.set_ylabel("decisions with regret > x (%)")
    ax.set_title("Regret tail (complementary CDF, log scale)")
    ax.legend(fontsize=8)
    save(fig, "fig03_regret_ccdf")


def fig_failure_subsets():
    F = pd.read_csv(TAB / "failure_subset_cv.csv")
    subs = ["persistence fails: sampling artifact", "persistence fails: genuine"]
    pols = ["Persistence", "Last-known-best", "EMA-6 RTT", "Proposed", "Oracle*"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    w = 0.16
    for i, p in enumerate(pols):
        v = [F[(F.subset == s) & (F.policy == p)].fail_pct.iloc[0] for s in subs]
        n = [F[(F.subset == s) & (F.policy == p)].n.iloc[0] for s in subs]
        b = ax.bar(np.arange(2) + (i - 2) * (w + 0.01), v, w, color=POL.get(p, GREY), label=p)
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + w / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(2), [f"sampling artifact\n(n = {n[0]})", f"genuine degradation\n(n = {n[1]})"])
    ax.set_ylabel("failure share on subset (%)"); ax.set_ylim(0, 112)
    ax.set_title("Decisions where persistence fails: who still fails?")
    ax.legend(ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig04_failure_subsets")


def fig_gate_pr():
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    g2 = pd.read_parquet(RUNS / "cv_gate_phase2.parquet")
    p, r, _ = precision_recall_curve(g2.fail, g2.p_gate)
    ax[0].plot(r, p, color=BLUE, label=f"LightGBM gate (AUPRC {average_precision_score(g2.fail, g2.p_gate):.2f})")
    ax[0].axhline(g2.fail.mean(), color=GREY, linestyle=":", linewidth=1)
    ax[0].text(0.55, g2.fail.mean() + 0.02, f"prevalence {g2.fail.mean():.3f}", fontsize=7.5, color=INK2)
    ax[0].set_title("Phase 2 incumbent: failures incl. artifact"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    for kind, c, lab in (("lgbm", BLUE, "LightGBM"), ("rf", AQUA, "Random forest"), ("lr", VIOLET, "Logistic regression")):
        g = pd.read_parquet(RUNS / f"cv_gate_{kind}.parquet")
        p, r, _ = precision_recall_curve(g.fail, g.p_gate)
        ax[1].plot(r, p, color=c, label=f"{lab} (AUPRC {average_precision_score(g.fail, g.p_gate):.3f})")
    ax[1].axhline(g.fail.mean(), color=GREY, linestyle=":", linewidth=1)
    ax[1].text(0.55, g.fail.mean() + 0.02, f"prevalence {g.fail.mean():.3f}", fontsize=7.5, color=INK2)
    ax[1].set_title("Final gate: incumbent = ranker's top path"); ax[1].set_xlabel("recall")
    for a in ax:
        a.legend(fontsize=7.5, loc="upper right"); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    save(fig, "fig05_gate_pr_curves")


def fig_threshold_tradeoff():
    T = pd.read_csv(TAB / "threshold_sweep_phase2.csv")
    S = pd.read_csv(TAB / "main_cv.csv").set_index("policy")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(100 * T.fired, T.fail_pct, color=BLUE, label="Phase 2 design: gate + ranker, threshold swept")
    for p, ls in (("Persistence", "-"), ("Last-known-best", "--"), ("Proposed", ":")):
        ax.axhline(S.loc[p, "fail_pct"], color=POL[p], linestyle=ls, linewidth=1.5, label=p if p != "Proposed" else "Proposed (final)")
    ax.set_xlabel("gate firing rate (% of decisions)"); ax.set_ylabel("failure share (%)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 14)
    ax.set_title("Switching budget vs failure share (pooled CV)")
    ax.legend(fontsize=7.5)
    save(fig, "fig06_threshold_tradeoff")


def fig_runage():
    R = pd.read_csv(TAB / "runage_cv.csv")
    order = ["<2 h", "2-6 h", "6-24 h", ">=24 h"]
    pols = ["Persistence", "Last-known-best", "Proposed", "Proposed, no blend"]
    cols = [ORANGE, AQUA, BLUE, MAGENTA]
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.3))
    w = 0.2
    n = R[R.policy == "Persistence"].set_index("run_age").reindex(order).n
    for a, (m, lab) in zip(ax, (("fail_pct", "failure share (%)"), ("regret_mean", "mean regret (ms)"),
                                ("switch_per_pair_day", "switches per pair-day"))):
        for i, (p, c) in enumerate(zip(pols, cols)):
            x = R[R.policy == p].set_index("run_age").reindex(order)
            a.bar(np.arange(4) + (i - 1.5) * (w + 0.01), x[m], w, color=c, label=p)
        a.set_xticks(np.arange(4), [f"{o}\n(n = {int(k)})" for o, k in zip(order, n)], fontsize=7.5)
        a.set_ylabel(lab)
    ax[0].set_title("Failures"); ax[1].set_title("Mean regret"); ax[2].set_title("Switching")
    ax[1].set_xlabel("run age of the Phase 2 incumbent (time since its path reappeared)")
    ax[0].legend(fontsize=7, ncol=1)
    for a in ax:
        a.tick_params(axis="x", labelsize=6.8)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig07_runage")


def fig_drift_lovo():
    D = pd.read_csv(TAB / "drift.csv"); L = pd.read_csv(TAB / "lovo.csv", keep_default_na=False)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    w = 0.26
    segs = list(dict.fromkeys(D.segment))
    for i, (m, c) in enumerate((("static (week 1)", VIOLET), ("retrained on all earlier data", BLUE))):
        x = D[D.model == m].set_index("segment").reindex(segs)
        ax[0].bar(np.arange(len(segs)) + (i - 1) * (w + 0.01), x.fail_pct, w, color=c, label=m)
    x = D[D.model == "static (week 1)"].set_index("segment").reindex(segs)
    ax[0].bar(np.arange(len(segs)) + (w + 0.01), x.lkb_fail_pct, w, color=AQUA, label="Last-known-best")
    lab = {"week 4 (days 21-23)": "week 4\n(days 21-23)", "days 24-26 (CH offline, EU<->NA only)": "days 24-26\n(CH offline)"}
    ax[0].set_xticks(np.arange(len(segs)), [f"{lab.get(s, s)}\nn = {int(n)}" for s, n in zip(segs, x.n)], fontsize=7.5)
    ax[0].set_ylabel("failure share (%)")
    ax[0].set_title("Drift: static vs retrained"); ax[0].legend(fontsize=7)
    models = (("in-vantage (all pairs)", BLUE), ("source held out", MAGENTA), ("vantage unseen (src and dst)", VIOLET))
    for i, (m, c) in enumerate(models):
        x = L[L.model == m]
        ax[1].bar(np.arange(3) + (i - 1.5) * (w * 0.8 + 0.01), x.fail_pct, w * 0.8, color=c, label=m)
    x = L[L.model == "in-vantage (all pairs)"]
    ax[1].bar(np.arange(3) + 1.5 * (w * 0.8 + 0.01), x.lkb_fail_pct, w * 0.8, color=AQUA, label="Last-known-best")
    ax[1].set_xticks(np.arange(3), [f"source {v}" for v in x.held_out]); ax[1].set_title("Cross-vantage transfer")
    ax[1].legend(fontsize=7, loc="upper left")
    save(fig, "fig08_drift_lovo")


def fig_ablation():
    A = pd.read_csv(TAB / "ablations_cv.csv")
    g = A.groupby("ablation", sort=False).agg(f=("fail_pct", "mean"), fmin=("fail_pct", "min"), fmax=("fail_pct", "max"),
                                             r=("regret_mean", "mean"), rmin=("regret_mean", "min"), rmax=("regret_mean", "max"),
                                             s=("switch_per_pair_day", "mean"))
    full = g.loc["Full model"]
    g = g.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 6.0), sharey=True)
    y = np.arange(len(g))
    for a, (m, lo, hi, lab) in zip(ax, (("f", "fmin", "fmax", "failure share (%)"), ("r", "rmin", "rmax", "mean regret (ms)"))):
        col = [BLUE if i == "Full model" else (ORANGE if g.loc[i, m] > full[m] * 1.05 + 1e-9 else GREY) for i in g.index]
        a.barh(y, g[m], color=col, height=0.6)
        a.errorbar(g[m], y, xerr=[g[m] - g[lo], g[hi] - g[m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
        a.axvline(full[m], color=BLUE, linewidth=1, linestyle=":")
        a.set_xlabel(lab); a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, g.index)
    fig.suptitle("Ablations (pooled CV, mean and range over 5 seeds; orange = worse than the full model by > 5 %)",
                 fontsize=10, fontweight="bold", color=INK, y=1.0)
    save(fig, "fig09_ablations")


def fig_importance():
    I = pd.read_csv(TAB / "importance_gate_phase2.csv").head(15).iloc[::-1]
    gcol = {"current": BLUE, "rolling": ORANGE, "stability": AQUA, "structure": YELLOW, "run_age": MAGENTA,
            "observation": GREEN, "relative": VIOLET, "temporal": RED, "volatility": INK2}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.barh(np.arange(len(I)), I.mean_abs_shap, color=[gcol[g] for g in I.group], height=0.6)
    ax.set_yticks(np.arange(len(I)), I.feature); ax.grid(axis="y", visible=False)
    for g in dict.fromkeys(I.group[::-1]):
        ax.barh([0], [0], color=gcol[g], label=g)
    ax.legend(title="feature group", fontsize=7.5, title_fontsize=8, loc="lower right")
    ax.set_xlabel("mean |SHAP value| (log-odds)"); ax.set_title("Phase 2 gate: top-15 features by SHAP (CV fold with most failure events)")
    save(fig, "fig10_shap_importance")


def fig_qoe():
    Q = pd.read_csv(TAB / "qoe_cv.csv")
    pols = ["Persistence", "Last-known-best", "ScionPathML heuristic (forecast)", "Proposed"]
    cols = [ORANGE, AQUA, VIOLET, BLUE]
    profs = list(dict.fromkeys(Q.profile))
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    w = 0.18
    y = np.arange(len(profs))
    ceil = Q.groupby("profile", sort=False).oracle_ceiling_pct.first().reindex(profs)
    ax.barh(y, ceil, 0.86, color="#f1f0ed", label="oracle ceiling")
    for i, (p, c) in enumerate(zip(pols, cols)):
        v = Q[Q.policy == p].set_index("profile").reindex(profs).satisfaction_pct
        ax.barh(y + (1.5 - i) * w, v, w * 0.9, color=c, label=p)
    ax.set_yticks(y, [p.replace(" (", "\n(") for p in profs], fontsize=7); ax.grid(axis="y", visible=False)
    for yi, c in zip(y, ceil):
        if c < 2:
            ax.text(2, yi, f"oracle ceiling {c:.1f} %: no path meets the threshold (min RTT ~246 ms)", va="center",
                    fontsize=7, color=INK2)
    ax.set_xlabel("QoE satisfaction (% of decisions)"); ax.set_xlim(0, 105)
    ax.set_title("QoE profile satisfaction: Phase 2 (published) thresholds, preprint thresholds, relative variants")
    ax.invert_yaxis()
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=5)
    save(fig, "fig11_qoe")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.4, 5.4)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.4, 8.4)
    xs = [0.6, 4.4, 8.2]; W, H = 3.2, 1.05
    stages = [("A  Data", 7.0, ["1 Research input\n6 directed pairs, 30-min", "2 Data collection\nScionPathML raw JSON",
                                "3 Preprocessing\ngrid, masks, splits, audit"], 1),
              ("B  Model", 5.0, ["6 Baselines\n9 policies + oracle", "5 Two-stage switcher\nranker top + aligned gate",
                                 "4 Features\n9 groups (+ volatility)"], -1),
              ("C  Evaluation", 3.0, ["7 Experiments A-G\ntest split + purged CV", "8 Metrics\nfailure, regret, AUPRC",
                                      "9 Statistics\nday bootstrap, McNemar"], 1),
              ("D  Analysis & output", 1.0, ["12 Output\ncode, tables, report", "11 Error analysis\nartifact split, SHAP",
                                             "10 Ablations\n23 variants x 5 seeds"], -1)]
    arrow = dict(arrowstyle="-|>", color=GREY, lw=1.1)
    for title, yc, boxes, d in stages:
        ax.add_patch(FancyBboxPatch((0.2, yc - 0.85), 11.6, 1.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#f7f7f5", ec=LIGHT, linestyle="--"))
        ax.text(0.35, yc + 0.62, title, fontsize=8.5, color=INK2, weight="bold")
        for x, b in zip(xs, boxes):
            hi = b.startswith("5 ")
            ax.add_patch(FancyBboxPatch((x, yc - 0.55), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc="#e8f1fc" if hi else "white", ec=BLUE if hi else GREY, linewidth=1.5 if hi else 0.8))
            t, sub = b.split("\n")
            ax.text(x + W / 2, yc + 0.12, t, ha="center", fontsize=8.5, color=INK, weight="bold")
            ax.text(x + W / 2, yc - 0.25, sub, ha="center", fontsize=7, color=INK2)
        for i in range(2):  # horizontal arrows in reading direction of the stage
            a, b = (xs[i] + W, xs[i + 1]) if d == 1 else (xs[i + 1], xs[i] + W)
            ax.annotate("", (b, yc - 0.03), (a, yc - 0.03), arrowprops=arrow)
    for yc, x in ((7.0, xs[2]), (5.0, xs[0]), (3.0, xs[2])):  # vertical connectors at the row ends
        ax.annotate("", (x + W / 2, yc - 2.0 + 0.5), (x + W / 2, yc - 0.55), arrowprops=arrow)
    # feedback loop: 11 Error analysis -> 4 Features / 5 Model (orange, along the left margin of row C)
    gx = (xs[0] + W + xs[1]) / 2  # gap between first and second column
    ax.plot([xs[1], gx, gx], [1.0, 1.0, 4.8], color=ORANGE, lw=1.4, linestyle="--")
    ax.annotate("", (xs[1], 4.8), (gx, 4.8), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.text(0.35, -0.05, "Orange: feedback from error analysis (Step 11) to the model (Steps 4-5): volatility features, "
            "incumbent = ranker top, gate trained on that incumbent.", fontsize=7, color=ORANGE, va="top")
    ax.set_title("Phase 3 implemented pipeline (changes from Phase 2: highlighted box and orange loop)", fontsize=9)
    save(fig, "fig00_architecture")


def fig_bandwidth():
    bw = pd.read_parquet(PROC / "bandwidth.parquet")
    bw["ratio"] = bw.achieved_mbps / bw.tier
    NA = dict(keep_default_na=False, na_values=[""])  # "NA" = North America
    PE = pd.read_csv(TAB / "bw_path_effect.csv", **NA); H = pd.read_csv(TAB / "bw_hourly.csv", **NA)
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.4), gridspec_kw=dict(wspace=0.38))
    ok = bw[bw.ok]
    data, labs = [], []
    for tier in (10, 50, 100):
        for d in ("S->C", "C->S"):
            data.append(ok[(ok.tier == tier) & (ok.direction == d)].ratio.to_numpy()); labs.append(f"{tier}\n{d}")
    ax[0].boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor="#e8f1fc", edgecolor=BLUE), medianprops=dict(color=BLUE))
    ax[0].set_ylabel("achieved / target rate"); ax[0].set_title("Throughput by tier and direction")
    for v, c in (("EU", BLUE), ("CH", ORANGE), ("NA", AQUA)):
        x = H[H.src_v == v]
        ax[1].plot(x.hour, x.ratio_mean, color=c, label=v)
    ax[1].set_xlabel("hour of day (UTC)"); ax[1].set_ylabel("mean achieved / 100 Mbps"); ax[1].legend(fontsize=7.5)
    ax[1].set_title("Hour-of-day profile (100 Mbps)")
    x = PE[PE.tier == 100]
    ax[2].bar(np.arange(len(x)), 100 * x.between_path_var_share, color=VIOLET, label="variance explained by path")
    ax[2].bar(np.arange(len(x)), -x.paired_gap_over_10pct, color=YELLOW,
              label="same-time path pairs differing > 10 % (shown negative)")
    ax[2].axhline(0, color=INK2, linewidth=0.8)
    ax[2].set_xticks(np.arange(len(x)), [f"{a}\n{b[-4:]}\n{c}" for a, b, c in zip(x.src_v, x.server, x.direction)], fontsize=6.5)
    ax[2].set_ylabel("%"); ax[2].set_title("Does the path matter? (100 Mbps)"); ax[2].legend(fontsize=6.5, loc="lower left")
    save(fig, "fig12_bandwidth")


def fig_label_sensitivity():
    L = pd.read_csv(TAB / "sensitivity_labels_cv.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(L)); w = 0.2
    for i, (col, lab, c) in enumerate((("pers_fail_pct", "Persistence", ORANGE), ("lkb_fail_pct", "Last-known-best", AQUA),
                                        ("proposed_fail_pct", "Proposed", BLUE), ("oracle_fail_pct", "Oracle*", INK2))):
        ax.bar(x + (i - 1.5) * (w + 0.01), L[col], w, color=c, label=lab)
    ax.set_xticks(x, [f"{int(a)} ms / {b:g} %\n(>= {int(n)} lost ping{'s' if n > 1 else ''})"
                      for a, b, n in zip(L.margin_ms, L.loss_pct, L.min_lost_pings)], fontsize=7)
    ax.set_ylabel("failure share (%)"); ax.set_title("Failure-threshold sensitivity (regret margin / loss limit), pooled CV")
    ax.legend(fontsize=7.5)
    save(fig, "fig13_label_sensitivity")


if __name__ == "__main__":
    for f in (fig_architecture, fig_data_overview, fig_main, fig_regret_ccdf, fig_failure_subsets, fig_gate_pr,
              fig_threshold_tradeoff, fig_runage, fig_drift_lovo, fig_ablation, fig_importance, fig_qoe,
              fig_bandwidth, fig_label_sensitivity):
        try:
            f()
        except Exception as e:  # keep going so a single missing table does not block the rest
            print("FAILED", f.__name__, repr(e))
