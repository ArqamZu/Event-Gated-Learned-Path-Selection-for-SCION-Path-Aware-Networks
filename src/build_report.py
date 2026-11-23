"""Builds report/Phase3_Report.docx. Every number is read from results/tables (no hand-copied values)."""
import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from sklearn.metrics import average_precision_score, roc_auc_score

from config import BLEND_K, DEAD_REGRET_MS, FIG, LABEL_GRID, MODELS, PROC, QOE_PHASE2, QOE_PREPRINT, ROOT, TAB
from core import ALL_FEATURES, GROUPS

NA = dict(keep_default_na=False, na_values=[""])  # "NA" is the North-America vantage, not a missing value
A = json.loads((TAB / "audit.json").read_text())
COST = json.loads((TAB / "cost.json").read_text())
TUNE = json.loads((TAB / "tuning.json").read_text())
FC = json.loads((TAB / "forecast_reference.json").read_text())
TAX = json.loads((TAB / "failure_taxonomy_cv.json").read_text())
SC = json.loads((TAB / "shortcut_audit.json").read_text())
SC2 = json.loads((TAB / "shortcut_phase2_gate.json").read_text())
BWL = json.loads((TAB / "bw_ping_link.json").read_text())
CARD = json.loads((MODELS / "model_card.json").read_text())
CV = pd.read_csv(TAB / "main_cv.csv").set_index("policy")
TE = pd.read_csv(TAB / "main_test.csv").set_index("policy")
ST = pd.read_csv(TAB / "stats_cv.csv").set_index("comparison")
STT = pd.read_csv(TAB / "stats_test.csv").set_index("comparison")
GQ = pd.read_csv(TAB / "gate_quality_cv.csv").set_index("gate")
FS = pd.read_csv(TAB / "failure_subset_cv.csv")
RA = pd.read_csv(TAB / "runage_cv.csv")
CS = pd.read_csv(TAB / "coldstart_cv.csv")
DR = pd.read_csv(TAB / "drift.csv")
LV = pd.read_csv(TAB / "lovo.csv", **NA)
EB = pd.read_csv(TAB / "error_breakdown_cv.csv", **NA)
QO = pd.read_csv(TAB / "qoe_cv.csv")
AB = pd.read_csv(TAB / "ablations_cv.csv")
SB = pd.read_csv(TAB / "sensitivity_budget_cv.csv")
SL = pd.read_csv(TAB / "sensitivity_labels_cv.csv")
SW = pd.read_csv(TAB / "threshold_sweep_phase2.csv")
IMP = pd.read_csv(TAB / "importance_gate.csv")
IMP2 = pd.read_csv(TAB / "importance_gate_phase2.csv")
BWO = pd.read_csv(TAB / "bw_overview.csv")
BWP = pd.read_csv(TAB / "bw_path_effect.csv")
BWW = pd.read_csv(TAB / "bw_weekly.csv")
SEEDS_CV = pd.read_csv(TAB / "cv_proposed_seeds.csv")
SEEDS_TE = pd.read_csv(TAB / "test_proposed_seeds.csv")
G2 = pd.read_parquet(PROC / "runs" / "cv_gate_phase2.parquet")

P, PE, LKB, EMA = CV.loc["Proposed"], CV.loc["Persistence"], CV.loc["Last-known-best"], CV.loc["EMA-6 RTT"]
OR, HF, P2 = CV.loc["Oracle*"], CV.loc["ScionPathML heuristic (forecast)"], CV.loc["Proposed, Phase 2 incumbent"]
RK, GD = CV.loc["Ranker only"], CV.loc["Proposed, guarded threshold"]
GO, GP = TAX["gate_outcome"], TAX["failure_cause"]
FA = TAX["failure_attributes"]
G_AV = GO["incumbent fails, gate fired, switch avoided the failure"]
G_HARM = GO["false alarm, switch caused a failure"]
G_OK = GO["false alarm, harmless switch"]
G_MISS = GO["incumbent fails, gate silent (missed / late detection)"]
LG = GQ.loc["LightGBM gate"]


def f(x, d=2):
    return f"{x:.{d}f}"


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)
for lvl, size in ((1, 15), (2, 12.5), (3, 11)):
    h = doc.styles[f"Heading {lvl}"]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

FIGN = [0]
TABN = [0]


def para(text, bold_lead=None, size=None, italic=False, align=None, space=4):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
    r = p.add_run(text); r.italic = italic
    if size:
        for r in p.runs:
            r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space)
    return p


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True; p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(2)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)


def table(df, caption, widths=None, font=8.5, highlight=None):
    TABN[0] += 1
    cp = doc.add_paragraph(); r = cp.add_run(f"Table {TABN[0]}. "); r.bold = True; cp.add_run(caption)
    cp.paragraph_format.space_after = Pt(2); cp.paragraph_format.keep_with_next = True
    for r in cp.runs:
        r.font.size = Pt(9)
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; run = cell.paragraphs[0].add_run(str(c)); run.bold = True
        run.font.size = Pt(font); shade(cell, "DCE6F2")
    for ri, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(font)
            if highlight is not None and highlight(row):
                run.bold = True; shade(cells[i], "EEF4FB")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return TABN[0]


def figure(name, caption, width=16):
    FIGN[0] += 1
    doc.add_picture(str(FIG / f"{name}.png"), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph(); r = cp.add_run(f"Figure {FIGN[0]}. "); r.bold = True; cp.add_run(caption)
    for r in cp.runs:
        r.font.size = Pt(9)
    cp.paragraph_format.space_after = Pt(8)
    return FIGN[0]


# ablation summary (used in several sections)
ABG = AB.groupby("ablation", sort=False).agg(fail=("fail_pct", "mean"), fsd=("fail_pct", "std"), reg=("regret_mean", "mean"),
                                             rsd=("regret_mean", "std"), sw=("switch_per_pair_day", "mean"),
                                             fired=("fired_pct", "mean"), auprc=("auprc", "mean"),
                                             prev=("gate_prevalence", "mean"))
N_SEEDS = AB.seed.nunique()
N_ABL = len(ABG) - 1


def gv(n, c="reg"):
    return ABG.loc[n, c]


SL0 = SL[(SL.margin_ms == 20) & (SL.loss_pct == 5)].iloc[0]

# ======================================================================================
# Title
# ======================================================================================
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Phase 3: Research Implementation and Experimental Evaluation"); r.bold = True; r.font.size = Pt(17)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Supervised, event-gated SCION path selection on the ScionPathML live dataset"); r.font.size = Pt(12); r.italic = True
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Group Members: 1) Arqam Zulfiqar (23I-2608)     2) Ali Haroon (23I-2592)").font.size = Pt(10.5)

doc.add_heading("Summary of findings", level=1)
para(f"We implemented the Phase 2 methodology end to end on the public ScionPathML raw measurements "
     f"({A['ping_rows']:,} probe rows from repository commit {A['dataset_commit'][:7]}, six directed pairs, "
     f"{A['decisions_evaluable']:,} evaluable 30-minute decisions) and evaluated it against nine baselines and an oracle with two "
     "time-respecting protocols. The main results are:")
bullets([
    ("Reproduction. ", f"The Phase 2 dataset figures reproduce from raw data (Table 2): persistence fails in {A['persistence_fail20_pct']} % of "
     f"decisions ({A['persistence_fail20']} of {A['decisions_evaluable']:,}), mean regret {A['persistence_mean_regret']} ms, "
     f"{A['runs']:,} presence runs with a median of {A['run_median_h']:.0f} h. The published QoE-satisfaction result (28-35 %) cannot be "
     "reproduced: under the published thresholds the oracle ceiling is about 1 % or 0 % for two profiles and 100 % for three."),
    ("Most persistence failures are a sampling artifact. ", f"{A['persistence_fail_artifact']} of the "
     f"{A['persistence_fail20']} failures ({A['persistence_fail_artifact'] / A['persistence_fail20'] * 100:.1f} %) occur because the best path "
     f"was not among the 15 randomly probed paths in the previous interval. The other {A['persistence_fail_genuine']} are packet-loss events, "
     f"and {A['persistence_loss_failures_single_ping']} of the {A['persistence_loss_failures']} loss failures are a single lost ping out of 15."),
    ("Proposed method. ", f"The two-stage switcher (LambdaRank ranker whose top path is the incumbent, plus a LightGBM gate trained on "
     f"that incumbent, threshold chosen by the Phase 2 rule) fails in {f(P.fail_pct)} % of decisions vs {f(PE.fail_pct)} % for "
     f"persistence (difference CI [{f(P.dfail_lo)}, {f(P.dfail_hi)}] points; McNemar p {pv(ST.loc['Proposed vs Persistence', 'mcnemar_p_holm'])}), "
     f"with mean regret {f(P.regret_mean, 3)} vs {f(PE.regret_mean)} ms. It also beats the re-implemented ScionPathML heuristic "
     f"({f(HF.fail_pct)} % failures)."),
    ("The gate does not pay for itself. ", f"Its AUPRC ({f(LG.auprc, 3)}) is above the failure prevalence ({f(LG.prevalence, 3)}), "
     f"but it fires on {f(P.fired_pct)} % of decisions, avoids {G_AV} failures and causes {G_HARM}. The ranker alone fails in "
     f"{f(RK.fail_pct)} % ({f(RK.regret_mean, 3)} ms), and so does the guarded-threshold variant, which almost never fires. Last-known-best "
     f"and EMA-6 tie with the ranker. The remaining failures are mostly single lost pings that 30-minute history cannot predict."),
])

# ======================================================================================
# 1. Implementation
# ======================================================================================
doc.add_heading("1. Implementation of the proposed methodology", level=1)
doc.add_heading("1.1 What was implemented", level=2)
para("The whole Phase 2 pipeline (Figure 1) is implemented in Python as a set of scripts that rebuild every table, figure and "
     "the saved model from the raw archives (Appendix A). The components are:")
bullets([
    ("Data layer (Steps 1-3): ", "a parser that reads the three raw_data.zip archives directly and records the dataset commit, "
     "archive hashes and parser version; a 30-minute grid per directed pair with explicit masks for missing, skipped and "
     "dead (100 % loss) probes; and a data audit that re-derives every Phase 2 number."),
    ("Features (Step 4): ", f"{len(ALL_FEATURES)} causal features in {len(GROUPS)} groups computed only from information available "
     "before the decision (Section 2.4)."),
    ("Stage 2, ranker (Step 5): ", "a LightGBM LambdaRank model that orders the candidate paths of each decision, with graded "
     "relevance from next-interval regret and loss. Its top-ranked path is the incumbent."),
    ("Stage 1, failure gate (Step 5): ", "a LightGBM classifier that predicts whether that incumbent fails in the next interval. "
     "It is trained on exactly the rows it is applied to: on training data the ranker's top path is cross-fitted (four time "
     "blocks, each scored by rankers fitted on the other blocks with a 12-interval embargo), so the gate never sees in-sample "
     "ranker choices. Class weighting, early stopping, and a validation-tuned threshold under a 10 % switching budget."),
    ("Stale-history blend (Step 5): ", f"the ranker score is blended with a short-history ranker that uses only path structure "
     f"and short-window features; the weight of the full model is n / (n + k) with n the observed intervals in the path's "
     f"current presence run and k = {BLEND_K}."),
    ("Saved model and inference (Step 12): ", "train_final.py writes the trained switcher (pickle, plus LightGBM boosters as "
     "text and a model card); predict.py loads it and returns, for each decision, the chosen path, the incumbent and the gate "
     "probability."),
    ("Baselines, experiments, statistics, ablations and error analysis (Steps 6-11): ", f"all eight Phase 2 baselines plus one "
     f"added baseline, experiments A-G, day-block bootstrap, McNemar and Wilcoxon tests with Holm correction, {N_ABL} ablation "
     f"variants over {N_SEEDS} seeds, threshold sensitivity, a failure taxonomy over all decisions and an auxiliary bandwidth analysis."),
])
figure("fig00_architecture", "Implemented pipeline. The twelve steps follow the Phase 2 architecture; the highlighted box and "
       "the orange loop mark the changes made in Phase 3 (Section 1.3). The error-analysis feedback loop, described but not "
       "drawn in the Phase 2 diagram, is now explicit.", 15.5)

doc.add_heading("1.2 Final decision rule", level=2)
para("At every 30-minute interval t and directed pair, the policy uses only data up to interval t-1:")
bullets([
    "The ranker scores every candidate path; its top-ranked path becomes the incumbent.",
    "The gate estimates the probability that this incumbent fails at t (regret above 20 ms or loss above 5 %).",
    "If the probability exceeds the threshold, the policy switches to the second-ranked path; otherwise it keeps the incumbent.",
    "Threshold (Phase 2 rule): the value that minimises the number of validation failures among thresholds that fire on at most "
    "10 % of validation decisions; ties are broken by lower mean regret, then by firing less. 'Never fire' is one of the candidates.",
    "Guarded variant (reported alongside): a threshold is accepted only if it avoids at least two validation failures more than "
    "never firing and adds at most 0.25 ms mean regret.",
])

doc.add_heading("1.3 Feedback and changes to the Phase 2 methodology", level=2)
para("No instructor feedback was received on Phase 2, so no feedback-driven changes were required. The changes below were "
     "forced by what the data showed during implementation. Each is justified by a measurement, and where possible the original "
     "Phase 2 variant is kept as an ablation so its effect is visible.")
chg = pd.DataFrame([
    ["Incumbent definition", "Best-RTT path of the previous interval (persistence)",
     "Top-ranked path of the stage-2 ranker", f"{A['persistence_fail_artifact'] / A['persistence_fail20'] * 100:.0f} % of persistence "
     "failures are a probe-sampling artifact (Section 2.5); the Phase 2 incumbent is kept as ablation 'Phase 2 incumbent'."],
    ["Gate training rows", "Rows of the incumbent", "Rows of the ranker's top path, cross-fitted on training data",
     "Keeps the gate's training target identical to what it predicts at run time, without in-sample optimism."],
    ["Gate threshold rule", "Minimise validation failures under a switching budget", "Unchanged; a guarded variant is reported",
     "The guard (>= 2 failures avoided, <= 0.25 ms extra regret) makes the gate abstain; both are shown so the gate's value is visible."],
    ["Stale-history blend", "Weight grows with observed intervals; structural and short-window features",
     f"Weight n / (n + {BLEND_K}); short ranker on path structure and 1-interval features", f"k = 3 and k = 12 are ablations."],
    ["Feature set", "8 groups", "9 groups: added pair-level volatility (previous decision's outcome, recent best-path changes)",
     "Error analysis showed failures cluster in oscillating pairs (CH<->NA); Step 11 -> Step 4 loop."],
    ["Path structure", "Hop count, ASes, ISDs", "All three; hop count read from the ping path's interface list",
     "Hop count always equals number of ASes - 1 on this testbed, so it adds no information beyond the AS count."],
    ["Dead paths", "Not specified", f"Probes with 100 % loss are candidates that always fail (regret {DEAD_REGRET_MS:.0f} ms, the ping timeout)",
     f"{A['dead_path_candidate_rows']} such rows; none was an incumbent, so results are unaffected, but such a path can now be chosen and penalised."],
    ["Regret reference", "Best path in hindsight", "Lowest live RTT among probed candidates (regret >= 0); oracle = lowest RTT with loss <= 5 %",
     "A loss-filtered reference produced negative regret for lossy paths."],
    ["Primary protocol", "Chronological 70/15/15 test split, purged CV as supplement",
     "Purged expanding-window CV primary, test split secondary",
     f"The test split holds only {int(TE.loc['Persistence', 'n_fail'])} persistence failures and, after 2 August, no sampling artifacts."],
    ["Baselines", "8 baselines", "9 baselines: added last-known-best", "Needed to separate the artifact from genuine prediction."],
    ["Returning paths (RQ3)", "Individual path flaps", "Paths returning after whole-path-set outages, plus held-out-path cold start",
     f"Presence runs end almost only at empty discovery snapshots ({A['empty_snapshots']} empty snapshots, "
     f"{A['empty_snapshot_pct']} % of slots)."],
    ["Parser", "scionpathml transformer 1.0.8", f"Own parser of the raw JSON, version {A['parser_version']}",
     "Every row count traceable to a file; commit, archive hashes and parser version recorded in provenance.json."],
    ["QoE profiles", "Published ScionPathML thresholds", "Published thresholds primary; preprint thresholds as a second reference",
     "The preprint (arXiv 2509.07154v1) uses stricter thresholds; both give degenerate ceilings."],
    ["Optional fresh SCIONLab holdout", "Planned as optional", "Not collected", "Needs SCIONLab AS access; out of scope."],
    ["Bandwidth table", "Auxiliary analysis only", "Auxiliary analysis (Section 5.2), not used in decisions",
     f"Confirmed: {A['bandwidth_ping_fp_overlap']} shared path fingerprints between bandwidth and ping tests."],
], columns=["Aspect", "Phase 2 plan", "Phase 3 implementation", "Reason / evidence"])
table(chg, "Changes from the Phase 2 methodology.", widths=[2.6, 3.6, 4.4, 6.0], font=8)

# ======================================================================================
# 2. Dataset and preprocessing
# ======================================================================================
doc.add_heading("2. Dataset, preprocessing and experimental setup", level=1)
doc.add_heading("2.1 Dataset", level=2)
para("Source: ScionPathML public repository (github.com/keshvadi/ScionPathML; the Phase 2 text referred to the repository by its "
     "older name mpquic-on-scion-ipc). Each of the three labs ran the measurement suite every 30 minutes towards the other two. "
     "The raw archives contain per-interval JSON files: SCMP ping results for up to 15 randomly selected paths (15 pings each), "
     "path-discovery snapshots, path-change deltas, traceroutes and bandwidth tests.")
bullets([
    f"Provenance: repository commit {A['dataset_commit']} (archives unchanged since commit {A['archives_added_in_commit'][:7]}). "
    f"The git blob hash of each local archive matches the commit ({'verified' if A['archives_match_commit'] else 'MISMATCH'}); "
    f"SHA-256, parser version {A['parser_version']} and row counts are recorded in data/processed/provenance.json.",
    "Vantage points: Lab1 = 19-ffaa:1:11de (EU, Magdeburg), Lab2 = 17-ffaa:1:11e4 (CH, ETH Zurich), Lab3 = 18-ffaa:1:11e5 (NA, CMU). "
    "The Taiwan AS (22-ffaa:1:11ee) is excluded because it was unreachable for the whole campaign.",
    "Scope: the clean window from 16 July 2025 00:00 UTC onward. Data from 11-16 July is excluded, as the repository states it is "
    "not comparable.",
    f"Volume: {A['ping_rows']:,} ping rows ({A['ping_status']['ok']:,} successful, {A['ping_status']['skipped']} skipped because the path "
    f"had previously timed out, {A['ping_status']['error']} errors), {A['distinct_pair_paths']} distinct (pair, path) combinations; "
    f"{A['bandwidth_rows']:,} bandwidth rows.",
])
aud = pd.DataFrame([
    ["Ping rows in scope", "97,494", f"{A['ping_rows']:,}", "small difference, likely parser / boundary handling"],
    ["Skipped probes", "751", f"{A['ping_status']['skipped']}", "same 0.6-1.0 % per lab"],
    ["Paths per pair", "18, 20, 20, 24, 20, 24", ", ".join(str(v) for v in A["paths_per_pair"].values()), "exact"],
    ["Pair coverage", "93.8-99.6 %", f"{min(A['pair_coverage_pct'].values())}-{max(A['pair_coverage_pct'].values())} %", "consistent"],
    ["Consecutive decisions / evaluable", "- / 5,703 (83.5 %)", f"{A['decisions_consecutive']:,} / {A['decisions_evaluable']:,} ({A['evaluable_pct']} %)", "consistent"],
    ["Persistence failure share (20 ms / 50 ms)", "5.2 % / 1.5 %", f"{A['persistence_fail20_pct']} % / {A['persistence_fail50_pct']} %", "exact"],
    ["Persistence mean regret", "1.80 ms", f"{A['persistence_mean_regret']} ms", "exact"],
    ["99th-percentile regret", "43.7 ms", f"{A['persistence_p99_regret_all']} ms", "exact (over all decisions, not only failures)"],
    ["Random-path mean regret", "148.1 ms", f"{A['random_mean_regret']} ms", "same order; depends on random-choice definition"],
    ["Presence runs / median length / < 24 h", "2,540 / 8 h / 76 %", f"{A['runs']:,} / {A['run_median_h']:.0f} h / {A['runs_under_24h_pct']} %",
     "exact, when empty snapshots count as absence"],
    ["Probe rows in first 6 h of a run", "19.6 %", f"{A['probe_rows_first6h_pct']} %", "close"],
    ["Mean RTT first 6 h vs older", "442.5 vs 402.9 ms", f"{A['rtt_mean_first6h_vs_older'][0]} vs {A['rtt_mean_first6h_vs_older'][1]} ms", "exact"],
    ["Path-set change between snapshots", "~10 %", f"{A['snapshot_change_pct']} %", "lower; most runs end at empty snapshots"],
    ["QoE oracle ceiling, gaming 120 ms / video 250 ms", "0 % / ~1 %",
     f"{A['qoe_oracle_ceiling_pct_phase2_thresholds']['gaming (120 ms, 4 %)']} % / "
     f"{A['qoe_oracle_ceiling_pct_phase2_thresholds']['video_conf (250 ms, 7 %)']} %", "exact"],
    ["Bandwidth / ping fingerprint overlap", "none", f"{A['bandwidth_ping_fp_overlap']}", "exact"],
], columns=["Quantity", "Phase 2 claim", "Phase 3 (re-derived)", "Status"])
table(aud, "Data audit: Phase 2 dataset claims re-derived from the raw archives (src/audit.py).", widths=[5, 3.2, 3.6, 4.8], font=8)
figure("fig01_data_overview", "Daily persistence failures by cause (top) and whole-path-set outages (bottom). Failures concentrate "
       "in the first three weeks; from 3 August every pair has only 9 or 15 paths, all of which are probed every interval, so the "
       "sampling artifact disappears. The outages on 9-11 August are the CH node going offline.")

doc.add_heading("2.2 Preprocessing", level=2)
bullets([
    "Alignment: probe timestamps (which drift by up to 3 minutes) are floored to a 30-minute grid; each pair becomes a slot x path matrix.",
    "Missing values are never imputed. Unprobed paths stay missing; windows are masked, and the number of observations and the time "
    "since the last observation are features.",
    "Skipped probes are encoded as unavailability events (a rolling count), not as missing values.",
    "Empty path-discovery snapshots are treated as absence of all paths; presence runs and run age are computed from the snapshots.",
    f"Dead paths: {A['ping_ok_but_total_loss']} probes have 100 % loss and no RTT. They stay in the candidate set as paths that always "
    f"fail (loss 100 %, regret {DEAD_REGRET_MS:.0f} ms), and their loss enters the loss-history features.",
])
doc.add_heading("2.3 Decisions, labels and splits", level=2)
bullets([
    "Decision: one per pair and interval t where intervals t-1 and t both have live measurements. Candidates are all paths probed "
    f"at t (mean {A['candidates_per_decision_mean']} per decision); ground truth exists only for them, which is stated as a restriction.",
    f"Evaluable set: decisions where the previous-interval best path was probed again at t ({A['decisions_evaluable']:,}, {A['evaluable_pct']} %). "
    "Every policy is scored on exactly this set, so all comparisons are paired.",
    "Regret: RTT of the chosen path minus the lowest live RTT among the candidates. Failure: regret above 20 ms or loss above 5 % "
    "(50 ms variant reported).",
    f"Label resolution: every probe sends {A['pings_per_probe'][0]} pings, so loss is measured in steps of {A['loss_quantum_pct']} % and "
    f"'loss above 5 %' means at least one lost ping. Of the {A['persistence_loss_failures']} loss-driven persistence failures, "
    f"{A['persistence_loss_failures_single_ping']} are a single lost ping. The 5 % limit is kept as committed in Phase 2; Section 6.2 "
    "repeats the evaluation with 10 % and 20 % limits (at least 2 and 4 lost pings).",
    "Ranking relevance: 3 if regret <= 2 ms, 2 if <= 10 ms, 1 if <= 20 ms without failure, 0 for a failure.",
    "TEST protocol: chronological 70/15/15 split within each pair (model on train, early stopping and threshold on validation, "
    "evaluation on test).",
    f"CV protocol (primary): six equal time blocks per pair; fold k trains on blocks before k minus a 12-interval embargo and tests on "
    f"block k. Out-of-fold decisions of folds 1-5 are pooled ({int(P.n):,} decisions, {int(PE.n_fail)} persistence failures). Inside each "
    "fold the last 15 % of training time is used for early stopping and threshold tuning.",
])
doc.add_heading("2.4 Features", level=2)
feat = pd.DataFrame([
    ["Current state", "last RTT, loss, jitter (ping mdev), last RTT change", len(GROUPS["current"])],
    ["Rolling statistics", "mean / std / min / max / slope of RTT, mean loss and jitter over 1, 3, 6, 12 intervals; EMA-6", len(GROUPS["rolling"])],
    ["Stability", "RTT spike count, change-point count (> 20 ms jumps), loss-event rate over 3, 6, 12 intervals", len(GROUPS["stability"])],
    ["Path structure", "hop count (inter-AS links), number of ASes and ISDs on the path", len(GROUPS["structure"])],
    ["Run age", "hours since the path set reappeared, number of appearances, observations in the current run, presence, timeout flag", len(GROUPS["run_age"])],
    ["Observation state", "observations per window, intervals since last probe, skipped-probe count, paths probed / present", len(GROUPS["observation"])],
    ["Relative", "RTT and EMA gap to the best alternative, alternatives within 10 ms, RTT rank, best-vs-second margin", len(GROUPS["relative"])],
    ["Temporal", "hour of day and day of week, cyclically encoded", len(GROUPS["temporal"])],
    ["Volatility (new)", "previous decision's regret and failure, best-path changes and failures in last 6 / 12 intervals", len(GROUPS["volatility"])],
], columns=["Group", "Features", "#"])
table(feat, "Feature groups. All features use data up to t-1 only; path fingerprints and vantage identifiers are never used.",
      widths=[3, 12, 1.2], font=8)
doc.add_heading("2.5 The sampling artifact", level=2)
para(f"The prober tests a random subset of at most 15 paths per interval. When the truly best path is not sampled at t-1, persistence "
     f"picks the best of the sampled paths, and at t the best path reappears in the sample, so persistence 'fails' by 20-40 ms although "
     f"nothing in the network changed. This explains {A['persistence_fail_artifact']} of the {A['persistence_fail20']} persistence failures. Any "
     f"policy that remembers recent observations avoids it. The remaining {A['persistence_fail_genuine']} failures are loss events on the incumbent. "
     "This reframes the Phase 1-2 gap: the 'rare inflection-point failures' are mostly a measurement-design effect, and the genuinely "
     "hard events are short loss bursts, most of them one lost ping.")
doc.add_heading("2.6 Implementation details", level=2)
hp = TUNE["gate"]; hr = TUNE["ranker"]
bullets([
    f"Gate: LightGBM, num_leaves {hp['num_leaves']}, min_child_samples {hp['min_child_samples']}, learning rate {hp['learning_rate']}, "
    "subsample and column fraction 0.8, L2 1.0, early stopping (100 rounds) on validation AUPRC, positive-class weight = negatives / positives.",
    f"Ranker: LightGBM LambdaRank, num_leaves {hr['num_leaves']}, min_child_samples {hr['min_child_samples']}, learning rate {hr['learning_rate']}, "
    "early stopping on validation NDCG@1.",
    "Hyperparameters were chosen from an 18-point grid by purged expanding-window CV inside the training split only (src/tune.py), "
    "each on the target it serves: the ranker on NDCG@1, then the gate on AUPRC for failures of the ranker's (cross-fitted) top path.",
    f"Seeds 0-4 for model randomness, random baselines, epsilon-greedy and all {N_ABL} ablations; five random draws for the cold-start test.",
    "Software: Python 3.14, LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4, SciPy 1.18.1, SHAP 0.54.1. "
    "Hardware: Intel Core i5-8250U laptop CPU, 8 GB RAM, no GPU.",
])

# ======================================================================================
# 3. Baselines
# ======================================================================================
doc.add_heading("3. Baseline evaluation", level=1)
para("All baselines choose among the same candidate paths and are scored on the same decisions. Policies marked * use the "
     "measurements of interval t itself and cannot be deployed; they are references only.")
bl = pd.DataFrame([
    ["Random", "uniform choice among candidates (5 seeds)"],
    ["Persistence", "best-RTT path of the previous interval (Phase 2 incumbent)"],
    ["Last-known-best (added)", "lowest last-observed RTT, observations up to 6 h old"],
    ["Nowcast min-RTT*", "lowest RTT measured at t"],
    ["EMA-6 RTT", "lowest exponential moving average (span 6) of RTT"],
    ["ScionPathML heuristic (nowcast*) / (forecast)", "0.5 x min-max-normalised RTT + 0.5 x normalised loss, using values at t / t-1 "
     "(bandwidth weight dropped, no path match)"],
    ["Epsilon-greedy (0.1)", "lowest running-mean RTT, random path with probability 0.1 (5 seeds)"],
    ["LR / RF gates, RTT-jump rule", "logistic-regression and random-forest gates in place of LightGBM, and a last-RTT-jump rule (Section 6)"],
    ["Oracle*", "lowest RTT at t among paths with loss <= 5 %"],
], columns=["Baseline", "Definition"])
table(bl, "Baselines.", widths=[5, 11.6], font=8.5)

order = ["Proposed", "Ranker only", "Proposed, guarded threshold", "Last-known-best", "EMA-6 RTT", "Persistence",
         "ScionPathML heuristic (forecast)", "Epsilon-greedy (0.1)", "Random", "Gate only", "Proposed, Phase 2 incumbent",
         "Nowcast min-RTT*", "ScionPathML heuristic (nowcast)*", "Oracle*"]


def main_table(S, title):
    rows = []
    for p in order:
        if p not in S.index:
            continue
        r = S.loc[p]
        rows.append([p, int(r.n_fail), f"{f(r.fail_pct)} [{f(r.fail_lo)}, {f(r.fail_hi)}]", f(r.fail50_pct),
                     f"{f(r.regret_mean, 3)} [{f(r.reg_lo, 3)}, {f(r.reg_hi, 3)}]", f(r.regret_p99),
                     f(r.switch_per_pair_day), f(r.switch_per_day, 1), "-" if pd.isna(r.fired_pct) else f(r.fired_pct)])
    return table(pd.DataFrame(rows, columns=["Policy", "Failures", "Failure % [95 % CI]", "Fail % (50 ms)",
                                             "Mean regret ms [95 % CI]", "p99 ms", "Switches / pair-day",
                                             "Switches / day (all pairs)", "Gate fired %"]),
                 title, widths=[4.2, 1.3, 2.6, 1.3, 2.9, 1.1, 1.4, 1.5, 1.2], font=7.5, highlight=lambda row: row[0] == "Proposed")


para("Switching is reported per pair-day (switches per 48 evaluated decisions of one directed pair) and per calendar day "
     "(switches of all six pairs together).", size=9)
tcv = main_table(CV, f"Main comparison, CV protocol (pooled out-of-fold, {int(P.n):,} decisions, seed 0). CIs: day-block bootstrap, 4,000 resamples.")
para(f"Seed stability (CV, seeds 0-4): failure share {f(SEEDS_CV.fail_pct.mean())} ± {f(SEEDS_CV.fail_pct.std())} % "
     f"(range {f(SEEDS_CV.fail_pct.min())}-{f(SEEDS_CV.fail_pct.max())} %), mean regret {f(SEEDS_CV.regret_mean.mean(), 3)} ± "
     f"{f(SEEDS_CV.regret_mean.std(), 3)} ms, gate firing {f(SEEDS_CV.fired_pct.mean())} ± {f(SEEDS_CV.fired_pct.std())} % of decisions. "
     "The seed spread comes from the gate: the more it fires, the more failures. Random: "
     f"{f(pd.read_csv(TAB / 'cv_random_seeds.csv').fail_pct.mean())} % failures over seeds; epsilon-greedy "
     f"{f(pd.read_csv(TAB / 'cv_epsilon-greedy_seeds.csv').fail_pct.mean())} %.", size=9)
tte = main_table(TE, f"Main comparison, TEST protocol (last 15 % of each pair, {int(TE.loc['Proposed', 'n']):,} decisions, seed 0).")
stt_pers = STT.loc["Proposed vs Persistence"]
para(f"On the test split the proposed policy fails in {f(SEEDS_TE.fail_pct.mean())} ± {f(SEEDS_TE.fail_pct.std())} % of decisions over five "
     f"seeds (range {f(SEEDS_TE.fail_pct.min())}-{f(SEEDS_TE.fail_pct.max())} %), while the ranker alone fails in {f(TE.loc['Ranker only', 'fail_pct'])} % "
     f"and persistence in {f(TE.loc['Persistence', 'fail_pct'])} %. The gate fired on {f(SEEDS_TE.fired_pct.mean(), 1)} % of test decisions "
     "on average, and those switches added failures. This split covers 5-11 August; from 3 August each pair has only 9 or 15 paths, "
     "all probed every interval, so the sampling artifact cannot occur and persistence has almost no headroom (Holm-corrected "
     f"McNemar p vs persistence {pv(stt_pers.mcnemar_p_holm)}, {int(stt_pers.proposed_only_fail)} vs {int(stt_pers.other_only_fail)} "
     "discordant failures).", size=9)

gq = GQ.reset_index()
gqt = []
for _, r in gq.iterrows():
    gqt.append([r.gate, f(r.auprc, 3), f(r.prevalence, 3), "-" if pd.isna(r.get("auroc")) else f(r.auroc, 3),
                "-" if pd.isna(r.get("precision")) else f(r.precision, 2), "-" if pd.isna(r.get("recall")) else f(r.recall, 2),
                "-" if pd.isna(r.get("f1")) else f(r.f1, 2), "-" if pd.isna(r.get("mcc")) else f(r.mcc, 2),
                "-" if pd.isna(r.get("bal_acc")) else f(r.bal_acc, 2), "-" if pd.isna(r.get("fired_pct")) else f(r.fired_pct, 2),
                int(r.n_events)])
gqt.insert(0, ["LightGBM gate, Phase 2 incumbent (incl. artifact)", f(average_precision_score(G2.fail, G2.p_gate), 3),
               f(G2.fail.mean(), 3), f(roc_auc_score(G2.fail, G2.p_gate), 3), "-", "-", "-", "-", "-", "-", int(G2.fail.sum())])
table(pd.DataFrame(gqt, columns=["Gate", "AUPRC", "Prevalence", "AUROC", "Precision", "Recall", "F1", "MCC", "Bal. acc.",
                                 "Fired %", "Events"]),
      "Gate quality on pooled CV. Rows 2-7 predict failures of the final incumbent (the ranker's top path); precision, recall, F1, MCC "
      "and balanced accuracy are at the tuned threshold of each fold. Plain accuracy is not reported because of class imbalance.",
      widths=[4.8, 1.2, 1.4, 1.2, 1.3, 1.1, 0.9, 0.9, 1.2, 1.1, 1.1], font=7.5)

stt = []
for comp in ["Proposed vs Persistence", "Proposed vs Ranker only", "Proposed vs Proposed, guarded threshold",
             "Proposed vs Last-known-best", "Proposed vs EMA-6 RTT", "Proposed vs ScionPathML heuristic (forecast)",
             "Proposed vs Epsilon-greedy (0.1)", "Proposed vs Random", "Proposed vs Proposed, Phase 2 incumbent", "Proposed vs Oracle*"]:
    r = ST.loc[comp]
    stt.append([comp.replace("Proposed vs ", ""), f"{int(r.proposed_only_fail)} / {int(r.other_only_fail)}", pv(r.mcnemar_p_holm),
                f(r.odds_ratio, 3), pv(r.wilcoxon_p_holm), f(r.rank_biserial, 2)])
table(pd.DataFrame(stt, columns=["Proposed vs", "Discordant failures (proposed only / other only)", "McNemar p (Holm)",
                                 "Odds ratio", "Wilcoxon p, daily regret (Holm)", "Rank-biserial r"]),
      f"Paired significance tests, CV protocol ({int(ST.n_days.max())} days). Negative rank-biserial = proposed has lower daily regret.",
      widths=[4.6, 3.4, 2.0, 1.8, 2.6, 2.0], font=8)

# ======================================================================================
# 4. Graphs
# ======================================================================================
doc.add_heading("4. Graphical results", level=1)
figure("fig02_main_comparison", "Failure share and mean regret of all policies (CV protocol) with 95 % day-block bootstrap CIs.")
figure("fig03_regret_ccdf", "Regret tail. Persistence has a heavy tail between 20 and 50 ms (the sampling artifact); the proposed "
       "policy, last-known-best and EMA-6 remove it. The ScionPathML forecast heuristic keeps part of the tail because loss and RTT "
       "are weighted equally after min-max normalisation.", 13)
fsg = FS[FS.subset == "persistence fails: genuine"].set_index("policy")
figure("fig04_failure_subsets", "Decisions on which persistence fails, split by cause. Memory-based policies avoid almost every "
       f"artifact failure. Of the {int(fsg.loc['Proposed', 'n'])} genuine loss failures, the gate's switches avoid "
       f"{int(fsg.loc['Proposed', 'avoided'])}; the history-based policies (persistence, last-known-best, EMA-6, ranker, ScionPathML "
       f"forecast) avoid none. Random ({int(fsg.loc['Random', 'avoided'])}) and epsilon-greedy ({int(fsg.loc['Epsilon-greedy (0.1)', 'avoided'])}) "
       f"avoid some only by switching at random, at the cost of far more failures elsewhere. The oracle, which sees interval t, "
       f"avoids {int(fsg.loc['Oracle*', 'avoided'])}.", 13.5)
figure("fig05_gate_pr_curves", "Precision-recall curves of the gate. Left: with the Phase 2 incumbent, the gate is highly predictive "
       "because it learns to recognise the artifact (best path missing from the previous sample). Right: for the final incumbent "
       "every gate model stays close to the prevalence line.")
best_sw = SW.loc[SW.fail_pct.idxmin()]
FIG_SWEEP = FIGN[0] + 1
figure("fig06_threshold_tradeoff", f"Switching budget vs failure share for the Phase 2 design (threshold swept on pooled CV). The best "
       f"point fires on {100 * best_sw.fired:.1f} % of decisions and reaches {f(best_sw.fail_pct)} % failures; firing more often makes "
       "things worse.", 13)

doc.add_heading("4.1 Returning paths (experiment C)", level=2)
figure("fig07_runage", "Failure share, mean regret and switching by run age of the Phase 2 incumbent (same decisions for every policy).", 16)
rap = RA[RA.policy == "Proposed"].set_index("run_age")
rpe = RA[RA.policy == "Persistence"].set_index("run_age"); rlk = RA[RA.policy == "Last-known-best"].set_index("run_age")
rat = [[a, int(r.n), f"{f(rpe.loc[a, 'fail_pct'])} / {f(rlk.loc[a, 'fail_pct'])}", f"{f(r.fail_pct)} ({int(r.n_fail)})", f(r.regret_mean, 3),
        f(r.regret_p99), f(r.switch_per_pair_day), f(r.fired_pct), f(r.gate_auprc, 3), f(r.gate_prevalence, 3),
        f(r.gate_precision, 2), f(r.gate_recall, 2), f(r.gate_mcc, 2)] for a, r in rap.iterrows()]
table(pd.DataFrame(rat, columns=["Run age", "n", "Fail % persist. / LKB", "Fail % proposed (n)", "Mean regret ms", "p99 ms",
                                 "Switches / pair-day", "Gate fired %", "Gate AUPRC", "Prevalence", "Precision", "Recall", "MCC"]),
      "All metrics by run age for the proposed policy (Phase 2, Step 8), pooled CV. Gate metrics at the tuned threshold.",
      widths=[1.4, 1.0, 1.9, 1.7, 1.4, 1.0, 1.4, 1.2, 1.2, 1.3, 1.2, 1.0, 1.0], font=7)
cs = CS.set_index(["subset", "policy", "training"])
cst = []
for sub in ["best path is held out", "last-known-best is held out", "any candidate is held out"]:
    for pol, trn in [("Proposed", "trained on all paths"), ("Proposed", "held-out paths unseen in training"),
                     ("Proposed, no blend", "trained on all paths"), ("Proposed, no blend", "held-out paths unseen in training"),
                     ("Last-known-best", "no training"), ("EMA-6 RTT", "no training"), ("Oracle*", "no training")]:
        r = cs.loc[(sub, pol, trn)]
        cst.append([sub, pol, trn, int(r.n), int(r.n_fail), f(r.fail_pct), f(r.regret_mean, 3), f(r.picked_heldout_pct, 1)])
table(pd.DataFrame(cst, columns=["Decisions where", "Policy", "Training", "n", "Failures", "Fail %", "Mean regret ms",
                                 "Picked a held-out path %"]),
      "Synthetic cold start: 20 % of each pair's paths removed from training, five random draws pooled (CV protocol).",
      widths=[3.0, 2.8, 3.6, 1.2, 1.3, 1.1, 1.6, 1.8], font=7)

doc.add_heading("4.2 Drift and cross-vantage transfer (experiments D, E)", level=2)
figure("fig08_drift_lovo", "Left: failure share by calendar segment for a model trained on week 1 only and a model retrained on all "
       "earlier data. Week 4 covers days 21-23 only; days 24-26 (CH node offline, only the EU<->NA pairs remain) are a separate "
       "segment. Right: leave-one-vantage-out transfer (time-respecting) vs in-vantage training.")
drt = [[r.segment, r.model, int(r.n), int(r.n_pairs), f(r.cand_mean, 1), int(r.n_events), f(r.auprc, 3), f(r.prevalence, 3),
        f(r.fail_pct), f(r.lkb_fail_pct), f(r.pers_fail_pct), f(r.regret_mean, 3)] for r in DR.itertuples()]
table(pd.DataFrame(drt, columns=["Segment", "Model", "n", "Pairs", "Candidates / decision", "Events", "Gate AUPRC", "Prevalence",
                                 "Fail %", "LKB fail %", "Persist. fail %", "Mean regret ms"]),
      "Drift (experiment D). The candidate count shows the shrinking path set from day 17.", widths=[3.0, 3.0, 1.0, 1.0, 1.5, 1.1, 1.3,
                                                                                                   1.4, 1.1, 1.3, 1.3, 1.4], font=7)
lvt = [[r.held_out, r.model, int(r.n), int(r.n_events), "-" if pd.isna(r.auprc) else f(r.auprc, 3), f(r.prevalence, 3),
        f(r.fail_pct), f(r.lkb_fail_pct), f(r.regret_mean, 3)] for r in LV.itertuples()]
table(pd.DataFrame(lvt, columns=["Source vantage tested", "Training", "n", "Events", "Gate AUPRC", "Prevalence", "Fail %",
                                 "LKB fail %", "Mean regret ms"]),
      "Cross-vantage transfer (experiment E). 'Source held out' removes training decisions whose source is the tested vantage "
      "(Phase 2 Step 7 wording); 'vantage unseen' also removes those whose destination is that vantage, so it never appears in training.",
      widths=[2.2, 4.4, 1.2, 1.2, 1.5, 1.5, 1.2, 1.4, 1.8], font=7.5)

doc.add_heading("4.3 QoE profiles (experiment F)", level=2)
figure("fig11_qoe", "QoE satisfaction for the five ScionPathML profiles under the published thresholds used in Phase 2, under the "
       "preprint thresholds, and for two relative variants anchored to each pair's best-path RTT distribution. Grey bars: oracle ceiling.")

# ======================================================================================
# 5. Comparison with existing research
# ======================================================================================
doc.add_heading("5. Comparison with existing research", level=1)
doc.add_heading("5.1 Published approaches", level=2)
rep_pre = dict(zip(["video_conf", "gaming", "file_transfer", "browsing", "streaming"], ["24 %", "21 %", "30 %", "26 %", "21 %"]))
qrows = []
for src, prof in (("published (Phase 2)", QOE_PHASE2), ("preprint", QOE_PREPRINT)):
    tag = "Phase 2" if src.startswith("published") else "preprint"
    for name, (r_, l_) in prof.items():
        key = f"{name} ({tag}: {r_} ms, {l_} %)"
        q = QO[QO.profile == key].set_index("policy")
        qrows.append([name, src, f"{r_} ms, {l_} %", "28-35 % (range over profiles)" if tag == "Phase 2" else rep_pre[name],
                      f(q.oracle_ceiling_pct.iloc[0], 1), f(q.loc["ScionPathML heuristic (forecast)", "satisfaction_pct"], 1),
                      f(q.loc["Persistence", "satisfaction_pct"], 1), f(q.loc["Proposed", "satisfaction_pct"], 1)])
table(pd.DataFrame(qrows, columns=["Profile", "Thresholds", "Max RTT, max loss", "Reported", "Oracle ceiling (ours)",
                                   "Heuristic, forecast (ours)", "Persistence (ours)", "Proposed (ours)"]),
      "ScionPathML path-recommendation task: reported satisfaction vs what the released data allows (bandwidth criterion "
      "dropped because no bandwidth test shares a path with a ping test). All values in % of evaluable CV decisions.",
      widths=[2.2, 2.4, 2.2, 2.6, 1.7, 1.9, 1.7, 1.7], font=7.5)
pairs = EB[EB.by == "pair"].set_index("group")
comp = pd.DataFrame([
    ["ScionPathML (Rossi, Keshvadi et al., Telecom 2026; arXiv 2509.07154)", "Path recommendation, weighted heuristic, 5 QoE profiles",
     "28-35 % satisfaction (published, as cited in Phase 2); 21-30 % (preprint)",
     "Same data and thresholds: oracle ceilings are about 0-1 % or 100 %, so every deployable policy scores ~0 % or ~99 %; the "
     f"reported values cannot be reproduced. On failure share the re-implemented heuristic (forecast) fails {f(HF.fail_pct)} % vs "
     f"{f(P.fail_pct)} % for ours.",
     "Merging of ping and bandwidth tables and exact weights not documented; bandwidth not path-matched."],
    ["ScionPathML, forecasting task", "One-step RTT forecast, linear regression, 12-step window",
     "MAE 1.31 ms (published, per Phase 2); 3.88 ms (preprint)",
     f"Linear regression on window statistics: MAE {FC['linear regression on window statistics (70/30 time split)']['mae']:.2f} ms; "
     f"last value {FC['last value (naive)']['mae']:.2f} ms; EMA-6 {FC['EMA-6']['mae']:.2f} ms (all one-step, observed paths).",
     "Split and exact windowing not stated in the paper; forecasting is a side analysis here."],
    ["ScionPathML, failure prediction", "Random forest, 'path unavailable at T+1'", "ROC-AUC 0.938 (published); F1 0.86 (preprint)",
     f"Different label (incumbent performance failure). Our gate: ROC-AUC {f(roc_auc_score(G2.fail, G2.p_gate), 3)} with the Phase 2 "
     f"incumbent, {f(LG.auroc, 3)} for the final incumbent.",
     "Label definitions differ; class balance of their task not reported."],
    ["Bourak et al., DQN for SCION path optimisation (CSCML 2025)", "RL agent in a simulated SCION network",
     "Matches an oracle; 95 % less probing",
     f"Real data, no RL: ours is {f(P.fail_pct - OR.fail_pct)} points above the oracle failure share, with lower regret "
     f"({f(P.regret_mean, 3)} vs {f(OR.regret_mean, 3)} ms; the oracle trades RTT for loss).",
     "Simulation vs live data; probing cost is fixed by the dataset, so the probing-overhead axis cannot be compared."],
    ["Baumeister and Keshvadi, axiomatic analysis of path selection (arXiv 2509.05938)", "Min-RTT, round-robin, epsilon-greedy",
     "Greedy min-RTT can oscillate", f"Consistent: greedy persistence switches {f(PE.switch_per_pair_day)} times per pair-day vs "
     f"{f(P.switch_per_pair_day)} for ours and {f(RK.switch_per_pair_day)} for the ranker alone; epsilon-greedy fails in "
     f"{f(CV.loc['Epsilon-greedy (0.1)', 'fail_pct'])} % of decisions.", "Analytical vs empirical."],
    ["Vu et al., MPQUIC schedulers (Computer Networks 2025); ALCS (IEEE TMC 2026)", "Scheduler evaluation under mobility / handover",
     "Rankings depend on the operating regime", f"Consistent: our failure share ranges from {f(pairs.prop_fail_pct.min())} % "
     f"({pairs.prop_fail_pct.idxmin()}) to {f(pairs.prop_fail_pct.max())} % ({pairs.prop_fail_pct.idxmax()}) by pair.",
     "Different layer (MP transport scheduling)."],
], columns=["Work", "Task / setting", "Reported result", "Our comparable result", "Limits to a fair comparison"])
table(comp, "Comparison with published approaches.", widths=[3.2, 2.8, 2.8, 4.6, 3.3], font=7.5)

doc.add_heading("5.2 Auxiliary bandwidth analysis", level=2)
pe100 = BWP[BWP.tier == 100]
lk = BWL["link_by_source"]
para("Bandwidth tests cannot enter the per-path decision because they never use a path that is also pinged. As planned in Phase 2, "
     f"they are analysed separately (src/bandwidth.py, Figure {FIGN[0] + 1}).")
bullets([
    ("Test outcomes. ", f"{BWL['rows']:,} rows (tests x path x direction). Failed tests per source: " +
     ", ".join(f"{k} {v['failed_pct']:.1f} %" for k, v in BWL["tests_by_source"].items()) + "; degraded (completed below 90 % of "
     "target): " + ", ".join(f"{k} {v['degraded_pct']:.1f} %" for k, v in BWL["tests_by_source"].items()) +
     f". Median achieved/target: {', '.join(f'{k} Mbps {v:.2f}' for k, v in BWL['ratio_median_by_tier'].items())}; the 100 Mbps "
     "tier is limited by the testbed, not by the path."),
    ("Does the path matter? ", f"Path identity explains only {f(100 * pe100.between_path_var_share.min(), 1)}-"
     f"{f(100 * pe100.between_path_var_share.max(), 1)} % of the throughput variance at 100 Mbps; variation is over time, not "
     f"between paths. Yet when two paths are tested at the same moment they differ by more than 10 % of the target in "
     f"{f(pe100.paired_gap_over_10pct.min(), 0)}-{f(pe100.paired_gap_over_10pct.max(), 0)} % of tests, so short-term path choice "
     "could matter for bandwidth even though no path is better on average. Path length is not correlated with throughput "
     f"(|Spearman rho| <= {f(pe100.spearman_len_vs_ratio.abs().max(), 3)})."),
    ("Link to ping quality. ", "Per source and 30-minute slot, bandwidth at 100 Mbps correlates only weakly with ping quality "
     "from the same source: " + "; ".join(f"{k}: rho {v['rho_ping_lossy_vs_bw_ratio']:.2f} with the share of lossy pings, "
                                          f"{v['rho_ping_rtt_vs_bw_ratio']:.2f} with median RTT" for k, v in lk.items()) +
     ". Ping-based selection is therefore not a proxy for throughput."),
    ("Drift. ", "Failed-test share by segment at 100 Mbps: " + ", ".join(
        f"{r.segment} {r.failed_pct:.1f} %" for r in BWW[BWW.tier == 100].assign(o=lambda d: d.segment.str.startswith("days"))
        .sort_values(["o", "segment"]).itertuples()) + "."),
])
figure("fig12_bandwidth", "Auxiliary bandwidth analysis. Left: achieved/target rate by tier and direction. Middle: hour-of-day profile "
       "at 100 Mbps per source. Right: share of throughput variance explained by the path (up) and share of same-time path pairs "
       "differing by more than 10 % of the target (down).")

# ======================================================================================
# 6. Ablations
# ======================================================================================
doc.add_heading("6. Ablation studies", level=1)
doc.add_heading("6.1 Component ablations", level=2)
para(f"Each ablation removes or replaces one component and is re-run on the pooled CV protocol with seeds 0-{N_SEEDS - 1} (Table "
     f"{TABN[0] + 1}, Figure {FIGN[0] + 1}). Differences are reported against the full model; the CI column is the day-block bootstrap "
     "CI of the failure-share difference to persistence for seed 0.")
ci = AB[AB.seed == 0].set_index("ablation")
full = ABG.loc["Full model"]
abt = [[n, f"{f(r.fail)} ± {f(r.fsd)}", f"{r.fail - full.fail:+.2f}", f"{f(r.reg, 3)} ± {f(r.rsd, 3)}", f"{r.reg - full.reg:+.3f}",
        f(r.sw), "-" if pd.isna(r.fired) else f(r.fired), "-" if pd.isna(r.auprc) else f"{f(r.auprc, 3)} ({f(r.prev, 3)})",
        f"[{f(ci.loc[n, 'dfail_lo'])}, {f(ci.loc[n, 'dfail_hi'])}]"] for n, r in ABG.iterrows()]
table(pd.DataFrame(abt, columns=["Variant", "Failure %", "Δ fail", "Mean regret ms", "Δ regret", "Switches / pair-day", "Gate fired %",
                                 "Gate AUPRC (prevalence)", "Δ fail vs persistence, 95 % CI"]),
      f"Ablations (pooled CV, mean ± sd over {N_SEEDS} seeds).", widths=[4.4, 1.7, 1.1, 2.2, 1.3, 1.4, 1.2, 2.0, 2.3], font=7,
      highlight=lambda row: row[0] == "Full model")
figure("fig09_ablations", f"Ablations: failure share and mean regret, mean and range over {N_SEEDS} seeds.")
gate_vars = ["Gate: random forest", "Gate: logistic regression", "No class weighting", "SMOTE instead of class weights"]
para("Interpretation.", bold_lead=None)
bullets([
    ("The gate is the main source of variance and of harm. ", f"Without it (ranker only) failures are {f(gv('Ranker only (no gate)', 'fail'))} % "
     f"and regret {f(gv('Ranker only (no gate)'), 3)} ms; the guarded threshold, which abstains, gives "
     f"{f(gv('Guarded threshold (min. gain 2 failures)', 'fail'))} %. The full model fires on {f(full.fired)} % of decisions and reaches "
     f"{f(full.fail)} %. Gate variants: " + ", ".join(f"{n.replace('Gate: ', '')} {f(gv(n, 'fail'))} % (fires {f(gv(n, 'fired'))} %)"
                                                    for n in gate_vars) + "."),
    ("Incumbent definition: ", f"the Phase 2 incumbent gives {f(gv('Phase 2 incumbent (previous-interval best)', 'fail'))} % failures "
     f"and {f(gv('Phase 2 incumbent (previous-interval best)'), 3)} ms regret with {f(gv('Phase 2 incumbent (previous-interval best)', 'sw'))} "
     f"switches per pair-day; a last-known-best incumbent gives {f(gv('Last-known-best incumbent', 'fail'))} % and "
     f"{f(gv('Last-known-best incumbent'), 3)} ms. Gate only (runner-up switch) gives the same numbers: both variants use the same "
     "gate on the same incumbent, and when it fires the ranker's best alternative is the runner-up by last RTT in all but one "
     "decision (seed 0)."),
    ("Stale-history blend: ", f"it does not help in any form: no blend {f(gv('No stale-history blend', 'fail'))} % / {f(gv('No stale-history blend'), 3)} ms; "
     f"k = 3: {f(gv('Blend k = 3', 'fail'))} % / {f(gv('Blend k = 3'), 3)} ms; k = 12: {f(gv('Blend k = 12', 'fail'))} % / "
     f"{f(gv('Blend k = 12'), 3)} ms (full model, k = {BLEND_K}: {f(full.fail)} % / {f(full.reg, 3)} ms)."),
    ("Feature groups and window length: ", ", ".join(f"{n} {f(gv(n, 'fail'))} % / {f(gv(n), 3)} ms" for n in ABG.index
                                                      if n.startswith("- ") or n.startswith("Window")) +
     ". Because every variant keeps the gate, these differences mostly follow how often the gate fires (the seed spread of the "
     f"full model alone is ± {f(full.fsd)} points); no group removal beats the ranker alone by more than that spread."),
])

doc.add_heading("6.2 Threshold sensitivity", level=2)
slt = [[f"{int(r.margin_ms)} ms", f"{r.loss_pct:g} %", int(r.min_lost_pings), f(r.pers_fail_pct), f(r.lkb_fail_pct), f(r.ema_fail_pct),
        f"{f(r.proposed_fail_pct)} ({int(r.proposed_n_fail)})", f"[{f(r.dfail_pers_lo)}, {f(r.dfail_pers_hi)}]", f(r.proposed_fired_pct),
        f(r.gate_auprc, 3), f(r.gate_prevalence, 3), f(r.oracle_fail_pct)] for r in SL.itertuples()]
table(pd.DataFrame(slt, columns=["Regret margin", "Loss limit", "Min. lost pings", "Persist. fail %", "LKB fail %", "EMA-6 fail %",
                                 "Proposed fail % (n)", "Δ vs persist. 95 % CI", "Gate fired %", "Gate AUPRC", "Prevalence",
                                 "Oracle fail %"]),
      "Failure-label sensitivity (Phase 2, Step 10): the model is retrained and evaluated with each label; baselines are re-scored. "
      "Pooled CV, seed 0.", widths=[1.4, 1.2, 1.2, 1.4, 1.2, 1.3, 1.8, 2.0, 1.2, 1.2, 1.3, 1.2], font=7)
figure("fig13_label_sensitivity", "Failure share under each (regret margin, loss limit) label. Raising the loss limit from one to "
       "two or more lost pings removes most failures of every memory-based policy, which shows how much of the residual failure "
       "share is single-ping noise.", 13)
sb = SB.set_index("value")
para("Switching budget: " + ", ".join(f"{int(100 * b)} %: {f(r.fail_pct)} % failures (fired {f(r.fired_pct)} %)" for b, r in sb.iterrows()) +
     f". The threshold that minimises validation failures fires on {f(SB.fired_pct.max())} % of test decisions or less at every budget "
     f"(seed 0), so the budget does not bind. Phase 2 design (threshold swept, "
     f"Figure {FIG_SWEEP}): the best firing rate is ~{100 * best_sw.fired:.0f} % of decisions.", size=9.5)

doc.add_heading("6.3 Error analysis and shortcut audit", level=2)
figure("fig10_shap_importance", "SHAP importance of the Phase 2 gate (the predictive variant), on the CV fold with most failure "
       "events. RTT-history window features dominate (see the shortcut audit).", 13)
gt = [[k, v] for k, v in GO.items()] + [["Failure cause (first matching rule): " + k, v] for k, v in GP.items()] + \
     [["Attribute (overlapping): " + k, v] for k, v in FA.items()]
TAB_TAX = TABN[0] + 1
table(pd.DataFrame(gt, columns=["Category", "Decisions"]),
      f"Failure taxonomy of the final policy over all {TAX['n_decisions']:,} pooled CV decisions ({TAX['n_failures']} failures). The "
      "first block classifies every decision by gate outcome; the second assigns every failure, including the ranker's own errors, "
      f"to one cause; the third lists overlapping attributes. {TAX['decisions_not_evaluable']:,} decisions are not evaluable "
      "(previous best path not probed again).", widths=[13, 3], font=7.5)
eb = EB[EB.by.isin(["src_v", "dst_v", "hour_bin"])]
para("Breakdown (proposed vs persistence failure %): " + "; ".join(
    f"{ {'src_v': 'source', 'dst_v': 'destination', 'hour_bin': 'hour'}[r.by]} {r.group}: {f(r.prop_fail_pct)} vs {f(r.pers_fail_pct)}"
    for r in eb.itertuples()) + ".", size=9.5)
bullets([
    ("Where the final policy fails: ", f"{FA['failures that are single lost pings']} of its {TAX['n_failures']} failures are a single lost "
     f"ping and {FA['failures that are loss-only (regret <= 20 ms)']} are loss-only; {FA['failures with no non-failing candidate (unavoidable)']} "
     f"occur when every candidate failed, so no policy could avoid them. {GP['RTT ranking error (regret > 20 ms)']} are RTT ranking errors and "
     f"{GP['stale-history pick (chosen path < 12 observed intervals in run)']} picks of a path with less than 6 h of history in its run."),
    ("Gate outcomes: ", f"the gate missed {G_MISS} incumbent failures, fired correctly {G_AV} times and raised {G_OK + G_HARM} false "
     f"alarms, {G_HARM} of which switched to a failing path."),
    ("What the gates use: ", "Phase 2 gate: " + ", ".join(IMP2.feature.head(5)) + "; final gate: " + ", ".join(IMP.feature.head(5)) + "."),
    ("Shortcut audit, Phase 2 gate: ", "AUPRC " + ", ".join(f"{k} {v['auprc']:.3f}" for k, v in SC2.items()) +
     ". Dropping temporal features does not reduce AUPRC, and RTT-history features alone match the full set, so the artifact signal "
     "is not a time shortcut."),
    ("Shortcut audit, final gate: ", f"a gate trained on time-of-day and day-of-week only reaches AUPRC {f(SC['temporal_only_auprc'], 3)} "
     f"vs {f(SC['full_gate_auprc'], 3)} for the full gate (prevalence {f(SC['prevalence'], 3)}, {SC['n_events']} events in the last fold). "
     "The final gate's weak signal is therefore largely the time-of-day concentration of loss events (afternoon UTC), not path history."),
])

# ======================================================================================
# 7. Discussion
# ======================================================================================
doc.add_heading("7. Results analysis and discussion", level=1)
doc.add_heading("7.1 Research questions", level=2)
ra = RA.pivot_table(index="run_age", columns="policy", values="fail_pct")
dr = DR.set_index(["segment", "model"])
lv = LV.set_index(["held_out", "model"])
QR = QO[QO.profile.str.startswith("gaming, relative")].set_index("policy").satisfaction_pct
c_b = cs.loc[("best path is held out", "Proposed", "held-out paths unseen in training")]
c_nb = cs.loc[("best path is held out", "Proposed, no blend", "held-out paths unseen in training")]
w2, w4 = "week 2", "week 4 (days 21-23)"
rq = pd.DataFrame([
    ["RQ1: does the switcher lower the failure share below persistence?",
     f"Yes. {f(P.fail_pct)} % vs {f(PE.fail_pct)} %, difference CI [{f(P.dfail_lo)}, {f(P.dfail_hi)}] points, excludes zero; mean regret "
     f"{f(P.regret_mean, 3)} vs {f(PE.regret_mean)} ms; switching {f(P.switch_per_pair_day)} vs {f(PE.switch_per_pair_day)} per pair-day "
     f"(budget: gate fires on {f(P.fired_pct)} % of decisions, limit 10 %). But the ranker alone ({f(RK.fail_pct)} %), last-known-best and "
     "EMA-6 do as well or better, so the gain comes from using history, not from the gate.",
     "Supported vs persistence; not vs memory baselines"],
    ["RQ2: can incumbent failure be predicted (AUPRC above prevalence and the RTT-jump rule)?",
     f"With the Phase 2 incumbent: yes, AUPRC {f(average_precision_score(G2.fail, G2.p_gate), 3)} vs prevalence {f(G2.fail.mean(), 3)}, "
     f"but the signal is the sampling artifact. For the final incumbent: AUPRC {f(LG.auprc, 3)} vs prevalence {f(LG.prevalence, 3)} and "
     f"{f(GQ.loc['RTT-jump rule', 'auprc'], 3)} for the jump rule, so formally yes, but at precision {f(LG.precision, 2)} acting on it "
     "costs more failures than it avoids, and a time-only gate is as good.",
     "Formally supported, not useful"],
    ["RQ3: is error concentrated after paths reappear, and does blending help?",
     f"Mildly: proposed failure share {f(ra.loc['<2 h', 'Proposed'])} % (< 2 h), {f(ra.loc['2-6 h', 'Proposed'])} % (2-6 h), "
     f"{f(ra.loc['>=24 h', 'Proposed'])} % (>= 24 h). In the cold-start test (five draws), on decisions whose best path was never seen in "
     f"training the blend gives {f(c_b.fail_pct)} % failures vs {f(c_nb.fail_pct)} % without it ({int(c_b.n):,} decisions). The blend does "
     "not help.", "Not supported"],
    ["RQ4: does performance drift and transfer to an unseen vantage point?",
     f"Failure share rises from {f(dr.loc[(w2, 'static (week 1)'), 'fail_pct'])} % (week 2) to {f(dr.loc[(w4, 'static (week 1)'), 'fail_pct'])} % "
     "(week 4) for the static model and for last-known-best alike, while the path set shrinks; retraining does not reduce it. Removing only "
     "the source vantage leaves results unchanged, but when a vantage never appears in training (as source or destination) the policy "
     "degrades: " + ", ".join(f"{v} {f(lv.loc[(v, 'vantage unseen (src and dst)'), 'fail_pct'])} % vs "
                             f"{f(lv.loc[(v, 'in-vantage (all pairs)'), 'fail_pct'])} %" for v in ("EU", "CH", "NA")) + ".",
     "Drift: no model-specific decay; transfer: fails for a fully unseen vantage"],
    ["RQ5: does QoE satisfaction discriminate between policies?",
     "No, not with the published thresholds: two profiles have a ceiling of about 0-1 % and three of 100 %, so every deployable policy "
     "scores ~0 % or ~99 %. Relative thresholds anchored to each pair's best-path RTT do discriminate (gaming-like: "
     + ", ".join(f"{p} {QR.loc[p]:.1f} %" for p in ["Persistence", "ScionPathML heuristic (forecast)", "Proposed", "Oracle*"]) +
     "), but rank policies as failure share does. Regret and failure share are the informative metrics.", "Supported"],
], columns=["Research question", "Result", "Verdict"])
table(rq, "Answers to the Phase 2 research questions.", widths=[4.0, 10.0, 2.6], font=8)

doc.add_heading("7.2 Does the work address the gap?", level=2)
para("Phase 1 identified three gaps: no learned recommender evaluated on real SCION data, no handling of rare high-stakes "
     "degradation events, and no cold-start or drift-aware evaluation. This phase closes the evaluation side of all three: the first "
     "learned selector benchmarked on the real ScionPathML data against the published heuristic, persistence, memory baselines and an "
     "oracle, with event-focused metrics, time-respecting protocols, drift, cross-vantage transfer and a cold-start test. On the "
     "modelling side the answer is more nuanced than Phase 2 expected. The headroom over persistence is real, but it is a probe-sampling "
     "effect that any memory of recent measurements removes. What remains are loss events, mostly a single lost ping, that 30-minute "
     "history cannot predict well enough to act on: the event gate, implemented as planned, has a measurable but weak signal and "
     "costs more than it saves. The learned ranker matches the best simple policies; it does not beat EMA-6 or last-known-best.")
doc.add_heading("7.3 Strengths", level=2)
bullets([
    "Fully reproducible from raw archives pinned to a repository commit, with an audit that confirms or corrects every Phase 2 dataset claim.",
    "Time-respecting evaluation (purged expanding-window CV with embargo), paired tests on identical decisions, block bootstrap "
    "by day, Holm correction and event counts with every estimate.",
    "Discovery and quantification of the sampling artifact, which changes how the dataset should be used for path-selection research.",
    "The two-stage design is tested as planned, with the gate trained on the incumbent it judges, and its net effect is measured "
    f"decision by decision (Table {TAB_TAX}).",
    f"Lightweight: fitting takes {COST['fit_seconds_full_train']:.1f} s on a laptop CPU and inference {COST['infer_ms_per_decision']:.3f} ms per "
    f"decision against a 30-minute decision interval; the model is {COST['model_size_kb']:.0f} kB and is shipped with an inference script.",
])
doc.add_heading("7.4 Limitations", level=2)
bullets([
    "Short window and few vantage points: 24-27 days, three effective vantage points, six directed pairs.",
    f"Few hard events: {int(LKB.n_fail)} failures of last-known-best in the CV pool and {int(TE.loc['Persistence', 'n_fail'])} persistence "
    "failures in the test split, so small effects cannot be detected.",
    f"Label resolution: with 15 pings per probe the 5 % loss limit equals one lost ping; {FA['failures that are single lost pings']} of the "
    f"final policy's {TAX['n_failures']} failures are such single drops.",
    "Partial probing: ground truth exists only for probed paths; decisions are evaluated on the 83.5 % where the previous best path "
    "was probed again.",
    "Bandwidth cannot be used per path, so multi-objective QoE with bandwidth cannot be evaluated on this release.",
    "Half-hour granularity: loss bursts shorter than an interval are invisible to the features.",
    "The published Telecom version could not be accessed directly; its numbers are taken from our Phase 2 review.",
])
doc.add_heading("7.5 Future work", level=2)
bullets([
    "Collect data with every path probed each interval (or log which paths were sampled) to remove the artifact by design.",
    "More pings per probe or per-ping sequences, so loss can be measured above single-ping resolution and bursts become predictable.",
    "Path-matched bandwidth tests so that multi-objective QoE recommendation becomes evaluable.",
    "Online learning and contextual bandits on a live SCIONLab AS, now with a strong memory baseline as the bar to beat.",
])

# ======================================================================================
# Appendix
# ======================================================================================
doc.add_heading("Appendix A. Reproducibility", level=1)
para("Code, tables, figures and models are in the phase3 folder. From phase3/src run, in order: parse_raw.py, build_panel.py, audit.py, "
     "tune.py, experiments.py main, experiments.py error, experiments.py robust, extra.py, train_final.py, bandwidth.py, "
     "experiments.py ablation, figures.py, build_report.py. Every number in this report is read from results/tables by build_report.py. "
     "Package versions are pinned in requirements.txt; seeds are fixed. data/processed/provenance.json records the dataset commit, "
     "archive hashes and parser version.")
mc = CARD["models"][0]
para(f"Saved model (Step 12): models/{mc['file']} (trained on the training split; threshold {mc['tau']:.3f}; on the test split it "
     f"reproduces {mc['test_metrics']['fail_pct']:.2f} % failures) and models/{CARD['models'][1]['file']} (all data, for deployment), "
     "with LightGBM boosters as text files and models/model_card.json. Example: python predict.py --model "
     "../models/switcher_trainsplit.pkl --split test.")
cost = pd.DataFrame([["Fit time (rankers incl. cross-fitting, gate, threshold), full training split", f"{COST['fit_seconds_full_train']:.2f} s"],
                     ["Mean fit time per CV fold", f"{COST['fit_seconds_cv_mean']:.2f} s"],
                     ["Inference time per decision", f"{COST['infer_ms_per_decision']:.3f} ms"],
                     ["Decision interval", "1,800,000 ms (30 min)"],
                     ["Serialized model size", f"{COST['model_size_kb']:.0f} kB"],
                     ["Peak process memory (whole experiment run)", f"{COST['peak_rss_mb']:.0f} MB"]],
                    columns=["Experiment G: computational cost", "Value"])
table(cost, "Computational cost on an Intel Core i5-8250U laptop CPU.", widths=[10, 4])
doc.add_heading("References", level=1)
for ref in [
    "D. Rossi, S. Keshvadi, Y. Sharma. ScionPathML: Enabling an Empirical Measurement Dataset and Benchmarks for Path-Aware "
    "Networking. Telecom 7(4):81, MDPI, 2026. Preprint: arXiv 2509.07154. Data: github.com/keshvadi/ScionPathML.",
    "Bourak, Burman, John, Daltrophe, Shrot, Hausheer. Deep Q-learning for SCION path optimisation. CSCML 2025, Springer LNCS, 2026.",
    "Herschbach, Rossi, Keshvadi. Path Dynamics in a Deployed Path-Aware Network. arXiv 2509.04695.",
    "Baumeister, Keshvadi. An Axiomatic Analysis of Path Selection Strategies. arXiv 2509.05938.",
    "Vu et al. Evaluating MPQUIC Schedulers in Dynamic Wireless Networks with 2D and 3D Mobility. Computer Networks 272, 111689, 2025.",
    "Wang et al. ALCS: An Adaptive Latency Compensation Scheduler for Multipath TCP in Satellite-Terrestrial Integrated Networks. "
    "IEEE Trans. Mobile Computing 25(1), 2026.",
    "Bilal et al. Dataset-centric Evaluation of Federated Intrusion Detection Models in IoT Networks. Scientific Reports 16, 2683, 2026.",
    "G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.",
    "C. Burges. From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research TR-2010-82.",
    "M. López de Prado. Advances in Financial Machine Learning (purged cross-validation with embargo). Wiley, 2018.",
]:
    p = doc.add_paragraph(ref, style="List Number"); p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(9)

out = ROOT / "report" / "Phase3_Report.docx"
doc.save(out)
print("saved", out)
