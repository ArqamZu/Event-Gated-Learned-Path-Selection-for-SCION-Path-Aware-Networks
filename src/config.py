"""Shared constants for the Phase 3 pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
MODELS = ROOT / "models"
for d in (PROC, FIG, TAB, MODELS):
    d.mkdir(parents=True, exist_ok=True)

VANTAGE = {"19-ffaa:1:11de": "EU", "17-ffaa:1:11e4": "CH", "18-ffaa:1:11e5": "NA"}

# ---- Dataset provenance (Phase 2, Step 2) -------------------------------------------------
# The three raw archives were verified on 2026-11-15 against the public repository: their git
# blob SHA-1s are identical to the files at the repository HEAD below, and the archives have not
# changed since the commit that added them. parse_raw.py re-checks the local files against these.
DATASET_REPO = "https://github.com/keshvadi/ScionPathML"
DATASET_COMMIT = "362ad60885dabeb2a067d3dae058c506ce5f092a"      # HEAD at verification (2025-09-07)
DATASET_ZIP_COMMIT = "f0d43d945a6b32b30b55ba765afd4db14205ff0c"  # last commit touching the zips (2025-08-15)
DATASET_BLOBS = {"Lab1": "f4bb5b90b25b958efdbddc4b5da7c003ff893fe0",
                 "Lab2": "07bb29e23c888a1ad52ec7d693dd276e35f38b62",
                 "Lab3": "c86cee63a658466eefc60f2aa78c43d2f42b959c"}
PARSER_NAME = "phase3 parse_raw.py (own parser of the raw JSON; the scionpathml transformer is not used)"
PARSER_VERSION = "1.1.0"   # 1.1.0: hop count from the ping path, 100 %-loss probes kept as dead-path events

# ---- Failure definition (Phase 2, Step 3) ---------------------------------------------------
# Incumbent worse than the best path in hindsight by more than FAIL_MS, or loss above FAIL_LOSS
# percent. Each probe sends 15 pings, so loss is quantized in steps of 6.67 %: FAIL_LOSS = 5 means
# "at least one lost ping", 10 means ">= 2 lost pings", 20 means ">= 4 lost pings".
FAIL_MS = 20.0
FAIL_MS_STRICT = 50.0
FAIL_LOSS = 5.0
PINGS_PER_PROBE = 15
# sensitivity grid (Phase 2, Step 10): (regret margin ms, loss %)
LABEL_GRID = [(20, 5), (50, 5), (20, 10), (50, 10), (20, 20), (50, 20)]
# a probe with 100 % loss (path dead) is a failure with this regret (the SCION ping timeout, 1 s)
DEAD_REGRET_MS = 1000.0

WINDOWS = (1, 3, 6, 12)    # history windows in 30-minute intervals (up to 6 h)
EMBARGO = 12               # purge gap (intervals) in expanding-window CV
SPLIT = (0.70, 0.15, 0.15) # chronological train / validation / test, per directed pair
SEEDS = (0, 1, 2, 3, 4)
SWITCH_BUDGET = 0.10       # max share of decisions on which the gate may fire (tuned on validation)
# stale-history blend (Phase 2, Step 5): weight of the full-history score w = n / (n + BLEND_K),
# n = observed intervals in the path's current presence run (w = 0.5 after BLEND_K observations)
BLEND_K = 6

# ---- QoE profiles: (max RTT ms, max loss %) -------------------------------------------------
# Bandwidth is excluded (no ping/bandwidth path overlap, see audit).
QOE_PHASE2 = {  # published ScionPathML thresholds used in Phase 2 (primary)
    "video_conf": (250, 7), "gaming": (120, 4), "file_transfer": (800, 12),
    "browsing": (500, 10), "streaming": (400, 5)}
QOE_PREPRINT = {  # arXiv 2509.07154v1, Table 4 (secondary comparison)
    "video_conf": (150, 2), "gaming": (50, 1), "file_transfer": (500, 5),
    "browsing": (300, 3), "streaming": (200, 1)}
