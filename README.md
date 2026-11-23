# Phase 3: Event-gated SCION path switching on ScionPathML data

Implementation and experimental evaluation for the Computer Networks research project
(Arqam Zulfiqar 23I-2608, Ali Haroon 23I-2592). Base dataset: ScionPathML
(github.com/keshvadi/ScionPathML), raw per-lab archives `AnalysisResults/LabN/Data/raw_data.zip`.

## Reproduce everything

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14 used
mkdir -p data/raw  # then copy <ScionPathML>/AnalysisResults/LabN/Data/raw_data.zip to data/raw/LabN.zip for N = 1, 2, 3
cd src
python parse_raw.py          # raw JSON -> pings / pathsets / bandwidth tables + provenance.json (~1 min)
python build_panel.py        # 30-min grid, decisions, causal features            (~20 s)
python audit.py              # re-derives every Phase 2 dataset number           (~10 s)
python tune.py               # purged-CV hyperparameter search on training split (~5 min)
python experiments.py main   # experiments A, B, G + statistics                  (~6 min)
python experiments.py error  # error analysis, SHAP, failure taxonomy            (~1 min)
python experiments.py robust # experiments C, D, E, F                            (~5 min)
python extra.py              # Phase 2 design: sweep, SHAP, shortcut audit, forecast refs (~5 min)
python train_final.py        # trains and saves the final model -> models/      (~1 min)
python bandwidth.py          # auxiliary bandwidth analysis                      (~30 s)
python experiments.py ablation  # 23 ablations x 5 seeds + budget / label sensitivity (~50 min)
python experiments.py labels    # (optional) only the failure-label sensitivity     (~4 min)
python figures.py            # all figures -> results/figures
python build_report.py       # report -> report/Phase3_Report.docx
python make_pdf.py ../report/Phase3_Report.docx "$PWD/../report/Phase3_Report.pdf"   # PDF via headless Chrome
```

Use the saved model:

```bash
python predict.py --model ../models/switcher_trainsplit.pkl --split test       # reproduces the test-split numbers
python predict.py --model ../models/switcher_alldata.pkl --pair EU->NA --last 5 # latest decisions for one pair
```

Dataset provenance: `data/processed/provenance.json` records the ScionPathML commit
(`362ad60`; archives unchanged since `f0d43d9`), the SHA-256 and git blob hash of every archive
(checked against the commit on each parse), the parser version and the table row counts.

All randomness is seeded (seeds 0-4). Hardware used: Intel Core i5-8250U (4 cores / 8 threads),
8 GB RAM, Linux 7.0, CPU only.

## Layout

| Path | Content |
|---|---|
| `src/config.py` | constants: dataset commit, failure thresholds and sensitivity grid, windows, splits, seeds, QoE profiles |
| `src/parse_raw.py` | Step 2: raw JSON parser (prober, path discovery, bandwidth) |
| `src/build_panel.py` | Steps 3-4: grid alignment, masking, decision construction, features |
| `src/audit.py` | data audit vs. Phase 2 claims |
| `src/core.py` | Steps 5-9: gate, ranker, switcher, baselines, metrics, statistics |
| `src/tune.py` | hyperparameter search (purged expanding-window CV inside training data) |
| `src/experiments.py` | Experiments A-G, ablations, error analysis |
| `src/extra.py` | Phase 2 design: threshold sweep, SHAP, shortcut audit; RTT forecast references |
| `src/train_final.py` | Step 12: trains and saves the final model (`models/*.pkl`, booster text files, `model_card.json`) |
| `src/predict.py` | inference with a saved model (chosen path, incumbent, gate probability per decision) |
| `src/bandwidth.py` | auxiliary bandwidth analysis (test outcomes, path effect, time of day, link to ping quality) |
| `models/` | saved models and model card |
| `src/figures.py` | figures |
| `src/build_report.py` | report generator (numbers are read from `results/tables`) |
| `src/make_pdf.py` | DOCX -> PDF conversion (mammoth + headless Google Chrome) |
| `results/tables/` | every number in the report (CSV / JSON) |
| `results/figures/` | every figure in the report (PNG) |
