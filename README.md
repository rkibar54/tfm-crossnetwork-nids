# Tabular Foundation Models for Few-Shot Cross-Network Intrusion Detection

Code and per-seed results for the paper:

> R. Kibar, M. Balta. *Tabular Foundation Models for Few-Shot Cross-Network Intrusion Detection: Strong Target-Only Learners, No Benefit from Source Data.* (under review)

The experiments extend the leak-free, target-only-controlled protocol of
[TRACE-IDS](https://github.com/rijan-rayamajhi/TRACE-IDS) (Rayamajhi, 2026, MIT licence)
on the four NetFlow v3 benchmarks with two tabular foundation models,
TabPFN-3.5 and TabICLv2.

## Repository layout

| Path | Content |
| --- | --- |
| `experiments/e7_tfm_kaggle.py` | Experiment 1: shared scaling, TabPFN-3.5, classical target-only reproduction, 50,000 test flows |
| `experiments/e8_tfm_kaggle.py` | Experiment 2: raw features for TFMs, TabPFN-3.5 and TabICLv2, context-size ablation, 20,000 test flows |
| `experiments/e9_kaggle.py` | Experiment 3: LightGBM, XGBoost and CatBoost; ROC-AUC, PR-AUC and confusion counts; timings; TTL ablation (same splits as Experiment 2) |
| `results/e7_runs.csv`, `results/e8_runs.csv`, `results/e9_runs.csv` | Per-seed results (one row per pair/target, seed, model, method, k) |
| `results/e7_summary.csv`, `results/e8_summary.csv` | Means over seeds |
| `results/e9_hardware.json` | Hardware and library versions of the Experiment 3 run |
| `analysis/analyze_e7.py`, `analysis/analyze_e8.py`, `analysis/analyze_e9.py` | Reproduce every number and table in the paper from the CSV files |
| `analysis/make_figures.py`, `figures/` | Reproduce Figures 1 and 2 |

## Reproducing the tables

```bash
pip install -r requirements.txt
python analysis/analyze_e7.py results/e7_runs.csv
python analysis/analyze_e8.py results/e8_runs.csv
python analysis/analyze_e9.py   # reads results/e9_runs.csv and results/e8_runs.csv
python analysis/analyze_units.py   # pair-level RQ3 tests and hierarchical bootstrap for RQ1
python analysis/make_figures.py results/e8_runs.csv figures
```

## Re-running the experiments

The experiment scripts were run as Kaggle scripts with two NVIDIA T4 GPUs. They:

1. clone TRACE-IDS and build the processed NetFlow v3 data with its own `data_loader.py`;
2. reproduce its splits, k draws and source samples with the original seeds;
3. write results after every job and resume from earlier outputs attached as Kaggle inputs.

Requirements: the NetFlow v3 CSV files (University of Queensland NIDS datasets page,
https://staff.itee.uq.edu.au/marius/NIDS_datasets/) attached as input, internet access, and a
`TABPFN_TOKEN` Kaggle secret after accepting the TabPFN-3.5 licence at https://ux.priorlabs.ai.
Set `MAX_HOURS` in each script to stay within the session limit. Approximate runtimes:
Experiment 1 about 16 GPU-hours, Experiment 2 about 10 GPU-hours on one T4, Experiment 3 about 7 hours on two T4s.

Versions used: `tabpfn` 9.2.0 in Experiments 1–2 and 10.2.0 in Experiment 3 (same checkpoint,
identical target-only results; `tabpfn-v3.5-20260909`), lightgbm 4.6.0, xgboost 3.4.1, catboost 1.2.10, TabICL checkpoint
`tabicl-classifier-v2-20260212`, Python 3.13, scikit-learn as provided by the Kaggle image.

## Licence

Our code is released under the MIT licence (`LICENSE`). The classical model definitions and
the data pipeline come from TRACE-IDS; its licence is reproduced in
`THIRD_PARTY_LICENSE_TRACE-IDS.txt`. TabPFN model weights are subject to the Prior Labs licence
and are not redistributed here.

## Citation

If you use this code, please cite our paper and Rayamajhi (2026),
doi:10.20944/preprints202610.1041.v1.
