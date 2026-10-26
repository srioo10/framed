# OnePass-IDS

**A one-pass detector of a few kilobytes for known and unseen attack families on IoT networks, evaluated without leakage.**

An IDS on an IoT gateway sees each flow once, has kilobytes of memory, and will meet attack families that were not in training. OnePass-IDS is a small MLP (about 6k weights, 10.9 KB as int8) that does two jobs in one forward pass. It names known attack families and gives an unknown-attack score. Two optional additions sit on top: 20 causal per-host context features, and a per-host conformal CUSUM (one float of state per source host).

The study has two parts:

1. **A leak-free re-evaluation of ARLHNIDS-IoT** (Srivastava & Sinha, *Computers & Security* 156, 2025, 104515). We re-run its pipeline in the published order and in a leak-free order, and attribute the change to each protocol step.
2. **OnePass-IDS under a leak-free, host-aware open-set protocol**:
   - 10-minute time-block grouped splits;
   - one attack family held out at a time;
   - a 1% benign false-alarm budget calibrated on held-out benign traffic and reported as measured;
   - host-bootstrap confidence intervals.

Every hypothesis, claim, deviation and result is logged, dated, in [`PREREG.md`](PREREG.md). Criteria were fixed before the confirmatory runs and never changed after a result.

## Main results

| Finding | Result |
|---|---|
| ARLHNIDS protocol effect | macro-F1 0.84–1.00 in the published order → 0.41–0.67 leak-free. The published order puts 14–53% of test rows in training and makes 35–46% of the test set synthetic. |
| Known families | MLP accuracy 0.93–0.97, about equal to RF and XGBoost |
| **Unseen families, host level (NF-CICIDS2018)** | **DDU-OnePass 0.682** attacker host-days detected vs RF 0.684 and XGBoost 0.672 (same host evidence), at about 1/1000 of their size. Measured benign host-day alarm rate 1.1%. Holds in host-hour units too. |
| Unseen families, flow level | MLP beats the tree ensembles on NF-ToN-IoT (0.15 vs ≤ 0.06) and CICIoT2023 (0.59 vs 0.46–0.55); trees and MCDE are better on UNSW-NB15 |
| Deployment | int8 MLP 10.9 KB / 78 µs per flow vs RF 287 MB / 13 ms (one CPU thread) |
| Negative results | Nested leave-one-family-out tuning does not transfer to real unseen families. Class balancing (weights, SMOTE) helps known classes only. Taxonomy back-off names unseen floods but not access-type attacks. |

The per-claim index is in [`RESULTS.md`](RESULTS.md).

## Data

The datasets are not redistributed. Download them and place the files in the repository root (or set `NIDS_DATA_DIR`):

| Dataset | Source | Files expected |
|---|---|---|
| UNSW-NB15 (raw + official split) | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `OneDrive_1_10-23-2026.zip` (`UNSW-NB15_1..4.csv`, `Training and Testing Sets/`, `NUSW-NB15_features.csv`) |
| BoT-IoT (5%, 10-best) | https://research.unsw.edu.au/projects/bot-iot-dataset | `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv`, `..._Testing.csv` |
| CICIoT2023 (per-capture CSV release) | https://www.unb.ca/cic/datasets/iotdataset-2023.html | `CSV.zip` |
| CICIoT2023 (original MERGED_CSV) | same | `archive(1).zip` |
| NF-CICIDS2018-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, NF-UNSW-NB15-v3 | UQ NIDS datasets, https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | unpacked CSVs in `data/raw/netflow_v3/` (or set `NIDS_NF_DIR`) |

File names are defined at the top of [`src/nids/load.py`](src/nids/load.py). Loaders cache parquet copies in `data/processed/`.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/pytest
```

No GPU is needed. The NetFlow score banks peak at about 18–20 GB of RAM; run them one at a time.

## Reproducing the paper

| Step | Command | Output |
|---|---|---|
| ARLHNIDS reproduction (paper vs leak-free order, GAMO vs SMOTE) | `scripts/arlhnids_sids.py`, `scripts/arlhnids_attrib.py` | `results/arlhnids/` |
| Score banks (all baselines + MLP + int8 + DDU, per held-out family) | `scripts/score_bank.py <dataset> --seed S --variant V --kind outer`; queue: `runs/onepass.sh` | `runs/bank/` (not in git) |
| Host metadata and entry audit | `scripts/host_audit.py <dataset>` | `results/host_stream/audit_*.json` |
| Host-level claims C1–C7 (host-day or host-hour) | `scripts/onepass_claims.py nf_cic --seeds 1 2 3 4 [--unit hour]` | `results/onepass/nf_cic*/claims.md` |
| Flow-level tables | `scripts/host_report.py --ds <dataset> --variants ...` | `results/onepass/flow_*.md` |
| Nested tuning (T1) and balancing (B1) | `scripts/tune_mlp.py <dataset> --seed S`; queue: `runs/optimise.sh` | `tuned.parquet` beside each bank part |
| Taxonomy back-off on CICIoT2023 (C7b) | `scripts/backoff_ciciot.py --seeds 0 1 2 3 4` | `results/onepass/backoff_ciciot.md` |
| Host-budget robustness | `scripts/host_robust.py --member mlp_ce` | `results/onepass/robust_nf_cic_*.md` |
| Deployment cost | `scripts/onepass_cost.py` | `results/onepass/cost.md` |
| Figures | `scripts/paper_figs.py --out figures` | `figures/` |

## Layout

```
PREREG.md      pre-registration: hypotheses, claims, deviations and results, dated
RESULTS.md     claim-by-claim index of results and where each lives
src/nids/      loaders, cleaning, splits, models (MLP, DDU, int8), host context, MCDE, ARLHNIDS components
scripts/       experiments, reports and figures (see the table above)
runs/          run queues and the temperature / memory monitor
results/       every result table cited in the paper
figures/       paper figures (regenerated by scripts/paper_figs.py)
tests/         loader, cleaning, split, leakage and host-stream checks
```

## Reference

Srivastava, A., & Sinha, D. (2025). ARLHNIDS-IoT: An accurate and robust lightweight hybrid-NIDS for IoT network security. *Computers & Security*, 156, 104515.
