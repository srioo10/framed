# OnePass-IDS

**One int8 forward pass for known-attack classification and unknown-attack detection on IoT networks.**

A tree ensemble is a strong intrusion detector, but it is large and gives no signal that a flow belongs to an attack family it has never seen. OnePass-IDS distils a seven-tree teacher into a tiny int8 MLP whose outputs serve two jobs at once: the class prediction handles known attacks, and an energy / Mahalanobis score on the same pass flags unseen families.

The study is also a leak-free re-evaluation of ARLHNIDS-IoT (Srivastava & Sinha, *Computers & Security* 156, 2025, 104515). We re-run its pipeline in the published order and in a leak-free order, and attribute the change in reported metrics to each protocol step.

**Status:** pre-registered, experiments in progress. Hypotheses, splits, metrics and statistical tests were fixed before any model was trained. See [`PREREG.md`](PREREG.md) (git tag `prereg-v1`).

## Data

The datasets are not redistributed. Download them and place the files in the repository root:

| Dataset | Source | Files expected |
|---|---|---|
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `OneDrive_1_10-23-2026.zip` (contains `UNSW-NB15_1..4.csv`, `Training and Testing Sets/`, `NUSW-NB15_features.csv`) |
| BoT-IoT (5%, 10-best) | https://research.unsw.edu.au/projects/bot-iot-dataset | `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv`, `..._Testing.csv` |
| CICIoT2023 (per-capture CSV release) | https://www.unb.ca/cic/datasets/iotdataset-2023.html | `CSV.zip` |
| CICIoT2023 (original MERGED_CSV) | same | `archive(1).zip` |

Paths are defined at the top of [`src/nids/load.py`](src/nids/load.py). Loaders cache parquet copies in `data/processed/`.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/pytest
```

No GPU is required.

## Verification

`tests/test_load.py` asserts that every loader reproduces the row counts in ARLHNIDS-IoT Tables 2–4 exactly. In doing so we found that the paper's CICIoT2023 subset is exactly one capture file per class (listed in `load.CIC_PAPER_FILES`), so class identity is confounded with capture session.

## Layout

```
PREREG.md          frozen hypotheses, protocol and metrics
src/nids/load.py   dataset loaders (labels, attack families, provenance for grouped splits)
tests/             data and leakage checks
```

## Reference

Srivastava, A., & Sinha, D. (2025). ARLHNIDS-IoT: An accurate and robust lightweight hybrid-NIDS for IoT network security. *Computers & Security*, 156, 104515.
