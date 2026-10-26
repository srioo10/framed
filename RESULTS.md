# Results index

Each pre-registered claim, its verdict, and the file that holds it. The criteria and the dated decision log are in [`PREREG.md`](PREREG.md). "Confirmatory" means the criterion was fixed before the run; "exploratory" means it was not, or the data had already been seen.

## 1. ARLHNIDS-IoT re-evaluation (H0, measurement)

| Dataset | Published SIDS accuracy | Paper order (ours): accuracy / macro-F1 | Leak-free: accuracy / macro-F1 | Files |
|---|---|---|---|---|
| UNSW-NB15 | 92.89% | 0.69–0.83 / 0.68–0.84 | 0.77 / 0.41–0.50 | `results/arlhnids/sids_unsw_official_*` |
| BoT-IoT | 99.19% | 1.00 / 1.00 | 0.93–0.94 / 0.55–0.65 | `results/arlhnids/sids_bot_iot_*` |
| CICIoT2023 | 98.31% | 0.72–0.88 / 0.73–0.89 | 0.85–0.86 / 0.59–0.67 | `results/arlhnids/sids_ciciot_paper_*` |

- Paper order puts 14–53% of test rows in training and makes 35–46% of the test set synthetic. On BoT-IoT, grey-wolf search keeps the row identifier `pkSeqID`, and the feature set includes the label columns.
- 5 seeds × GAMO / SMOTE × paper / leak-free.

## 2. Exploratory directions that did not go forward

| Direction | Outcome | Files |
|---|---|---|
| Tree-teacher → int8 MLP distillation (the original plan) | The KD student lost to plain RF/XGB on unseen-family AUROC, and KD was worse than CE | `results/pilot/` |
| Budgeted fusion of tree, MLP and density scores | Macro unseen-family TPR@1%FPR ceiling about 0.47–0.49 for every method on CICIoT2023 and UNSW raw | `results/fusion/` |
| Class-reference / quantile novelty gates (CRN) | No-go: gates failed | `results/crn/` |
| Per-host conformal CUSUM kill test (NF-CICIDS2018, seed 0) | Failed gate G2 (Bot gain 0); host context then became part of the model instead | `results/host_stream/` |
| NF-UNSW-NB15 host arm | Saturated (1.000 for every method), generator fingerprint | `results/host/` |

## 3. OnePass-IDS confirmatory claims

**Host level, NF-CICIDS2018, seeds 1–4**: `results/onepass/nf_cic/claims.md` (host-day), `results/onepass/nf_cic_hour/claims.md` (host-hour)

| Claim | Host-day | Host-hour | Verdict |
|---|---|---|---|
| C1 OnePass − per-flow MLP ≥ 0.10, CI > 0 | +0.334 [0.256, 0.405] | +0.283 [0.205, 0.352] | PASS (one dataset; two were required) |
| C2 within 0.02 of the best large per-flow model | +0.206 | +0.051 | PASS |
| C3 IP shuffling removes the host gain | +0.102 vs −0.162 | +0.048 vs −0.208 | PASS |
| C4 realised benign alarm rate ≤ 1.5% | 1.12% | 1.18% | PASS |
| C5 int8 loss ≤ 0.02 | 0.001 | −0.001 | PASS |
| C6 DDU − OnePass ≥ 0.03, CI > 0 | +0.143 [0.110, 0.169] | +0.200 [0.164, 0.226] | PASS → DDU-OnePass is the headline model |
| C7 taxonomy back-off ≥ 0.6 and ≥ 2× chance | 0.508 | 0.508 | FAIL |

**Dataset entry** (≥ 200 benign calibration units):

| Dataset | Calibration units | Status |
|---|---|---|
| NF-CICIDS2018 | 11.5k+ host-days | enters |
| NF-ToN-IoT | 151 host-days, 185 host-hours | does not enter; flow level only |
| NF-BoT-IoT | — | realised flow FPR 15–16% at a 1% threshold, so not calibratable |
| UNSW raw | 4 attacker hosts, 99 benign host-days | flow level only |
| BoT-IoT | 477 benign rows | reproduction only |

**Flow level** (macro TPR@1%FPR, unseen family): `results/onepass/flow_*.md`, `figures/fig3_values.csv`

| Dataset | Tiny MLP | RF | XGB | MCDE |
|---|---|---|---|---|
| NF-CICIDS2018 (host context) | 0.536 | 0.607 | 0.585 | 0.654 |
| NF-ToN-IoT (host context) | 0.147 | 0.031 | 0.061 | 0.104 |
| CICIoT2023 | 0.592 | 0.462 | 0.546 | 0.596 |
| UNSW raw (causal ct_*) | 0.725 | 0.752 | 0.727 | 0.728 |
| UNSW official | 0.473 | 0.509 | 0.516 | 0.607 |

## 4. Optimisation steps (pre-registered 2026-10-25)

| Claim | Result | Verdict |
|---|---|---|
| Host-hour unit | NF-CICIDS2018: C1–C6 hold. NF-ToN-IoT: does not enter. | reported |
| T1 nested leave-one-family-out tuning (8 settings) | host +0.137 (≈ DDU); flow −0.050 (NF-CICIDS2018), −0.114 (NF-ToN-IoT) | FAIL |
| B1 class-weighted / SMOTE | unseen −0.02 to −0.04; known-family macro-F1 +0.01 | FAIL |

## 5. Taxonomy back-off on a fresh dataset (C7b, CICIoT2023, 5 seeds)

`results/onepass/backoff_ciciot.md`

| | Macro over leaf families (chance 0.25) | Verdict |
|---|---|---|
| DDU-likelihood back-off (the claim) | 0.376 | FAIL |
| Softmax back-off (control, the C7 rule) | 0.617 | passes the same bar |

Volumetric families are named correctly (DoS 1.00, Mirai 1.00, DDoS 0.89); access-type families are not (BruteForce 0.10, Web 0.03).

## 6. Robustness and cost

| Check | Result | File |
|---|---|---|
| Benign alarm spread (NF-CICIDS2018) | about 127 of 14.6k test hosts alarm per fold; the top 10% give 27–34% of alarm-days; max 10 days on one host | `results/onepass/robust_nf_cic_*.md` |
| Within-host dependence | lag-1 autocorrelation of benign conformal p-values 0.53–0.65; the Ville rule realises 6–7% per-flow alarms vs a 1% target | same |
| Deployment cost | int8 MLP 10.9 KB / 78 µs; DDU 233 KB / 214 µs; XGB 14 MB / 619 µs; RF 287 MB / 13 ms | `results/onepass/cost.md` |
