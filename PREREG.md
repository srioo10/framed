# Pre-registration — Project 1

**Frozen:** 2026-10-23, before any model was trained. Git tag `prereg-v1`.
Any later change goes in §9 (Deviations) with date and reason; nothing above §9 is edited after the tag.

**Study:** Leak-free re-evaluation of ARLHNIDS-IoT (Srivastava & Sinha, Computers & Security 156, 2025, 104515) and a tree-distilled int8 MLP that does known-class classification and unknown-family detection from one forward pass.

---

## 1. Research questions and hypotheses

| ID | Hypothesis | Decision rule |
|---|---|---|
| H0 | The paper's reported metrics are inflated by its evaluation protocol. | Measurement, no pass/fail. Report Δ (paper-order − leak-free) per metric, per dataset, with 95% bootstrap CI, and attribute Δ to each protocol step (§4). |
| H1 | The distilled student is close to its teacher on known classes. | Primary KD student (S6) macro-F1 ≥ teacher macro-F1 − 2.0 pts, mean over families × seeds. |
| H2 | int8 post-training quantisation does not change unknown-family detection. | \|AUROC(int8) − AUROC(fp32)\| ≤ 1 std of AUROC across seeds, for energy and Mahalanobis, on every dataset. |
| H3 | KD transfers useful uncertainty: the KD student's unknown score beats no-teacher students and the teacher's own score. | Primary metric M1. S6 vs S1, S6 vs S3, S6 vs teacher vote entropy; paired one-sided Wilcoxon over (family, seed), Holm-corrected α = 0.05, for each of energy and Mahalanobis. |
| H4 | The paper's DDQN (re-implemented leak-free) is not better than a plain classifier + score. | Exploratory, two-sided, no prior direction. |

## 2. Data

| Key | Source | Role |
|---|---|---|
| `unsw_official` | UNSW-NB15 official train/test (175,341 / 82,332) | Paper reproduction |
| `unsw_raw` | UNSW-NB15_1..4 (≈2.54M) with timestamps | Leak-free + LOFO |
| `bot_iot` | BoT-IoT 5% "10-best" train/test (2,934,817 / 733,705) | Paper reproduction + leak-free + LOFO |
| `ciciot_paper` | 8 capture files from CIC 2024 CSV release, exactly as ARLHNIDS Table 4 (401,260 rows) | Paper reproduction |
| `ciciot_csv` | All per-pcap CSVs in the same release (34 classes) | Leak-free + LOFO (primary CICIoT set) |
| `ciciot_merged` | Original CICIoT2023 MERGED_CSV (46 features) | Duplicate-rate audit; secondary leak-free |

Loader row counts are asserted against the paper's tables (`tests/test_load.py`).

**Observation recorded before training:** ARLHNIDS's CICIoT2023 subset uses exactly one capture file per class. Class identity is therefore confounded with capture session; no grouped split is possible on that subset.

## 3. Cleaning (identical for every leak-free experiment)

1. Drop rows with NaN or ±inf in any feature. Count them.
2. Drop identifier / known-leak columns:
   - UNSW: `id`, `srcip`, `sport`, `dstip`, `dsport`, `stime`, `ltime`, `sttl`, `dttl`, `ct_state_ttl`, `label`.
   - BoT-IoT: `pkSeqID`, `saddr`, `sport`, `daddr`, `dport`, `seq`, `attack`, `subcategory`.
   - CICIoT: `Number`; `Weight` also dropped if it is a deterministic function of `Number` (checked on training data only, result logged).
   The paper-order reproduction keeps whatever the paper keeps.
3. Hash each row over the retained features. Drop every row whose hash carries more than one label. Then drop exact duplicates (keep first). Report counts per dataset and per class.

## 4. Splits

Seeds {0, 1, 2, 3, 4} for every stochastic step.

| Dataset | Leak-free split (train / val / test = 60 / 20 / 20) |
|---|---|
| `ciciot_csv` | `StratifiedGroupKFold`, group = capture file |
| `unsw_raw` | `StratifiedGroupKFold`, group = 1-hour window of `stime` |
| `bot_iot` | `StratifiedGroupKFold`, group = block of 10,000 consecutive `pkSeqID` (time proxy) |
| `unsw_official`, `ciciot_paper`, `ciciot_merged` | split first, stratified random (no group id available — stated as a limitation) |

All fitted transforms (imputer, clamp quantiles, scaler, balancer, feature selector) see training rows only. Synthetic rows never enter val or test. Hyperparameters are chosen on val only.

**H0 attribution:** starting from the paper's order, switch one step at a time to leak-free and record Δ: (a) merge-then-split → split-first, (b) oversample-then-split → split-then-oversample-train, (c) Optuna on test → Optuna on val, (d) dedup off → on, (e) DoS-held-out with DDoS in train → Flood held out.

## 5. Models

**Teacher:** majority VotingClassifier of DT, RF, ExtraTrees, AdaBoost, LightGBM, CatBoost, XGBoost (paper's seven). Optuna TPE, 50 trials per model, objective = val macro-F1, sampler seed = run seed. Soft labels = mean predicted probability, produced **5-fold out-of-fold** on the training set.

**Students:** MLP, input → 64 → 32 → classes, ReLU, Adam lr 1e-3, batch 512, max 50 epochs, early stopping on val loss (patience 5).

| Arm | Training signal |
|---|---|
| S1 | cross-entropy (CE) |
| S2 | CE, then post-hoc temperature scaling on val |
| S3 | CE with label smoothing 0.1 |
| S4 | KD, T = 1, α = 1 |
| S5 | KD, T = 4, α = 1 |
| **S6 (primary)** | KD, T = 4, α = 0.5 + CE on hard labels |

int8: PyTorch → TFLite full-integer post-training quantisation, calibration = 1,000 random training rows (seeded).

## 6. Unknown-family protocol (LOFO)

Families (DoS + DDoS merged):
- `ciciot_csv`: Flood, Recon, Spoofing, Web, BruteForce, Mirai
- `unsw_raw`: DoS, Exploits, Fuzzers, Generic, Recon, Analysis+Backdoor, Shellcode+Worms
- `bot_iot`: Flood, Recon, Theft (Theft n = 79 → reported, flagged low-n, excluded from H-tests)

Per round: every row of the held-out family is removed from train and val and appears only in test. Benign is present in every round.

**Scores:** energy (T = 1), Mahalanobis (penultimate layer, class means + tied covariance fitted on train each round), kNN (k = 10, penultimate, cosine), MSP.
**Baselines:** teacher vote entropy (primary tree comparator), teacher top-2 probability margin, calibrated RF max-prob, IsolationForest on benign train, leak-free DDQN (paper's AIDS).

## 7. Metrics

| ID | Metric |
|---|---|
| **M1 (primary)** | AUROC of the unknown score: held-out family (positive) vs benign test (negative) |
| M2 | Held-out-family recall under the combined rule (predicted class ≠ benign OR score > τ), τ = 99th percentile of benign-val scores |
| M3 | FPR at 95% TPR |
| M4 | Alarm precision at attack base rates 1% and 0.1% (Bayes from M2's TPR and benign FPR) |
| M5 | OOD AUROC: held-out family vs all in-distribution test rows |
| K1 | Known-class macro-F1 on test (held-out family excluded) |

Reporting: mean ± 95% cluster-bootstrap CI (10,000 resamples over families). Every reported F1 is recomputed from P and R as a consistency check.

## 8. Exclusions and stopping

- A run is excluded only for a crash or a failing leak test (§ tests). Excluded runs are listed, not dropped silently.
- No extra seeds, families or arms are added after seeing test results for H1–H3; additions are labelled exploratory.

## 9. Deviations

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-10-23 | §4 | For grouped splits, the seed selects the fold (test = fold `seed % 5` of a fixed `StratifiedGroupKFold(5)`, val = fold `seed % 4` of the inner `StratifiedGroupKFold(4)`) instead of seeding the shuffle. Seeds 0–4 therefore form 5-fold cross-validation over groups. | sklearn's `StratifiedGroupKFold(shuffle=True)` only shuffles ties, so different seeds gave identical splits (caught by `tests/test_split.py`). Decided before any model was trained. |
| 2026-10-23 | §4 | `bot_iot` uses the stratified random split (after dedup) instead of `pkSeqID` blocks. | `pkSeqID` in the 5% / 10-best files is ordered by class (DoS 1–1,650,260, DDoS, Normal, Reconnaissance, Theft), not by time, so the blocks grouped by label: all Normal rows fell in one block and none reached test. The files carry no timestamp or capture id. Found by `scripts/audit_clean.py` before any model was trained. |
| 2026-10-23 | §2, §3 | `unsw_raw`: blank (empty or whitespace) `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd` are read as 0, not dropped as missing. | Blank means "not applicable" (non-HTTP / non-FTP flow); UNSW's own train/test sets store these as 0. Treating them as missing dropped 57% of rows. Found by the audit before any model was trained. |
| 2026-10-23 | §4 | `unsw_raw` groups are 10-minute `stime` windows, not 1-hour. | Hourly windows gave 27 groups (16 with attacks); SGKF put every attack in train and none in test. 10-minute windows give 149 groups, each attack family in ≥ 82. Found by the audit before any model was trained. |
| 2026-10-23 | §4 | `ciciot_csv` groups are contiguous tenths of each capture file, not whole files. | BruteForce has 1 capture file, Spoofing 3, Benign 4, Recon 5, so whole-file groups put those families entirely in train. Contiguous blocks keep temporally adjacent flows together while giving every family ≥ 10 groups. Found by the audit before any model was trained. |
