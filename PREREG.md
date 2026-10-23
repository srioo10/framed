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
| 2026-10-23 | §4 (H0 attribution) | Added step (f): BoT-IoT label columns `attack` and `subcategory` kept as features (paper order) → dropped. | The paper's Table 11 gives BoT-IoT 18 feature dimensions, which equals the 10-best file's 19 columns minus the target `category`, so the paper's features include the binary attack label and the attack subcategory. Found while reading the method before any ARLHNIDS run. |
| 2026-10-23 | §4, §5 (reproduction details the paper does not report) | FCM uses the paper's Table 8 settings instead of re-running Optuna (its objective is not reported); centroids are fit on a 50k-row sample per class; the reduced class keeps Table 9's size, drawn from every cluster in proportion to its size ("Grouping" is not described). GWO uses Table 11 (wolves, iterations, seed 40, α = 0.5, correlation < 0.6) with fitness from a 50-tree XGBoost on a 20k-row stratified training sample (75/25 internal holdout), run for Table 11's fixed iteration counts. SIDS uses the paper's Table 13 settings; ET `max_features="auto"` = `"sqrt"`, AdaBoost runs SAMME (SAMME.R was removed from scikit-learn), and the hard vote is computed by hand so XGBoost can early-stop on its eval set (test in paper order, val in leak-free order). Until the modified GAMO is built, SMOTE oversamples minority classes; class targets keep Table 10's per-class ratios, so paper order reproduces Table 10 exactly. | Needed to run the paper's pipeline at all; each choice is the closest reading of the paper and is applied identically in paper order and leak-free order, so it cannot create the paper-vs-leak-free gap. Decided before any ARLHNIDS run. |
| 2026-10-23 | §4, §5 (modified GAMO details the paper does not report) | GAMO follows Table 6 and the §3.4 text: noise length 32, conditional feature mapping unit, one convex SGU per oversampled class whose softmax weights (text §3.4.3; Table 6 puts the softmax one layer earlier) mix that class's real training rows, conditional discriminator, classifier M. Each "Dense n softmax" + "Attention n" pair of Table 6 is read as one soft feature gate, x[:n]·softmax(Wx+b), because the paper does not define attention; the literal reading (a softmax layer on the input) collapsed every class to a single synthetic row on UNSW. Only the classes that are not undersampled enter GAMO; 75% of each is the training set and convex basis (matching Table 7, e.g. UNSW Worms 130 of 174), and 25% scores M. Training budget: 12,000 Adam steps (3,000 left M at 43% holdout accuracy on UNSW), batch 256, lr 2e-4, betas (0.5, 0.999); fake classes are drawn in proportion to the rows still needed (the (P_c − P_i) weights of Eqs. 9 and 12); the generator's classifier term is −log(1 − M_i), the bounded form of Eq. 9. Per-class targets are unchanged (Table 10). Partial mode collapse remains for some classes (e.g. UNSW Worms); each run records per-class diversity (spread vs real, nearest-real distance, unique share) instead of adding a diversity term the paper does not have. SMOTE stays as the control arm. GWO fitness evaluations now run in parallel, one single-threaded XGBoost per wolf; the search is unchanged. | The paper gives the architecture but no attention definition, epochs, batch size or learning rate. The settings are fixed before any GAMO result and applied identically in both orders. |
| 2026-10-23 | §4 (H0 attribution) | Added step (g): BoT-IoT identifier columns (`pkSeqID`, `saddr`, `sport`, `daddr`, `dport`, `seq`) kept as features (paper order) → dropped. Seed-0 runs also report, in paper order, accuracy on real vs synthetic test rows and the share of test rows that exactly match a training row. | Added after the seed-0 paper-order BoT-IoT run, where GWO kept `pkSeqID` alone and every model but AdaBoost scored ≥ 99.5% on it (`pkSeqID` follows class order; see the `bot_iot` split row above). Labelled post hoc: it is an attribution step, not a change to H1–H3. |
| 2026-10-24 | §1, §5, §6 (new H5, post hoc) | **H5 — Rehearsed Budgeted Fusion.** Added after the exploratory LOFO pilot (seed 0, `scripts/pilot_lofo.py`), where the KD student lost to RF/XGB `not_benign` on unseen families (H3 direction failed). Members: RF, XGB and MLP-CE `not_benign` (supervised block); IsolationForest on benign train and global-reference benign-cluster distance (benign-deviation block). Each member's score is mapped to a tail p-value `p = (1 + #{ref ≥ x}) / (n + 1)` against `benign_rank`; fused score `S = −min_m p_m / share_m`; threshold = higher empirical (1 − FPR)-quantile of `S` on `benign_cal`, applied with strict `>`. One free constant `alpha_sup` (supervised block share; the rest goes to the deviation block; equal split inside each block), grid 0.30–1.00 step 0.05, chosen by rehearsal (row below). **Win:** macro TPR@1%FPR over eligible families beats both XGB and faithful MCDE, 5-seed CI excluding 0, and no eligible family worse than either by more than 0.03. Reported per family. | The pilot showed tail complementarity on CICIoT (MLP-CE TPR@1%FPR Flood 1.00, BruteForce 0.41 vs RF 0.37/0.16). Budgeted p-value fusion is prior art (MCDE, Yue et al., Entropy 28(9):1026, 2026); what H5 adds is rehearsal-chosen shares, the multi-member supervised block and evaluation on the hard families. Fixed before any score-bank run. |
| 2026-10-24 | §6 (H5 rehearsal) | For outer fold k, `alpha_sup` is chosen on pair models trained on benign + F∖{k, j} (each with its own preprocessing and its own early-stopping split), for every inner family j ≠ k. Each pair model scores j's **train + val** rows as pseudo-unknowns; test rows never enter alpha selection. Objective: macro TPR@1%FPR over eligible inner families. Alphas within max(0.005, paired SE) of the best are tied, paired SE = std over inner families of (TPR_α − TPR_best) / √F; ties are broken by the highest harmonic mean of (TPR + 0.05), then the alpha closest to 0.9. If the whole curve lies inside the tolerance, `flat = True` is reported (the share does not matter). Pair models are cached by the unordered set {k, j}. | One tuned constant, because flexible combiners fit on few held-out domains overfit (leave-one-domain-out selection in DomainBed; outlier-set overfitting in OD-test). Fixed before any score-bank run. |
| 2026-10-24 | §6, §7 (gate protocol (a)) | Decisions use rehearsal (inner) families only. **Stage 2 ceiling gate:** for each eligible inner family j, `headroom_vs_trees` (best of Bonferroni union and recalibrated equal fusion, minus the best of RF and XGB, from `union_ceiling_check`) is averaged over the outer folds where j is inner, then a 5-seed t-CI is taken. Go if the macro CI over eligible inner families is > 0, or if at least one family has headroom > 0 on ≥ 4 of 5 seeds; otherwise fusion is reported as feature-capped for that dataset. **Stage 3 abort gate:** fusion is abandoned on a dataset if no member's (TPR@1%FPR − max(RF, XGB)) on any eligible inner family has a 5-seed CI > 0. Outer-family ceiling and seed tables are benchmark diagnostics and are never used for a decision. With many families × members some comparisons pass by chance; the seed-consistency rule is the guard against that. | Keeps every go/no-go decision off the test rows of the held-out family. Fixed before any score-bank run. |
| 2026-10-24 | §2, §7 (datasets and operating points) | `unsw_raw` (grouped `stime//600` split) is the primary UNSW dataset of the tail benchmark; `unsw_official` (stratified random split) is secondary. `ciciot_csv` is primary. BoT-IoT is reported by AUROC only (477 benign rows in the whole file, so a 1% threshold has no support). Fixed FPR set {0.1%, 1%, 5%}; the 0.1% column only where n_cal ≥ 10,000 (unsw_raw, ciciot_csv; not unsw_official). α_min: 0.1% for unsw_raw and ciciot_csv, 1% for unsw_official. Every threshold asserts n_cal ≥ 10 / FPR. Hard gate: calibration-set FPR ≤ target (holds by construction; asserted). Flag only: realised benign-test FPR, per score, family and dataset. pAUC is the standardized partial AUC over FPR ∈ [0, 1%] (McClish 1989; 0.5 = chance, 1 = perfect). Families with fewer than 300 rows scored in a role are reported but excluded from every macro objective and gate, identically for all methods, inside rehearsal too. | A random split leaves near-duplicate benign rows in val and test, which flatters the realised FPR; unsw_raw's grouped split does not. Budgeted fusion gives a deviation member 0.05% of the budget at alpha_sup = 0.9, so its p-values must resolve ~5e-4, which needs ~20k reference rows. Fixed before any score-bank run. |
| 2026-10-24 | §4, §5 (benign roles, training cap, early stopping) | Benign val is split 50/50 into `benign_rank` and `benign_cal` by group block (random on unsw_official). Benign val, rank, cal and test are uncapped. Training is capped at 40,000 rows per family, drawn once per seed by proportional allocation across training blocks (within-group random rank, never the first N rows); every outer and pair model uses the same rows of each family. IsolationForest and the k-means members use a fixed 200,000-row benign-train sample per seed. No member uses `benign_rank` or `benign_cal` for anything: early stopping, eval sets, hyperparameters or the choice of K. Early stopping uses a 10% group-carved split of each model's own training rows. Benign rows are reused across outer and inner configurations; benign is always a known class, so this is not leakage. | The pilot MLP early-stopped on val, which includes the benign rows that set the threshold; the pilot stays exploratory. Fixed before any score-bank run. |
| 2026-10-24 | §6 (MCDE baseline) | Faithful MCDE (Yue et al. 2026, §3.2–§4.2): supervised branch = HistGradientBoosting (binary benign vs known attack; sklearn defaults, the paper gives no hyperparameters); benign branch = median imputation, signed log, robust scaling to the 5th–95th percentile range, fitted on benign train; mini-batch k-means, K = 15, on benign train; RMS diagonal Mahalanobis distance to the nearest mode; global scale floored per coordinate at max(1e-3, 5% of the median global std among coordinates > 1e-6) (1e-3 if none); per-mode scale floored at 5% of the floored global scale; modes with fewer than 20 training vectors use the global scale. Tail probabilities u = (1 + #{ref ≥ x}) / (n + 1); supervised reference and mode-conditioned distance references come from MCDE's reference set = our `benign_rank`; a mode with fewer than n_min = ⌈5 / α_min⌉ reference rows uses the pooled reference (5,000 at α_min = 0.1%, 500 at 1%). `S = −min(u_sup / w, u_mode / (1 − w))`, w = 0.9; threshold = higher (1 − α) quantile on MCDE's threshold set = our `benign_cal`, strict `>`. Inside H5, the cluster-distance member uses the global reference; each table states which reference it uses. | Checked against the full text before freezing. The paper's CICIoT2023 stress test held out DDoS, DoS and Mirai one at a time with the other two in training; our joint hold-out (row below) measures that setting. |
| 2026-10-24 | §6 (extra CICIoT hold-outs) | In addition to the six families: (i) Flood + Mirai held out jointly (unseen-family; inner families BruteForce, Recon, Spoofing and Web only, so alpha is tuned on hard families, reported as such); (ii) DoS alone and (iii) DDoS alone, labelled **partial-family (attack-type) hold-outs**, not unseen-family results, because the sibling type stays in training. All three are rehearsed like the main folds. The Flood extrapolation check is reported: share of Flood test rows outside the training range on the top features, and whether RF/XGB scores saturate where the MLP score keeps rising. | `cic_family` merges DDoS and DoS into Flood while Mirai's floods (greeth, greip, udpplain) stay in training when Flood is held out, which may make Flood look easy. Fixed before any score-bank run. |
| 2026-10-24 | §3, §6 (unsw_raw `ct_*` features) | The creators computed the windowed `ct_*` features over the last 100 connections of all flows, attacks included, so training rows can encode the held-out family's flows. Before any full run we measure, per held-out family, the share of training rows and of benign-test rows whose window holds that family's flows, plus the share of windows crossing a `stime//600` block. If both shares are below 0.1% for every family, the original `ct_*` stay in the main variant and our causal `ct_*` are a robustness check; otherwise our causal `ct_*` are the main variant. Both are reported, with `ct_*` dropped as a third variant. Our causal `ct_*` recompute the 7 windowed features (`ct_src_ltm`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_srv_src`, `ct_srv_dst`); `ct_flw_http_mthd` and `ct_ftp_cmd` are per-flow and unchanged; `ct_state_ttl` stays dropped. Window = the smallest run of whole seconds back from t holding ≥ 100 flows, inside the row's own block, with the current second included in full (a 1-second decision latency); `stime` has 1-second resolution (96.6% of rows share a second) and the file is only 85–88% time-sorted, so file order is not used and the exposure count is approximate. Removal sets by role: for outer fold k, every row used to fit, early-stop, rank or calibrate is computed with k's flows removed, and only final test scoring (benign test plus k) uses all flows; for pair {k, j} in fold k's rehearsal, fit, early-stopping, rank and cal rows remove k and j, and j's pseudo-unknown rows remove k and keep j (fold j is the mirror). Windows are cached by removal set. A benign-test FPR above the calibration FPR is reported per family as the false-alarm cost of an unknown attack being present, never recalibrated away. | Same machinery as the per-host window features of Idea 2, built once. Fixed before any score-bank run. |
