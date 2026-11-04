# CICIoT results text (draft, numbers from stage5_ciciot_csv.md and diagnostics_ciciot_csv.md)

**Labels.** H5 is post hoc: it was added after the exploratory pilot. The two single-member comparisons (MLP-CE
alone; the rehearsal-selected member) are post hoc, added after Stage 3. The fixed-alpha, leave-one-family-out and
leave-one-member-out analyses are post hoc diagnostics (PREREG §9, 2026-10-24 rows).

**Main result.** Over the six unseen-family hold-outs, rehearsed budgeted fusion reaches a macro TPR of 0.471 at 1%
benign FPR. It beats XGB by +0.076 [+0.058, +0.094] and faithful MCDE by +0.010 [+0.002, +0.018] (wider of the
5-seed t-CI and the two-level bootstrap), with no family more than 0.03 worse than either (worst: Web, −0.025 vs
MCDE). The MCDE margin comes from Flood, where MLP-CE covers what the tree branch misses: without MLP-CE the fused
Flood TPR falls from 0.992 to 0.771. It costs about 0.02 on Spoofing (−0.021) and Web (−0.025). Without Flood,
fusion minus MCDE is −0.004 [−0.008, −0.001]: on the other five families fusion does not beat MCDE.

**Alpha.** Fixed alpha = 0.9 (MCDE's w) gives the same macro TPR as rehearsal (0.472 vs 0.471), and equal shares
give 0.470. The contribution is therefore multi-member budgeted fusion; rehearsal is a leak-free way to pick alpha
that turns out not to matter much on CICIoT.

**Low FPR.** At 0.1% FPR, MLP-CE alone is better than fusion on BruteForce (0.112 vs 0.074), Recon (0.133 vs
0.096) and Flood (0.949 vs 0.896).

**Members (leave-one-member-out, alpha re-rehearsed).** MLP-CE carries most of the gain (macro 0.471 → 0.428
without it) and cluster distance the next largest share (→ 0.456; BruteForce 0.284 → 0.233, Recon 0.260 → 0.229).
XGB supports Spoofing and Web (without it 0.124 / 0.113). IsolationForest adds nothing measurable (0.474 without
it). RF lowers the result (0.480 without it; Spoofing 0.171, Web 0.160); this was found post hoc and is not used to
change the method. MCDE's own branches plus MLP-CE (HGB and MLP-CE sharing w = 0.9, mode-conditioned distance)
reach 0.485, +0.024 [+0.014, +0.034] over MCDE; this is a post hoc, unrehearsed variant.

**MCDE settings the paper does not give:** sklearn default HistGradientBoosting; mini-batch k-means with batch
size 4,096 and n_init 3. Categorical fields are one-hot encoded with block weight 0.5 as in the paper (no
categorical fields in CICIoT).
