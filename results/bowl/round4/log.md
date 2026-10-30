# Round 4 roster rationale (cert round, PREREG §9 CERT)

## Round 3 result (Kaggle kernel v7; 259 fight trials, all within limits; full arena = CTU-13 + UGR'16 + CIC, 18 pairs × 2 seeds)
- 259 > 256: the four workers' trial-cap check races, so up to 3 extra trials ran (disclosed; the cap is checked after each trial).
- Ablation: **ensemble dropped.** It did not beat the champion by more than the seed noise (0.028). ens pinned to 1.
- Champion: trial 4, MLP (raw, w = 64, depth 2, ReLU, **energy head**), KD 0.5, syn 0.5. 6.1 KB, speed ratio 0.02.
  Fitness 0.802 (unknown **0.661**, known 0.943).
- Bosses (18 pairs): RF 0.819 (unknown 0.657, known 0.981), gang 0.817, cat 0.812, lgbm 0.799, xgbx 0.797, XGB 0.771.
- **The champion is above RF on unknown TPR (0.661 vs 0.657) and loses on known TPR (0.943 vs 0.981).** The gap to RF
  (0.017) is all on known families.

## Autopsy (champion vs RF unknown TPR per pair)
- Champion far ahead: CTU Murlo 0.88 vs 0.28; CIC Web 0.32 vs 0.00; CIC Bot 0.93 vs 0.87.
- Champion far behind: CTU Rbot 0.48 vs 0.99; UGR nerisbotnet (R0 dos) 0.53 vs 0.90; CIC Infiltration 0.21 vs 0.33.
- Spam (3 UGR pairs) is caught by no detector at 1% FPR (all ≤ 0.01).

## Round 4 settings (as fixed in PREREG §9 CERT round-4 row)
- cert_mode: host-context clip on, cert λ ∈ {0.5, 1, 2, 4} with warm-up, constraint certified FPR(10) ≤ 2%.
- Encoders raw / quantile / tree (monotone only); heads nlog (single) and nlog / min (TabM members). The energy head
  of the round-3 champion has no certificate, so its body is seeded with the nlog head.
- Genes: syn {0, 0.5, 1}. No new power-up (powerups []), so no ablation step. ens stays 1 (dropped in round 3).
- Budget: 256 trials as every round. Wall-clock brake at 8 h per fight worker only to avoid Kaggle's 12 h kill;
  if it fires, the actual trial count is reported.
- Bosses in the death match: RF, XGB, the ultra bosses, and the tuned boss once its run finishes (scored locally on
  the same pairs and seeds).

## Seeds (8)
- Round-3 champion body, nlog head, cert 2 and 4.
- Round-2 champion, cert 1, 2, 4; the same on tree bins, cert 2.
- Round-2 trial 40 (quantile MLP), cert 2. Round-1 champion (TabM, min head), cert 2.
