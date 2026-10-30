# Round 2 roster rationale

## Round 1 (Kaggle kernel v4; 256 fight trials; 10 finalists; full arena = CTU-13 + UGR'16, 12 pairs, 2 seeds)
- Champion: trial 125. TabM with quantile bins (e = 4, w = 32, depth 1), 4 members, min-member head, KD 0.5 to OOF XGB. 9.9 KB, speed ratio 0.24.
  Fitness 0.795 (unknown 0.602, known 0.989).
- Boss XGB: 0.770 (unknown 0.548, known 0.992). The champion beats it.
- Boss RF: 0.816 (unknown 0.638, known 0.994). The champion loses.
- The standalone NODE and LeafNet fighters lost. The best LeafNet finalist scored 0.754.

## Autopsy (vs both bosses)
The gap to RF is nearly all CTU pair_4, where Rbot is the unknown family: champion 0.766, RF 0.991, XGB 0.584.
- The depth-3 tree on boss-caught, champion-missed rows splits on L4_DST_PORT AND hc_dst_newpeer_10s AND
  hc_src_n_120s ≤ 0.26 (z-score). These are low-rate flows from quiet source hosts.
- CTU pair_1 (unknown Murlo): hc_src_newport_120s high AND hc_dst_bytes_120s high.

Hidden pattern: RF isolates sparse axis-aligned conjunctions of host-context counters, and the smooth net does not.
KD to XGB cannot transfer them: XGB itself misses pair_4.
- UGR spam pairs (1, 3, 5) sit at about 0 for every detector, bosses included, so they are uninformative.

## Power-ups (each ablated vs the champion on the cheap arena, seeds 0 and 1; kept only if Δ > the champion's seed difference)
- **rf_teacher:** KD to 3-fold OOF RF (100 trees, min leaf 2, 0.5 bootstrap) instead of OOF XGB. RF is the boss that catches the pattern.
- **syn (cross-context distillation):** synthetic rows made of fit-row flow features plus another fit row's host context. An RF fit on the fit rows labels them, and the student matches it (KL, T = 4, weight 0.5).
  This puts RF's conjunction boundaries into the off-manifold region where unknown families appear. Fit rows only.
- **node_branch:** crossover of the champion with the losing NODE. A NODE-lite branch (hard oblivious splits) is added to the TabM logits, giving axis-aligned AND units inside the network.
- Informational only: the champion with the −log P(benign) head instead of the min-member head. tabm_kd_min failed the flood rule (48% spoofed FPR), so a flood-safe head is a fallback.

## Roster
- Families: tabm, mlp, tabm_node. Standalone NODE and LeafNet are dropped (they lost); NODE survives as a branch.
- 9 seed genomes:
  - the champion, plus each power-up alone, plus all three;
  - trial 219 (8 members, spectral norm) + rf + syn;
  - trial 221 (selu, KD 0.9, focal) + NODE branch;
  - trial 229 (MLP w64, spectral norm, focal) + rf teacher;
  - champion + all three with the nlog head.
- Power-ups the ablation drops are pinned off for the fight.

## Ablation result (Kaggle kernel v5; cheap arena, seeds 0 / 1)
| variant | fitness s0 / s1 | mean | unknown | known |
|---|---|---|---|---|
| champion (base) | 0.8168 / 0.8197 | 0.8182 | 0.649 | 0.987 |
| + rf_teacher | 0.8124 / 0.8184 | 0.8154 | 0.653 | 0.978 |
| + syn | 0.8182 / 0.8249 | 0.8215 | 0.664 | 0.979 |
| + node_branch | 0.8133 / 0.8162 | 0.8147 | 0.647 | 0.982 |
| all three | 0.8132 / 0.8102 | 0.8117 | 0.656 | 0.968 |
| base, nlog head (info) | 0.8164 / 0.8194 | 0.8179 | 0.649 | 0.987 |

Noise (base seed difference) is 0.0029.
- **Kept:** syn (+0.0033, just above the noise).
- **Dropped and pinned off:** rf_teacher, node_branch.

Each power-up raises unknown TPR slightly but costs known-attack TPR. The pair_4 smoke gain (0.70 → 0.98) did not carry over to the arena average.
The flood-safe nlog head matches the min-member head (Δ −0.0003).

The fight's first attempt crashed: an enqueued seed named the dropped family. Seeds are now filtered through the pins. The round-2 study was deleted and restarted. No fight trial had been scored.
