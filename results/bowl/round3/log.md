# Round 3 roster rationale

## Round 2 result (Kaggle kernel v6; 256 fight trials, 248 within limits; full arena = CTU-13 + UGR'16, 12 pairs × 2 seeds)
- Champion: trial 63, a plain MLP (raw features, w = 64, depth 2, SELU, nlog head), KD 0.5 to OOF XGB plus cross-context distillation (syn 0.5). 6.3 KB, speed ratio 0.04 (25× faster than heavy XGB).
  Fitness 0.810 (unknown 0.633, known 0.987).
- Bosses: RF 0.816, XGB 0.770. Gap to RF: 0.006, down from 0.021 in round 1. Stop rule not met.
- 3 of the top 3 finalists use syn 0.5. The nlog head is flood-safe (§9 ENG-X lesson: min-member failed).

## Autopsy
**Seed instability is the dominant pattern.** The champion's fitness is 0.832 on seed 1 (above RF) and 0.788 on seed 2.
- Unknown TPR swings by pair:
  - CTU Murlo: 0.864 vs 0.303;
  - CTU Rbot: 0.981 vs 0.785;
  - UGR pair_2 nerisbotnet: 0.869 vs 0.614.
- Known TPR is stable (±0.006).

How the network extrapolates to unseen families depends on its initialisation.
On seed 1, the boss-caught, champion-missed slices are small (≤ 0.08 per pair), with no single conjunction left like round 1's pair_4.

## Power-up
**ensemble:** a deep ensemble (Lakshminarayanan et al. NeurIPS'17) of 3 independently initialised bodies trained side by side. The nlog head averages P(benign) over members.
This directly targets the seed-dependent extrapolation, and it stays inside the limits (CIC pair 0: 25 KB, ratio 0.20).
- It is ablated against the round-2 champion as before (cheap arena, now including nf_cic, seeds 0 and 1). If the ablation drops it, ens is pinned to 1.
- syn options widened to {0, 0.5, 1.0}. syn is the kept round-2 power-up; this only widens its choices and is not a new power-up.

## Arena change
nf_cic is added (6 pairs, 59 features: 39 flow + 20 host context), as pre-registered once its arena is built.
Bosses are re-scored on the 18-pair arena.

## Speed (protocol unchanged)
- 4 fight workers share one Optuna study, and the trial cap holds across workers.
- The death match is split over 4 workers, then merged; tested identical to a single process.
- Batch 4096 was tested locally before the round and **rejected**: the champion fell from 0.78 to 0.61 (undertraining at about 9 steps per epoch). Batch stays 1024.

## Roster
- Families: mlp, tabm.
- 8 seeds:
  - champion ± ensemble;
  - champion with syn 1.0 (single and ensemble);
  - trials 33, 40 and 204 as ensembles;
  - the champion ensemble on quantile bins.
