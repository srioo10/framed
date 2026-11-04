# OnePass-IDS claims — nf_cic, seeds [1, 2, 3, 4], host-hour unit

calibration host-hours per fold (min over folds and seeds): 17047

(no mlp_tuned_int8 column in these banks)
| method | Bot | BruteForce | DDoS | DoS | Infiltration | Web | macro |
|---|---|---|---|---|---|---|---|
| attacker hosts / host-days | 8 / 113 | 1 / 5 | 10 / 135 | 3 / 8 | 260 / 1049 | 126 / 340 | |
| OnePass (host, mlp_ce, A) | 0.301 | 0.600 | 0.741 | 1.000 | 0.051 | 0.012 | 0.451 |
| per-flow MLP (none, P) | 0.124 | 0.000 | 0.504 | 0.250 | 0.075 | 0.056 | 0.168 |
| rf (none, P) | 0.000 | 0.000 | 0.000 | 0.000 | 0.165 | 0.347 | 0.085 |
| xgb (none, P) | 0.310 | 0.000 | 0.193 | 0.375 | 0.160 | 0.406 | 0.241 |
| mcde (none, P) | 0.053 | 0.400 | 0.919 | 0.750 | 0.140 | 0.138 | 0.400 |
| rf (host, A) upper ref | 0.885 | 1.000 | 0.830 | 1.000 | 0.078 | 0.000 | 0.632 |
| xgb (host, A) upper ref | 0.814 | 1.000 | 0.881 | 1.000 | 0.062 | 0.000 | 0.626 |
| mlp_ce_int8 (host, A) | 0.319 | 0.600 | 0.741 | 1.000 | 0.048 | 0.006 | 0.452 |
| mlp_sn_gda (host, A) | 0.850 | 1.000 | 0.941 | 1.000 | 0.038 | 0.074 | 0.650 |
| mlp_tuned (host, A) | 0.876 | 1.000 | 0.867 | 1.000 | 0.043 | 0.038 | 0.637 |
| mlp_bal (host, A) | 0.478 | 1.000 | 0.785 | 1.000 | 0.047 | 0.026 | 0.556 |
| mlp_smote (host, A) | 0.487 | 0.600 | 0.741 | 1.000 | 0.041 | 0.024 | 0.482 |
| host_perm (mlp_ce, A) | 0.027 | 0.000 | 0.637 | 0.375 | 0.077 | 0.053 | 0.195 |

C1 OnePass − per-flow MLP: +0.283 [+0.205, +0.352] → PASS
C2 OnePass − best large per-flow baseline (mcde): +0.051 [-0.011, +0.106] → PASS
C3 host gain +0.048, perm gain -0.208 → PASS
C4 realised benign host-hour alarm rate 0.0118 → PASS
C5 int8 loss -0.001 → PASS
C6 DDU − OnePass +0.200 [+0.164, +0.226] → PASS
T1 (host level): mlp_tuned − OnePass +0.187 [+0.151, +0.211] → PASS
B1 class-weighted (host level, reported): mlp_bal − OnePass +0.105 [+0.085, +0.125] → PASS
B1 SMOTE (host level, reported): mlp_smote − OnePass +0.031 [+0.004, +0.058] → PASS
C7 back-off accuracy (leaf case: ['BruteForce', 'DDoS', 'DoS', 'Web']) 0.508, chance 0.250 → FAIL
   per family: BruteForce 0.000 (n 52971), DDoS 0.897 (n 79455), DoS 0.415 (n 42748), Web 0.000 (n 22)
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
