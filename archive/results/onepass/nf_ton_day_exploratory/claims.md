# OnePass-IDS claims — nf_ton, seeds [0, 1, 2, 3, 4], host-day unit

calibration host-days per fold (min over folds and seeds): 151

(no mlp_tuned_int8 column in these banks)
| method | Backdoor | ddos | dos | injection | mitm | password | ransomware | scanning | xss | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| attacker hosts / host-days | 3 / 21 | 5 / 44 | 6 / 35 | 6 / 27 | 2 / 8 | 6 / 38 | 3 / 15 | 17 / 80 | 4 / 20 | |
| OnePass (host, mlp_ce, A) | 0.810 | 0.841 | 0.657 | 1.000 | 0.500 | 0.947 | 0.533 | 0.075 | 0.950 | 0.701 |
| per-flow MLP (none, P) | 0.667 | 0.773 | 0.371 | 0.889 | 0.875 | 0.842 | 0.467 | 0.150 | 1.000 | 0.670 |
| rf (none, P) | 0.524 | 0.955 | 0.743 | 1.000 | 0.875 | 0.868 | 0.667 | 0.225 | 1.000 | 0.762 |
| xgb (none, P) | 0.571 | 0.932 | 0.629 | 1.000 | 0.875 | 0.868 | 0.933 | 0.237 | 1.000 | 0.783 |
| mcde (none, P) | 0.095 | 0.682 | 0.086 | 0.963 | 0.250 | 0.447 | 0.200 | 0.125 | 0.650 | 0.389 |
| rf (host, A) upper ref | 0.810 | 0.886 | 0.829 | 1.000 | 0.875 | 1.000 | 0.667 | 0.200 | 1.000 | 0.807 |
| xgb (host, A) upper ref | 0.810 | 0.818 | 0.771 | 1.000 | 1.000 | 1.000 | 0.667 | 0.212 | 0.950 | 0.803 |
| mlp_ce_int8 (host, A) | 0.810 | 0.841 | 0.657 | 1.000 | 0.375 | 0.947 | 0.400 | 0.100 | 0.950 | 0.676 |
| mlp_sn_gda (host, A) | 0.286 | 0.568 | 0.314 | 0.000 | 0.875 | 0.079 | 0.200 | 0.100 | 0.050 | 0.275 |
| mlp_tuned (host, A) | 0.524 | 0.818 | 0.571 | 0.000 | 0.250 | 0.000 | 0.667 | 0.138 | 0.600 | 0.396 |
| mlp_bal (host, A) | 0.810 | 0.818 | 0.629 | 0.963 | 0.375 | 0.868 | 0.533 | 0.163 | 0.950 | 0.679 |
| mlp_smote (host, A) | 0.810 | 0.795 | 0.657 | 0.963 | 0.375 | 0.947 | 0.400 | 0.113 | 0.900 | 0.662 |
| host_perm (mlp_ce, A) | 0.667 | 0.705 | 0.371 | 1.000 | 0.875 | 0.868 | 0.733 | 0.138 | 0.950 | 0.701 |

C1 OnePass − per-flow MLP: +0.031 [-0.020, +0.091] → FAIL
C2 OnePass − best large per-flow baseline (xgb): -0.081 [-0.116, -0.037] → FAIL
C3 host gain -0.107, perm gain -0.108 → PASS
C4 realised benign host-day alarm rate 0.0249 → FAIL
C5 int8 loss +0.026 → FAIL
C6 DDU − OnePass -0.427 [-0.486, -0.366] → FAIL
T1 (host level): mlp_tuned − OnePass -0.305 [-0.336, -0.277] → FAIL
B1 class-weighted (host level, reported): mlp_bal − OnePass -0.023 [-0.055, +0.004] → FAIL
B1 SMOTE (host level, reported): mlp_smote − OnePass -0.039 [-0.074, -0.009] → FAIL
C7 back-off accuracy (leaf case: ['Backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'xss']) 0.599, chance 0.200 → FAIL
   per family: Backdoor 0.006 (n 19741), ddos 0.023 (n 98435), dos 0.545 (n 56970), injection 0.882 (n 95802), password 0.899 (n 96010), ransomware 0.625 (n 1007), xss 0.778 (n 88197)
underpowered (< 5 attacker hosts): ['Backdoor', 'mitm', 'ransomware', 'xss']
