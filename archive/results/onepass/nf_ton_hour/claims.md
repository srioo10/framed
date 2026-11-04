# OnePass-IDS claims — nf_ton, seeds [1, 2, 3, 4], host-hour unit

calibration host-hours per fold (min over folds and seeds): 185

(no mlp_tuned_int8 column in these banks)
| method | Backdoor | ddos | dos | injection | mitm | password | ransomware | scanning | xss | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| attacker hosts / host-days | 3 / 93 | 5 / 140 | 6 / 88 | 6 / 35 | 2 / 9 | 6 / 91 | 3 / 16 | 16 / 94 | 4 / 56 | |
| OnePass (host, mlp_ce, A) | 0.957 | 0.886 | 0.727 | 0.971 | 0.556 | 0.989 | 0.562 | 0.074 | 1.000 | 0.747 |
| per-flow MLP (none, P) | 0.925 | 0.779 | 0.364 | 1.000 | 0.778 | 0.835 | 0.500 | 0.181 | 1.000 | 0.707 |
| rf (none, P) | 0.516 | 0.871 | 0.761 | 1.000 | 1.000 | 0.758 | 0.812 | 0.362 | 1.000 | 0.787 |
| xgb (none, P) | 0.634 | 0.871 | 0.625 | 1.000 | 1.000 | 0.813 | 0.938 | 0.319 | 0.875 | 0.786 |
| mcde (none, P) | 0.183 | 0.464 | 0.080 | 0.914 | 0.333 | 0.451 | 0.188 | 0.138 | 0.571 | 0.369 |
| rf (host, A) upper ref | 0.968 | 0.929 | 0.841 | 1.000 | 0.889 | 1.000 | 0.750 | 0.245 | 1.000 | 0.847 |
| xgb (host, A) upper ref | 0.978 | 0.893 | 0.818 | 1.000 | 1.000 | 1.000 | 0.750 | 0.277 | 0.982 | 0.855 |
| mlp_ce_int8 (host, A) | 0.957 | 0.829 | 0.716 | 0.971 | 0.444 | 1.000 | 0.500 | 0.085 | 0.946 | 0.717 |
| mlp_sn_gda (host, A) | 0.409 | 0.500 | 0.284 | 0.000 | 0.778 | 0.033 | 0.188 | 0.085 | 0.018 | 0.255 |
| mlp_tuned (host, A) | 0.634 | 0.886 | 0.557 | 0.000 | 0.333 | 0.000 | 0.688 | 0.138 | 0.750 | 0.443 |
| mlp_bal (host, A) | 0.903 | 0.886 | 0.750 | 0.971 | 0.444 | 0.901 | 0.562 | 0.160 | 1.000 | 0.731 |
| mlp_smote (host, A) | 0.946 | 0.893 | 0.773 | 0.971 | 0.333 | 0.945 | 0.438 | 0.117 | 1.000 | 0.713 |
| host_perm (mlp_ce, A) | 0.968 | 0.814 | 0.477 | 0.971 | 0.889 | 0.934 | 0.750 | 0.170 | 1.000 | 0.775 |

C1 OnePass − per-flow MLP: +0.040 [-0.007, +0.135] → FAIL
C2 OnePass − best large per-flow baseline (rf): -0.040 [-0.098, +0.027] → FAIL
C3 host gain -0.043, perm gain -0.015 → FAIL
C4 realised benign host-hour alarm rate 0.0825 → FAIL
C5 int8 loss +0.030 → FAIL
C6 DDU − OnePass -0.492 [-0.562, -0.438] → FAIL
T1 (host level): mlp_tuned − OnePass -0.304 [-0.358, -0.279] → FAIL
B1 class-weighted (host level, reported): mlp_bal − OnePass -0.016 [-0.061, +0.014] → FAIL
B1 SMOTE (host level, reported): mlp_smote − OnePass -0.034 [-0.070, -0.000] → FAIL
C7 back-off accuracy (leaf case: ['Backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'xss']) 0.590, chance 0.200 → FAIL
   per family: Backdoor 0.007 (n 15741), ddos 0.024 (n 78457), dos 0.512 (n 50564), injection 0.870 (n 74219), password 0.897 (n 77972), ransomware 0.683 (n 912), xss 0.759 (n 76387)
underpowered (< 5 attacker hosts): ['Backdoor', 'mitm', 'ransomware', 'xss']
