# Host-context arm — unsw_official (seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host − none (paired) |
|---|---|---|---|
| Analysis+Backdoor | 160 | 0.456 [0.423, 0.490] (mcde) | — |
| DoS | 724 | 0.848 [0.838, 0.858] (mcde) | — |
| Exploits | 5090 | 0.821 [0.800, 0.841] (mcde) | — |
| Fuzzers | 3750 | 0.099 [0.095, 0.104] (rf) | — |
| Generic | 1456 | 0.935 [0.889, 0.981] (mcde) | — |
| Recon | 1684 | 0.490 [0.431, 0.549] (rf) | — |
| Shellcode+Worms | 321 | 0.768 [0.694, 0.842] (mcde) | — |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold             Analysis+Backdoor    DoS  Exploits  Fuzzers  Generic  Recon  Shellcode+Worms
member  variant                                                                              
clust   none                 0.071  0.137     0.183    0.040    0.860  0.005            0.016
iforest none                 0.004  0.001     0.000    0.004    0.098  0.000            0.000
mcde    none                 0.451  0.841     0.820    0.084    0.837  0.449            0.768
mlp_ce  none                 0.404  0.760     0.752    0.075    0.255  0.444            0.621
rf      none                 0.431  0.808     0.803    0.099    0.365  0.452            0.606
xgb     none                 0.434  0.839     0.796    0.091    0.506  0.454            0.492
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant    none
member         
clust    0.0100
iforest  0.0096
mcde     0.0096
mlp_ce   0.0099
rf       0.0093
xgb      0.0095
```
