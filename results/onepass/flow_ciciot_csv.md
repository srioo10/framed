# Host-context arm — ciciot_csv (seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host − none (paired) |
|---|---|---|---|
| BruteForce | 2605 | 0.315 [0.266, 0.363] (clust) | — |
| DDoS | 15390 | 1.000 [1.000, 1.000] (mlp_ce) | — |
| DoS | 4601 | 1.000 [1.000, 1.000] (mlp_ce) | — |
| Flood | 20000 | 0.998 [0.995, 1.001] (mlp_ce) | — |
| Flood__Mirai | 40000 | 0.945 [0.927, 0.964] (mlp_ce) | — |
| Mirai | 20000 | 1.000 [1.000, 1.000] (mlp_ce) | — |
| Recon | 20000 | 0.243 [0.215, 0.271] (mcde) | — |
| Spoofing | 20000 | 0.184 [0.142, 0.226] (mcde) | — |
| Web | 4809 | 0.181 [0.119, 0.242] (mcde) | — |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold             BruteForce   DDoS    DoS  Flood  Flood__Mirai  Mirai  Recon  Spoofing    Web
member  variant                                                                              
clust   none          0.256  0.930  0.757  0.890         0.915  0.939  0.216     0.087  0.054
iforest none          0.085  0.671  0.599  0.655         0.739  0.824  0.206     0.119  0.081
mcde    none          0.279  0.999  1.000  0.909         0.885  1.000  0.242     0.181  0.155
mlp_ce  none          0.262  1.000  1.000  0.998         0.934  1.000  0.223     0.087  0.163
rf      none          0.124  0.983  0.993  0.346         0.449  0.933  0.084     0.119  0.112
xgb     none          0.196  0.999  0.999  0.677         0.620  0.973  0.193     0.177  0.152
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant    none
member         
clust    0.0100
iforest  0.0103
mcde     0.0097
mlp_ce   0.0103
rf       0.0097
xgb      0.0095
```
