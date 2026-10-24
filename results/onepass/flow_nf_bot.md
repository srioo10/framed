# Host-context arm — nf_bot (seeds [np.int64(0)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host − none (paired) |
|---|---|---|---|
| DDoS | 20000 | 0.998 (xgb) | — |
| DoS | 20000 | 1.000 (mlp_sn) | — |
| Reconnaissance | 20000 | 0.009 (clust) | — |
| Theft | 120 | 0.717 (mlp_ce) | — |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold                  DDoS    DoS  Reconnaissance  Theft
member      variant                                     
clust       none     0.534  0.868           0.010  0.067
iforest     none     0.076  0.354           0.002  0.042
mcde        none     0.990  0.849           0.007  0.133
mlp_ce      none     0.990  0.997           0.006  0.717
mlp_ce_int8 none     0.994  0.995           0.000  0.617
mlp_sn      none     0.997  1.000           0.008  0.092
mlp_sn_gda  none     0.008  0.010           0.009  0.133
rf          none     0.992  0.991           0.004  0.100
xgb         none     0.998  0.998           0.009  0.425
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant        none
member             
clust        0.0123
iforest      0.0103
mcde         0.0032
mlp_ce       0.0087
mlp_ce_int8  0.0085
mlp_sn       0.0117
mlp_sn_gda   0.0141
rf           0.0079
xgb          0.0092
```
