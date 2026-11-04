# Host-context arm — nf_bot (seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host | host − none (paired) |
|---|---|---|---|---|
| DDoS | 20000 | 0.947 [0.808, 1.086] (iforest) | 0.995 [0.987, 1.004] (rf) | 0.048 [-0.091, 0.188] |
| DoS | 20000 | 0.958 [0.858, 1.057] (mlp_ce) | 0.992 [0.981, 1.003] (mlp_ce_int8) | 0.034 [-0.066, 0.135] |
| Reconnaissance | 20000 | 0.326 [-0.158, 0.810] (clust) | 0.863 [0.522, 1.204] (rf) | 0.537 [0.048, 1.026] |
| Theft | 13 | 0.740 [0.315, 1.165] (mcde) | 0.648 [0.126, 1.170] (clust) | -0.092 [-0.449, 0.264] |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold                     DDoS    DoS  Reconnaissance  Theft
member         variant                                     
clust          host     0.583  0.729           0.211  0.269
               none     0.376  0.507           0.057  0.228
iforest        host     0.507  0.520           0.007  0.372
               none     0.236  0.391           0.010  0.426
mcde           host     0.400  0.397           0.466  0.310
               none     0.598  0.562           0.105  0.389
mlp_bal        host     0.987  0.953           0.478  0.276
mlp_ce         host     0.978  0.976           0.283  0.228
               none     0.797  0.906           0.144  0.387
mlp_ce_int8    host     0.974  0.970           0.557  0.228
               none     0.793  0.935           0.225  0.370
mlp_smote      host     0.975  0.974           0.376  0.230
mlp_sn         host     0.972  0.968           0.224  0.214
               none     0.688  0.850           0.069  0.111
mlp_sn_gda     host     0.052  0.124           0.029  0.558
               none     0.030  0.043           0.025  0.293
mlp_tuned      host     0.432  0.667           0.369  0.243
mlp_tuned_int8 host     0.999  0.994           0.566  0.254
rf             host     0.927  0.647           0.832  0.218
               none     0.839  0.733           0.183  0.278
xgb            host     0.804  0.943           0.449  0.094
               none     0.911  0.953           0.268  0.252
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant           host    none
member                        
clust           0.1479  0.0847
iforest         0.0204  0.0245
mcde            0.1569  0.1477
mlp_bal         0.1584     NaN
mlp_ce          0.1633  0.1466
mlp_ce_int8     0.1485  0.1227
mlp_smote       0.1645     NaN
mlp_sn          0.1454  0.1471
mlp_sn_gda      0.0180  0.0369
mlp_tuned       0.1477     NaN
mlp_tuned_int8  0.2031     NaN
rf              0.1616  0.1610
xgb             0.1613  0.1339
```
