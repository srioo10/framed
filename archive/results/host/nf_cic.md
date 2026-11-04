# Host-context arm — nf_cic (seeds [np.int64(0)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host | host_iso | host_perm | host − none (paired) |
|---|---|---|---|---|---|---|
| Bot | 505 | 0.370 (clust) | 0.760 (xgb) | 0.818 (mcde) | 0.095 (clust) | 0.390 |
| BruteForce | 18777 | 1.000 (mcde) | 0.963 (rf) | 0.963 (rf) | 0.437 (mcde) | -0.037 |
| DDoS | 20000 | 0.540 (clust) | 0.997 (mlp_ce) | 0.997 (mlp_ce) | 0.665 (clust) | 0.457 |
| DoS | 20000 | 0.739 (clust) | 0.996 (mlp_ce) | 0.997 (mlp_ce) | 0.712 (clust) | 0.258 |
| Infiltration | 5295 | 0.111 (rf) | 0.156 (rf) | 0.151 (rf) | 0.107 (rf) | 0.045 |
| Web | 151 | 0.801 (mcde) | 0.093 (mlp_ce) | 0.146 (clust) | 0.596 (rf) | -0.709 |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold                 Bot  BruteForce   DDoS    DoS  Infiltration    Web
member  variant                                                        
clust   host       0.000       0.000  0.004  0.873         0.027  0.093
        host_iso   0.000       0.000  0.004  0.881         0.027  0.146
        host_perm  0.095       0.063  0.665  0.712         0.047  0.007
        none       0.370       0.000  0.540  0.739         0.060  0.007
iforest host       0.000       0.000  0.004  0.000         0.007  0.013
        host_iso   0.000       0.000  0.004  0.000         0.007  0.013
        host_perm  0.002       0.000  0.002  0.597         0.016  0.033
        none       0.000       0.000  0.004  0.489         0.010  0.013
mcde    host       0.741       0.932  0.980  0.987         0.037  0.000
        host_iso   0.818       0.904  0.987  0.986         0.036  0.007
        host_perm  0.087       0.437  0.490  0.024         0.071  0.517
        none       0.271       1.000  0.474  0.028         0.049  0.801
mlp_ce  host       0.081       0.000  0.996  0.996         0.078  0.093
        host_iso   0.624       0.000  0.997  0.997         0.083  0.106
        host_perm  0.002       0.012  0.410  0.012         0.032  0.556
        none       0.000       0.991  0.155  0.092         0.029  0.364
rf      host       0.303       0.963  0.976  0.965         0.156  0.000
        host_iso   0.709       0.963  0.970  0.955         0.151  0.000
        host_perm  0.016       0.000  0.000  0.000         0.107  0.596
        none       0.000       0.000  0.000  0.000         0.111  0.477
xgb     host       0.760       0.000  0.969  0.980         0.107  0.000
        host_iso   0.806       0.000  0.928  0.978         0.106  0.000
        host_perm  0.034       0.047  0.002  0.030         0.101  0.536
        none       0.250       0.000  0.004  0.000         0.066  0.543
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant    host  host_iso  host_perm    none
member                                      
clust    0.0097    0.0098     0.0107  0.0105
iforest  0.0115    0.0113     0.0102  0.0094
mcde     0.0052    0.0047     0.0075  0.0082
mlp_ce   0.0062    0.0060     0.0107  0.0100
rf       0.0104    0.0101     0.0115  0.0106
xgb      0.0102    0.0097     0.0096  0.0098
```
