# Host-context arm — unsw_raw (seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | ct_drop | ct_causal | host − none (paired) |
|---|---|---|---|---|
| Analysis+Backdoor | 132 | 0.755 [0.557, 0.953] (xgb) | 0.646 [0.435, 0.856] (mcde) | -0.109 [-0.230, 0.011] |
| DoS | 690 | 0.943 [0.904, 0.981] (mcde) | 0.979 [0.965, 0.993] (xgb) | 0.036 [0.009, 0.064] |
| Exploits | 4980 | 0.945 [0.918, 0.973] (rf) | 0.959 [0.944, 0.973] (mcde) | 0.013 [-0.003, 0.030] |
| Fuzzers | 3590 | 0.740 [0.556, 0.925] (mlp_ce) | 0.589 [0.452, 0.727] (rf) | -0.151 [-0.261, -0.040] |
| Generic | 3243 | 0.969 [0.925, 1.013] (mcde) | 0.730 [0.584, 0.877] (iforest) | -0.239 [-0.399, -0.079] |
| Recon | 1634 | 0.797 [0.653, 0.941] (mcde) | 0.965 [0.937, 0.993] (rf) | 0.168 [0.050, 0.286] |
| Shellcode+Worms | 303 | 0.876 [0.778, 0.973] (rf) | 0.981 [0.947, 1.014] (mcde) | 0.105 [0.022, 0.188] |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold               Analysis+Backdoor    DoS  Exploits  Fuzzers  Generic  Recon  Shellcode+Worms
member  variant                                                                                
clust   ct_causal              0.138  0.143     0.162    0.081    0.130  0.110            0.051
        ct_drop                0.247  0.229     0.273    0.205    0.380  0.186            0.177
iforest ct_causal              0.039  0.091     0.060    0.096    0.730  0.048            0.141
        ct_drop                0.116  0.155     0.096    0.231    0.400  0.087            0.280
mcde    ct_causal              0.582  0.973     0.949    0.429    0.252  0.947            0.969
        ct_drop                0.636  0.931     0.932    0.625    0.960  0.778            0.850
mlp_ce  ct_causal              0.570  0.967     0.939    0.497    0.232  0.912            0.958
        ct_drop                0.700  0.902     0.890    0.740    0.223  0.591            0.756
rf      ct_causal              0.614  0.970     0.943    0.589    0.231  0.962            0.956
        ct_drop                0.725  0.937     0.943    0.702    0.953  0.777            0.873
xgb     ct_causal              0.617  0.975     0.927    0.522    0.211  0.928            0.910
        ct_drop                0.702  0.937     0.935    0.685    0.843  0.715            0.804
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant  ct_causal  ct_drop
member                     
clust       0.0172   0.0106
iforest     0.0113   0.0120
mcde        0.0104   0.0102
mlp_ce      0.0103   0.0103
rf          0.0101   0.0104
xgb         0.0099   0.0102
```
