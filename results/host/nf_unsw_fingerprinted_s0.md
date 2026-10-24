# Host-context arm — nf_unsw (seeds [np.int64(0)])

TPR@1%FPR on the held-out family, best member per seed (member most often best in brackets); 5-seed mean [CI]

| family | n | none | host | host − none (paired) |
|---|---|---|---|---|
| Analysis+Backdoor | 495 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| DoS | 1090 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| Exploits | 7378 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| Fuzzers | 4865 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| Generic | 980 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| Recon | 2222 | 1.000 (rf) | 1.000 (rf) | 0.000 |
| Shellcode+Worms | 437 | 1.000 (rf) | 1.000 (rf) | 0.000 |

Same member, TPR@1%FPR mean over seeds (rows: member × variant)

```
fold             Analysis+Backdoor    DoS  Exploits  Fuzzers  Generic  Recon  Shellcode+Worms
member  variant                                                                              
clust   host                 0.069  0.149     0.150    0.044    0.163  0.153            0.410
        none                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
iforest host                 0.469  0.767     0.771    0.465    0.674  0.829            0.931
        none                 0.018  0.008     0.009    0.021    0.040  0.011            0.002
mcde    host                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
        none                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
mlp_ce  host                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
        none                 1.000  1.000     0.999    1.000    1.000  1.000            1.000
rf      host                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
        none                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
xgb     host                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
        none                 1.000  1.000     1.000    1.000    1.000  1.000            1.000
```

Realised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds

```
variant    host    none
member                 
clust    0.0099  0.0100
iforest  0.0093  0.0092
mcde     0.0101  0.0105
mlp_ce   0.0082  0.0090
rf       0.0100  0.0088
xgb      0.0099  0.0086
```
