# Post hoc fusion diagnostics — /home/srioo/Desktop/Projects/root/BigP/runs/bank/ciciot_csv_none_cap40000_s?

TPR@1%FPR; eligible main families ['BruteForce', 'Flood', 'Mirai', 'Recon', 'Spoofing', 'Web']; CI used = wider of the 5-seed t-CI and the two-level bootstrap (PREREG §9 Stage 5 rule).

## (i) Alpha: rehearsed vs fixed

| variant | macro TPR@1% | minus XGB | minus MCDE | minus MLP-CE |
|---|---|---|---|---|
| rehearsed | 0.471 [0.458, 0.484] | +0.076 [+0.058, +0.094] ✔ | +0.010 [+0.002, +0.018] ✔ | +0.015 [+0.002, +0.029] ✔ |
| fixed 0.90 | 0.472 [0.452, 0.491] | +0.077 [+0.054, +0.100] ✔ | +0.011 [-0.004, +0.026] | +0.016 [+0.006, +0.027] ✔ |
| equal (0.60) | 0.470 [0.461, 0.479] | +0.075 [+0.053, +0.097] ✔ | +0.009 [+0.001, +0.017] ✔ | +0.014 [-0.004, +0.032] |

## (iii) Leave-one-member-out (alpha re-rehearsed) and MCDE + MLP-CE

| variant | macro TPR@1% | minus XGB | minus MCDE | minus MLP-CE |
|---|---|---|---|---|
| rehearsed | 0.471 [0.458, 0.484] | +0.076 [+0.058, +0.094] ✔ | +0.010 [+0.002, +0.018] ✔ | +0.015 [+0.002, +0.029] ✔ |
| without rf | 0.480 [0.469, 0.491] | +0.085 [+0.063, +0.108] ✔ | +0.019 [+0.010, +0.029] ✔ | +0.025 [+0.015, +0.035] ✔ |
| without xgb | 0.458 [0.447, 0.470] | +0.064 [+0.045, +0.083] ✔ | -0.002 [-0.011, +0.006] | +0.003 [-0.009, +0.015] |
| without mlp_ce | 0.428 [0.412, 0.445] | +0.034 [+0.013, +0.055] ✔ | -0.032 [-0.048, -0.017] | -0.027 [-0.048, -0.007] |
| without iforest | 0.474 [0.458, 0.489] | +0.079 [+0.063, +0.094] ✔ | +0.013 [+0.002, +0.024] ✔ | +0.018 [+0.007, +0.029] ✔ |
| without clust | 0.456 [0.436, 0.477] | +0.062 [+0.036, +0.087] ✔ | -0.004 [-0.022, +0.013] | +0.001 [-0.009, +0.011] |
| MCDE + MLP-CE (w 0.9) | 0.485 [0.474, 0.496] | +0.090 [+0.070, +0.110] ✔ | +0.024 [+0.014, +0.034] ✔ | +0.029 [+0.016, +0.042] ✔ |

### Per family, TPR@1% (5-seed mean)

| variant | BruteForce | Flood | Mirai | Recon | Spoofing | Web |
|---|---|---|---|---|---|---|
| rehearsed | 0.284 | 0.992 | 1.000 | 0.260 | 0.160 | 0.130 |
| fixed 0.90 | 0.264 | 0.992 | 1.000 | 0.257 | 0.164 | 0.155 |
| equal (0.60) | 0.283 | 0.995 | 1.000 | 0.255 | 0.148 | 0.137 |
| without rf | 0.291 | 0.994 | 1.000 | 0.264 | 0.171 | 0.160 |
| without xgb | 0.272 | 0.991 | 1.000 | 0.250 | 0.124 | 0.113 |
| without mlp_ce | 0.267 | 0.771 | 0.999 | 0.254 | 0.157 | 0.122 |
| without iforest | 0.284 | 0.993 | 1.000 | 0.260 | 0.164 | 0.140 |
| without clust | 0.233 | 0.992 | 1.000 | 0.229 | 0.161 | 0.124 |
| MCDE + MLP-CE (w 0.9) | 0.302 | 0.999 | 1.000 | 0.254 | 0.179 | 0.173 |
| xgb | 0.196 | 0.677 | 0.973 | 0.193 | 0.177 | 0.152 |
| mlp_ce | 0.262 | 0.998 | 1.000 | 0.223 | 0.087 | 0.163 |
| mcde | 0.279 | 0.909 | 1.000 | 0.242 | 0.181 | 0.155 |

re-rehearsed alpha per reduced set (mean over seeds and folds): without rf 0.82, without xgb 0.73, without mlp_ce 0.71, without iforest 0.87, without clust 0.85

realised benign-test FPR at 1% (mean over seeds and folds): rehearsed 0.0099, fixed 0.90 0.0099, equal (0.60) 0.0097, without rf 0.0099, without xgb 0.0099, without mlp_ce 0.0098, without iforest 0.0099, without clust 0.0099, MCDE + MLP-CE (w 0.9) 0.0099, xgb 0.0096, mlp_ce 0.0104, mcde 0.0097

## (ii) H5 with each family dropped in turn (rehearsed fusion; 5-seed t-CI)

| dropped family | minus XGB | minus MCDE | minus MLP-CE |
|---|---|---|---|
| (none) | 0.076 [0.058, 0.094] | 0.010 [0.002, 0.018] | 0.015 [0.002, 0.029] |
| BruteForce | 0.074 [0.051, 0.096] | 0.011 [-0.000, 0.022] | 0.014 [0.005, 0.024] |
| Flood | 0.029 [0.006, 0.051] | -0.004 [-0.008, -0.001] | 0.020 [0.003, 0.037] |
| Mirai | 0.086 [0.057, 0.115] | 0.012 [0.002, 0.022] | 0.019 [0.002, 0.035] |
| Recon | 0.078 [0.061, 0.096] | 0.008 [-0.002, 0.019] | 0.011 [-0.002, 0.025] |
| Spoofing | 0.095 [0.073, 0.118] | 0.016 [0.008, 0.025] | 0.004 [-0.017, 0.025] |
| Web | 0.096 [0.074, 0.118] | 0.017 [0.006, 0.028] | 0.025 [0.016, 0.034] |
