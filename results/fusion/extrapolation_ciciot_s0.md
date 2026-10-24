ciciot_csv held-out Flood: 20000 test rows, 182667 training rows

| feature | KS | share outside train range |
|---|---|---|
| Tot size | 0.785 | 0.000% |
| AVG | 0.785 | 0.000% |
| IAT | 0.690 | 0.000% |
| Rate | 0.682 | 0.000% |
| Std | 0.626 | 0.000% |
| Variance | 0.626 | 0.000% |
| Max | 0.621 | 0.000% |
| HTTPS | 0.563 | 0.000% |
| ack_flag_number | 0.562 | 0.000% |
| Tot sum | 0.561 | 0.000% |

any of top 10 outside range: 0.000%; any feature: 32.585%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 20000 | 0.895 / 0.345 | 0.970 / 0.846 | 0.960 / 0.858 | 0.998 / 0.996 | 0.990 / 0.892 |

score at the member's maximum (saturation): rf 0.1%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 9.1%, mlp_kd 0.0%
| feature | KS | share outside train range |
|---|---|---|
| ack_flag_number | 0.842 | 0.000% |
| IAT | 0.772 | 0.000% |
| Rate | 0.755 | 0.000% |
| HTTPS | 0.752 | 0.000% |
| Tot sum | 0.719 | 3.628% |
| ack_count | 0.716 | 9.072% |
| Header_Length | 0.664 | 0.000% |
| Time_To_Live | 0.658 | 0.000% |
| TCP | 0.634 | 0.000% |
| psh_flag_number | 0.633 | 0.000% |

any of top 10 outside range: 11.447%; any feature: 59.208%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 35421 | 0.906 / 0.173 | 0.923 / 0.431 | 0.720 / 0.251 | 0.985 / 0.934 | 0.991 / 0.925 |
| 0-0.5sd | 54 | 0.910 / 0.500 | 0.954 / 0.685 | 0.922 / 0.704 | 0.997 / 0.981 | 0.999 / 0.981 |
| 0.5-2sd | 169 | 0.903 / 0.527 | 0.921 / 0.621 | 0.888 / 0.645 | 0.995 / 0.959 | 0.996 / 0.976 |
| 2-10sd | 1431 | 0.866 / 0.224 | 0.864 / 0.324 | 0.770 / 0.294 | 0.995 / 0.978 | 0.995 / 0.984 |
| >10sd | 2925 | 0.882 / 0.159 | 0.877 / 0.336 | 0.862 / 0.549 | 0.983 / 0.945 | 0.992 / 0.974 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 0.7%, mlp_kd 1.3%
