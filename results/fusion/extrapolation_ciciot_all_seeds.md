# CICIoT Flood extrapolation check, 5 seeds (PREREG §9, 2026-10-24 row)

**Negative result about the mechanism.** On every seed, at most 0.025% of held-out Flood test rows lie outside the
training range on the 10 most-shifted features (KS), and inside that range MLP-CE catches 0.994–1.000 at 1% FPR vs
XGB 0.56–0.85 and RF 0.30–0.39. MLP-CE's Flood advantage is therefore not explained by extrapolation. No other
mechanism is claimed (untested). With Flood + Mirai held out jointly, ~11% of rows are outside the range; MLP-CE
still leads inside the range (0.91–0.94 vs XGB 0.43–0.93).

| seed | Flood rows outside top-10 range | MLP-CE TPR@1% inside | XGB inside | RF inside | Flood+Mirai outside | MLP-CE inside (F+M) | XGB inside (F+M) |
|---|---|---|---|---|---|---|---|
| 0 | 0.000% | 0.996 | 0.846 | 0.345 | 11.4% | 0.934 | 0.431 |
| 1 | 0.025% | 1.000 | 0.665 | 0.385 | 11.1% | 0.914 | 0.702 |
| 2 | 0.025% | 1.000 | 0.557 | 0.297 | 11.4% | 0.931 | 0.563 |
| 3 | 0.015% | 0.994 | 0.607 | 0.329 | 11.5% | 0.937 | 0.573 |
| 4 | 0.005% | 0.999 | 0.712 | 0.372 | 11.5% | 0.937 | 0.931 |

Full per-seed output (KS features, distance bins, saturation) below.

```
### seed 0, held-out Flood
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

### seed 0, held-out Flood__Mirai
ciciot_csv held-out Flood__Mirai: 40000 test rows, 142667 training rows

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

### seed 1, held-out Flood
ciciot_csv held-out Flood: 20000 test rows, 182687 training rows

| feature | KS | share outside train range |
|---|---|---|
| Tot size | 0.806 | 0.000% |
| AVG | 0.806 | 0.000% |
| IAT | 0.692 | 0.000% |
| Rate | 0.684 | 0.000% |
| Std | 0.642 | 0.000% |
| Variance | 0.642 | 0.000% |
| Max | 0.635 | 0.000% |
| ack_flag_number | 0.566 | 0.000% |
| Tot sum | 0.559 | 0.025% |
| HTTPS | 0.559 | 0.000% |

any of top 10 outside range: 0.025%; any feature: 47.415%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 19995 | 0.922 / 0.385 | 0.963 / 0.665 | 0.929 / 0.681 | 1.000 / 1.000 | 0.973 / 0.611 |
| 0-0.5sd | 1 | 0.795 / 0.000 | 0.989 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 0.985 / 1.000 |
| 0.5-2sd | 3 | 0.817 / 0.000 | 0.151 / 0.000 | 0.150 / 0.000 | 1.000 / 1.000 | 0.988 / 0.667 |
| 2-10sd | 1 | 0.841 / 0.000 | 0.341 / 0.000 | 0.121 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 |

score at the member's maximum (saturation): rf 0.1%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 81.6%, mlp_kd 0.0%

### seed 1, held-out Flood__Mirai
ciciot_csv held-out Flood__Mirai: 40000 test rows, 142687 training rows

| feature | KS | share outside train range |
|---|---|---|
| ack_flag_number | 0.845 | 0.000% |
| IAT | 0.774 | 0.000% |
| HTTPS | 0.757 | 0.000% |
| Rate | 0.757 | 0.000% |
| ack_count | 0.719 | 8.742% |
| Tot sum | 0.717 | 3.415% |
| Header_Length | 0.666 | 0.000% |
| Time_To_Live | 0.664 | 0.000% |
| psh_flag_number | 0.642 | 0.000% |
| TCP | 0.633 | 0.000% |

any of top 10 outside range: 11.085%; any feature: 65.680%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 35566 | 0.938 / 0.596 | 0.955 / 0.702 | 0.940 / 0.712 | 0.983 / 0.914 | 0.987 / 0.925 |
| 0-0.5sd | 45 | 0.924 / 0.556 | 0.969 / 0.733 | 0.966 / 0.733 | 0.993 / 0.933 | 0.997 / 0.978 |
| 0.5-2sd | 189 | 0.903 / 0.450 | 0.950 / 0.661 | 0.939 / 0.667 | 0.992 / 0.947 | 0.994 / 0.958 |
| 2-10sd | 1320 | 0.862 / 0.195 | 0.947 / 0.569 | 0.938 / 0.666 | 0.996 / 0.975 | 0.998 / 0.978 |
| >10sd | 2880 | 0.898 / 0.108 | 0.939 / 0.564 | 0.886 / 0.301 | 0.994 / 0.976 | 0.995 / 0.982 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 1.1%, mlp_kd 1.3%

### seed 2, held-out Flood
ciciot_csv held-out Flood: 20000 test rows, 182527 training rows

| feature | KS | share outside train range |
|---|---|---|
| Tot size | 0.795 | 0.000% |
| AVG | 0.795 | 0.000% |
| IAT | 0.691 | 0.000% |
| Rate | 0.683 | 0.000% |
| Std | 0.634 | 0.000% |
| Variance | 0.634 | 0.000% |
| Max | 0.624 | 0.000% |
| ack_flag_number | 0.563 | 0.000% |
| Tot sum | 0.562 | 0.025% |
| HTTPS | 0.560 | 0.000% |

any of top 10 outside range: 0.025%; any feature: 33.980%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 19995 | 0.877 / 0.297 | 0.941 / 0.557 | 0.946 / 0.727 | 0.999 / 1.000 | 0.983 / 0.785 |
| 0-0.5sd | 1 | 0.853 / 0.000 | 0.999 / 1.000 | 0.996 / 1.000 | 1.000 / 1.000 | 0.888 / 0.000 |
| 0.5-2sd | 1 | 0.909 / 0.000 | 0.897 / 0.000 | 0.988 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 2-10sd | 3 | 0.662 / 0.000 | 0.196 / 0.000 | 0.360 / 0.000 | 0.999 / 1.000 | 0.628 / 0.000 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 4.8%, mlp_kd 0.0%

### seed 2, held-out Flood__Mirai
ciciot_csv held-out Flood__Mirai: 40000 test rows, 142527 training rows

| feature | KS | share outside train range |
|---|---|---|
| ack_flag_number | 0.843 | 0.000% |
| IAT | 0.776 | 0.000% |
| Rate | 0.759 | 0.000% |
| HTTPS | 0.756 | 0.000% |
| Tot sum | 0.721 | 3.547% |
| ack_count | 0.718 | 8.992% |
| Header_Length | 0.665 | 0.000% |
| Time_To_Live | 0.651 | 0.000% |
| psh_flag_number | 0.635 | 0.000% |
| TCP | 0.632 | 0.000% |

any of top 10 outside range: 11.370%; any feature: 59.267%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 35452 | 0.909 / 0.285 | 0.958 / 0.563 | 0.953 / 0.554 | 0.972 / 0.931 | 0.985 / 0.916 |
| 0-0.5sd | 39 | 0.905 / 0.462 | 0.848 / 0.487 | 0.916 / 0.564 | 0.984 / 0.923 | 0.988 / 0.872 |
| 0.5-2sd | 172 | 0.915 / 0.523 | 0.919 / 0.686 | 0.934 / 0.686 | 0.989 / 0.942 | 0.992 / 0.942 |
| 2-10sd | 1385 | 0.856 / 0.204 | 0.815 / 0.264 | 0.909 / 0.315 | 0.994 / 0.960 | 0.995 / 0.973 |
| >10sd | 2952 | 0.888 / 0.063 | 0.945 / 0.724 | 0.916 / 0.214 | 0.978 / 0.892 | 0.984 / 0.893 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 15.0%, mlp_kd 0.1%

### seed 3, held-out Flood
ciciot_csv held-out Flood: 20000 test rows, 182544 training rows

| feature | KS | share outside train range |
|---|---|---|
| Tot size | 0.788 | 0.000% |
| AVG | 0.788 | 0.000% |
| IAT | 0.685 | 0.000% |
| Rate | 0.677 | 0.000% |
| Std | 0.622 | 0.000% |
| Variance | 0.622 | 0.000% |
| Max | 0.618 | 0.000% |
| HTTPS | 0.569 | 0.000% |
| ack_flag_number | 0.563 | 0.000% |
| Tot sum | 0.558 | 0.015% |

any of top 10 outside range: 0.015%; any feature: 32.840%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 19997 | 0.924 / 0.329 | 0.941 / 0.607 | 0.963 / 0.760 | 0.999 / 0.994 | 0.977 / 0.750 |
| 0.5-2sd | 2 | 0.805 / 0.000 | 0.379 / 0.000 | 0.342 / 0.000 | 1.000 / 1.000 | 0.998 / 1.000 |
| 2-10sd | 1 | 0.838 / 0.000 | 0.277 / 0.000 | 0.283 / 0.000 | 1.000 / 1.000 | 0.980 / 1.000 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 60.6%, mlp_kd 0.0%

### seed 3, held-out Flood__Mirai
ciciot_csv held-out Flood__Mirai: 40000 test rows, 142544 training rows

| feature | KS | share outside train range |
|---|---|---|
| ack_flag_number | 0.842 | 0.000% |
| IAT | 0.760 | 0.000% |
| HTTPS | 0.752 | 0.000% |
| Rate | 0.743 | 0.000% |
| Tot sum | 0.716 | 3.363% |
| ack_count | 0.713 | 9.235% |
| Header_Length | 0.673 | 0.000% |
| Time_To_Live | 0.657 | 0.000% |
| psh_flag_number | 0.635 | 0.000% |
| TCP | 0.633 | 0.000% |

any of top 10 outside range: 11.525%; any feature: 59.247%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 35390 | 0.937 / 0.663 | 0.917 / 0.573 | 0.941 / 0.620 | 0.986 / 0.937 | 0.998 / 0.995 |
| 0-0.5sd | 55 | 0.880 / 0.400 | 0.860 / 0.382 | 0.799 / 0.400 | 0.996 / 0.964 | 0.987 / 0.945 |
| 0.5-2sd | 237 | 0.898 / 0.519 | 0.884 / 0.544 | 0.855 / 0.553 | 0.997 / 0.983 | 0.995 / 0.979 |
| 2-10sd | 1368 | 0.857 / 0.268 | 0.761 / 0.298 | 0.744 / 0.297 | 0.999 / 0.997 | 0.990 / 0.923 |
| >10sd | 2950 | 0.900 / 0.381 | 0.910 / 0.393 | 0.933 / 0.779 | 0.996 / 0.979 | 0.988 / 0.972 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 12.4%, mlp_kd 2.1%

### seed 4, held-out Flood
ciciot_csv held-out Flood: 20000 test rows, 182539 training rows

| feature | KS | share outside train range |
|---|---|---|
| Tot size | 0.790 | 0.000% |
| AVG | 0.790 | 0.000% |
| IAT | 0.689 | 0.000% |
| Rate | 0.681 | 0.000% |
| Std | 0.632 | 0.000% |
| Variance | 0.632 | 0.000% |
| Max | 0.627 | 0.000% |
| HTTPS | 0.567 | 0.000% |
| ack_flag_number | 0.566 | 0.000% |
| Tot sum | 0.561 | 0.005% |

any of top 10 outside range: 0.005%; any feature: 33.480%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 19999 | 0.897 / 0.372 | 0.932 / 0.712 | 0.934 / 0.754 | 0.999 / 0.999 | 0.953 / 0.751 |
| 0-0.5sd | 1 | 0.772 / 0.000 | 0.899 / 0.000 | 0.948 / 1.000 | 0.977 / 1.000 | 0.933 / 0.000 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 29.9%, mlp_kd 0.9%

### seed 4, held-out Flood__Mirai
ciciot_csv held-out Flood__Mirai: 40000 test rows, 142539 training rows

| feature | KS | share outside train range |
|---|---|---|
| ack_flag_number | 0.845 | 0.000% |
| IAT | 0.772 | 0.000% |
| HTTPS | 0.760 | 0.000% |
| Rate | 0.755 | 0.000% |
| Tot sum | 0.720 | 3.885% |
| ack_count | 0.719 | 8.883% |
| Header_Length | 0.669 | 0.000% |
| Time_To_Live | 0.654 | 0.000% |
| psh_flag_number | 0.638 | 0.000% |
| TCP | 0.633 | 0.000% |

any of top 10 outside range: 11.520%; any feature: 59.370%

| bin | n | rf mean / TPR@1% | xgb mean / TPR@1% | xgb_tail mean / TPR@1% | mlp_ce mean / TPR@1% | mlp_kd mean / TPR@1% |
|---|---|---|---|---|---|---|
| inside | 35392 | 0.923 / 0.691 | 0.982 / 0.931 | 0.945 / 0.712 | 0.992 / 0.937 | 0.988 / 0.934 |
| 0-0.5sd | 130 | 0.913 / 0.608 | 0.933 / 0.638 | 0.874 / 0.515 | 0.998 / 0.992 | 0.997 / 0.985 |
| 0.5-2sd | 145 | 0.905 / 0.552 | 0.921 / 0.614 | 0.909 / 0.614 | 0.995 / 0.972 | 0.996 / 0.972 |
| 2-10sd | 1414 | 0.875 / 0.217 | 0.882 / 0.275 | 0.838 / 0.275 | 0.996 / 0.979 | 0.997 / 0.988 |
| >10sd | 2919 | 0.853 / 0.087 | 0.917 / 0.547 | 0.891 / 0.201 | 0.982 / 0.954 | 0.986 / 0.917 |

score at the member's maximum (saturation): rf 0.0%, xgb 0.0%, xgb_tail 0.0%, mlp_ce 13.6%, mlp_kd 5.3%

```
