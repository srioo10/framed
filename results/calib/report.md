# Few-host calibration (PREREG §9, CAL)

## nf_cic

| m | method | benign host-day alarm rate | macro host-day TPR |
|---|---|---|---|
| 20 | B | 0.0539 | 0.945 |
| 20 | E | 0.0580 | 0.935 |
| 20 | M | 0.0326 | 0.913 |
| 20 | V | 0.0741 | 0.934 |
| 5 | B | 0.0833 | 0.489 |
| 5 | E | 0.0892 | 0.490 |
| 5 | M | 0.0465 | 0.471 |
| 5 | V | 0.0156 | 0.420 |
| 50 | B | 0.0415 | 0.971 |
| 50 | E | 0.0289 | 0.934 |
| 50 | M | 0.0205 | 0.930 |
| 50 | V | 0.0800 | 0.968 |
| all | Full | 0.0129 | 0.995 |

M not calibratable (no root) in 0.0% of draws

## nf_ton

| m | method | benign host-day alarm rate | macro host-day TPR |
|---|---|---|---|
| 20 | B | 0.0230 | 0.256 |
| 20 | E | 0.0352 | 0.261 |
| 20 | M | 0.0264 | 0.215 |
| 20 | V | 0.1868 | 0.355 |
| 5 | B | 0.0121 | 0.080 |
| 5 | E | 0.0132 | 0.078 |
| 5 | M | 0.0102 | 0.070 |
| 5 | V | 0.0861 | 0.108 |
| 50 | B | 0.0199 | 0.326 |
| 50 | E | 0.0200 | 0.299 |
| 50 | M | 0.0093 | 0.240 |
| 50 | V | 0.2904 | 0.522 |
| all | Full | 0.0072 | 0.290 |

M not calibratable (no root) in 0.5% of draws

**CAL primary (M) FAIL**: rate <= 0.015 at m = 5, 20 on both datasets no; nf_cic TPR >= Full - 0.05 (0.945) no
