# C engine parity and footprint (engine/onepass.c, commit bb5ecc8)

Model: nf_iot23 seed 1, outer config held = Okiru, d-64-32-11 MLP (CE), weights rounded to int8 per tensor (4,128 weights).
Stream: every raw flow of IoT-23 captures c7, c34, ch4 (323,597 flows: 320,799 Mirai, 2,798 Benign), time ordered.
Python reference: exact hostctx windows + float model with the same int8-rounded weights.

| measure | value |
|---|---|
| score abs diff, median / p99 / max | 2.4e-4 / 2.4e-3 / 0.96 |
| flows with abs score diff > 0.05 | 0.04% |
| flows whose P decision differs (score > threshold) | 33 (0.01%) |
| host features, p99 relative error (bucketed windows, 16-slot pair tables) | 0.104 (worst column: hc_src_n_10s) |
| host-days alarmed (A / P), benign, 215 host-days | 1.4% / 0.9% |
| host-days alarmed (A / P), Mirai, 3 host-days | 3 / 3 |

Footprint (laptop, single thread, includes CSV parsing):

| host slots | state | us / flow | evictions |
|---|---|---|---|
| 1,048,576 | 2,281.7 MB | 9.40 | 51 |
| 65,536 | 142.6 MB | 7.37 | 233,924 |
| 4,096 | 8.9 MB | 6.71 | 295,502 |
| 512 | 1.1 MB | 5.68 | 299,343 |

Scores and alarms are identical across table sizes on this stream: evicted entries are single-flow external hosts whose
state is empty again on the next sighting. Per-host state is 2,176 B (bucketed windows 1,232 B, pair tables 896 B),
so an ESP32-S3 with 8 MB PSRAM holds about 3.6k hosts. Not yet measured: denial-of-state flood, device runs.
