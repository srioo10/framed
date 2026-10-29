# Denial-of-state flood (PREREG §9 ENG), engine commit a9a2ea6

Stream: IoT-23 parity stream (c7, c34, ch4), model nf_iot23 s1 held Okiru. N spoofed one-flow sources before every real
flow; Mirai flow times dilated by k. Attacker host-days grow with k (3 / 10 / 64) because dilation spreads them over days.
Raw rows: results/engine/flood.csv.

**Claim rule result: holds.** Every attacker host-day that alarms at N = 0 also alarms at N = 100, for every table size
(512 / 4,096 / 65,536) and k (1 / 10 / 100). Limit: Mirai scores high on almost every flow, so losing host state
costs little; first-alarm delay grows (k = 100, N = 100: median 4 flows, max 40, vs 0 / 23 clean). An existence result
on 3-64 host-days, not a rate.

**Unplanned finding (not under a claim rule): spoofing induces false alarms on benign hosts.** Spoofed flows to benign
destinations inflate destination-side host context, so benign clients of those destinations score high:

| k | N | benign host-days alarmed, A (T = 512 / 4,096 / 65,536) | P (same) |
|---|---|---|---|
| 1 | 0 | 1.4% / 1.4% / 1.4% | 0.9% / 0.9% / 0.9% |
| 1 | 1 | 4.2% / 4.2% / 4.2% | 0.9% / 0.9% / 0.9% |
| 1 | 10 | 99.1% / 99.1% / 99.1% | 98.1% / 98.1% / 98.1% |
| 1 | 100 | 16.3% / 99.1% / 99.1% | 3.3% / 98.6% / 98.6% |
| 10 | 10 | 6.5% / 6.5% / 6.5% | 0.9% / 0.9% / 0.9% |
| 10 | 100 | 6.5% / 82.3% / 82.3% | 0.9% / 80.0% / 80.0% |
| 100 | 100 | 1.4% / 1.4% / 1.4% | 0.9% / 0.9% / 0.9% |

Spoofed flows reuse real flow times, so their density per window follows the (dilated) Mirai burst; the induced alarms
fall as k spreads them. The smallest table evicts the inflated destination entries and so alarms less. With the
mitigation hook on, each induced alarm blocks a benign host: the flood turns the IDS into a denial-of-service tool.
215 benign host-days; the same features drive the Python banks, so this is a property of the host-context design, not
of the C port.
