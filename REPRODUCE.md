# Reproducing the paper's numbers

Every headline number traces to a saved result file and the script that wrote it. Rows marked *recompute* run
from saved files on a laptop; rows marked *Kaggle* retrain models (we ran all training on Kaggle; launchers in
`scripts/kaggle_*.py`). Pre-registration rows (`PREREG.md`) name the same files. Run from the repository root with
`PYTHONPATH=src` and `pip install -r requirements.txt`. Datasets: see `DATA.md`.

| Paper claim | Result file(s) | Command | Cost |
|---|---|---|---|
| Ordinary IPS cut the sensor off in 25 of 27 runs; Veto never blocked it in 36; sound build 0 in 13 | `results/live/fair/{fair_runs.txt,sub.log,gw-*.log}` | `python scripts/live/fair_report.py results/live/fair veto` (and `sound`) | recompute, s |
| Every live block decision follows from the logged flow records (12,605 flows incl. LIVE-POS, 0 mismatches) | same logs | `python scripts/live/replay_check.py results/live/fair` | recompute, ~1 min |
| Live positive control: attacker's own address blocked in 5 of 5 runs; sensor, broker, clean runs 0 | `results/live/pos/{fair_runs.txt,sub.log,gw-*.log}` | `python scripts/live/fair_report.py results/live/pos pos`; `python scripts/live/replay_check.py results/live/pos 0.478` | recompute, ~1 min |
| 10 of 10 at the clean 99th percentile; 2 of 10 above 0.477 in replay | `results/live/natenf_runs.csv`, `results/live/natenf_timeline.csv` | `python scripts/live/natenf_report.py runs`, `python scripts/live/timeline.py` | recompute |
| CrowdSec (whitelist removed) banned the shared address in 5 of 5 runs | `results/live/crowdsec/cs_runs.txt` | read directly (R1 rows; R0 = whitelist kept) | — |
| Re-implemented KitNET / HorusEye / Deep SAD alarm on 92–100% of innocent host-days | `results/cert/gate_k4_{ds}.csv` | `python scripts/gate_attack.py {ds} --genome '<Veto genome>'` (genome in `PREREG.md`, CONF rows) | Kaggle |
| Slips blocks 18.7% of innocent host-days on UGR'16 (5.7% clean) | `results/cert/slips_frame_nf_ugr16.json` | `python scripts/slips_frame.py export nf_ugr16 OUT`, run Slips 1.1.24 on each run dir (Kaggle), then `slips_frame.py score nf_ugr16 OUT` | Kaggle, hours |
| Veto 0.824 vs best frame-proof rule 0.309 on CIC | `results/cert/gate_throttle_{ds}_rulesfptpr.csv` | `scripts/gate_attack.py` (RULES-FP row) | Kaggle |
| C1 / C2 / C5 confirmatory claims | `results/conf/*.jsonl` → `results/conf/conf_summary.json` | `python scripts/conf_analyze.py` | recompute |
| Veto reaches ≥ 95.5% of the most permissive rule | `results/cert/opt_gap_{ds}.csv` | `scripts/gate_attack.py --opt-gap` | Kaggle |
| 0 violations under every in-budget attack | `violations` column of `results/cert/gate_k4_*.csv`, `results/conf/*` | as above | — |
| Veto is 3.6 KB int8 | `results/bowl/k10_b.json` | `python scripts/k10_capacity.py --out k10.json` (size from the int8 export) | recompute |
| 65 ms per certified decision (86 ms sound build) | `results/live/natenf_latency.csv`; `us` column of the fair logs | `python scripts/live/latency.py` | recompute |
| Margin build's float64 error < 1e-6 over all gateway inputs | — | `cc -I engine -I firmware/gateway/main engine/test_apriori.c -lm && ./a.out` | s |
| LRU inclusion behind Lemma (deployed) (iii) | — | `engine/test_lru.c`, `engine/test_evict.c`, `engine/test_bloom.c` (build line in each header) | s |

Hardware for the live tests: one Raspberry Pi 4B (access point, broker, attacker station) and two ESP32 boards
(gateway = ESP32-WROOM-32, plus the sensor), no router; topology and procedure in `PREREG.md` (LIVE-NAT-ENF, LIVE-FAIR, LIVE-POS rows). Firmware:
`firmware/gateway` (ESP-IDF; build flags per test in `firmware/gateway/main/CMakeLists.txt`). Lab Wi-Fi and MQTT
credentials are not in the repository (`sdkconfig*` is ignored); set your own with `idf.py menuconfig`.
