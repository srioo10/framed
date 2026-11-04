# Results index

Each claim in the paper, the files that hold its numbers, and the `PREREG.md` row that fixed its rule before the run.
Files are in `results/cert/` unless stated. `{ds}` is `nf_ctu13`, `nf_ugr16`, `nf_cic` or `nf_ton`.

## Claim 1: host-context poisoning turns ML IPSs against innocent hosts

| Result | Files | PREREG row |
|---|---|---|
| Spoofed one-way injection: RF / XGBoost and published detectors block 88–100% of innocent host-days (CTU-13, UGR'16) | `k1_{ds}.csv`, `k1_pub_{ds}.csv` | K1, K14 |
| Shared-key (established, no spoofing) injection, and a real bot's flows re-keyed onto victims | `gate_k4_{ds}_day_sharedkey.csv` | SHARED-KEY |
| White-box per-flow optimiser | `opt_harm_{ds}_opt.csv` | OPT (M6) |
| Unmodified Kitsune on its own Mirai capture: spoofed, flow-sized, shared-key | `kitsune_native_{mirai,flow,sharedkey}.json` | K-NATIVE, K-NATIVE-FLOW, K-NATIVE-SK |
| Per-flow share of victims' flows dropped by plain RF / XGBoost | `gate_k4_{ds}_day.csv` | SMALL-4 |
| UNSW-NB15: attack below the bar (failure) | `k1_unsw_raw.csv` | AUDIT F4 |

## Claim 2: Veto blocks safely

| Result | Files | PREREG row |
|---|---|---|
| 0 injection-caused blocks over 72 patterns + adaptive attacker; harm at clean level | `gate_k4_{ds}_day.csv` | K4, HOSTDAY, CONF C3 |
| 95% intervals on harm and on the zero-violation result | `harm_ci.json` (`scripts/harm_ci.py`) | SMALL-2 |
| Over-budget harm, and certifying a larger m (Table IV) | `gate_k4_{ds}_day_k19.csv`, `..._mcert20.csv`, `..._mcert50.csv` | M-CERT |
| Low-rate (throttled) attacker | `gate_throttle_{ds}_k19.csv` | K19 |
| Padding the attacker's own context, and the flow-only fix | `dilute_{ds}_opt.csv`, `dilute_{ds}_dilfix.csv` | DILUTE, DILUTE-FIX |
| Bounded-memory engine under eviction floods | `engine_k11_nf_ugr16.csv` | K11 |
| Adversarial training and randomized smoothing baselines | `gate_k4_{ds}_day_k18.csv`, `gate_throttle_{ds}_rs.csv` | K18, RS |

## Claim 3: Veto still catches attacks while blocking safely

| Result | Files | PREREG row |
|---|---|---|
| Certified detection vs certified trees, +0.104 (proof rule), +0.030 (K5 convention) | `conf_k5b_*`, `conf_trees_*`, `conf_groot_*`, `results/conf/` | CONF C2, K5, K5b |
| Size-matched certified tree (23 KB XGBoost 0.654 vs Veto 0.785); model sizes | `k5_tiny.csv`, `model_sizes.json` | SMALL-1 |
| IPS operating point under attack | `ips_tpr_{ds}.csv` | K15 |
| Ingredient ablation (Fig. 4: no tree teachers, no deletion-box loss, no insertion-box loss) | `gate_throttle_{ds}_abl_no{T,R,C}.csv` | ABL |
| Host-context ablation (unknown TPR 0.886 → 0.534) | see the row | LOAD-BEARING |
| Failures: no speed advantage (K8), more width does not help (K10) | `k8_ops.csv`, `results/bowl/k10_*.json` | K8, K10 |

## Claim 4: Veto runs on a $5 gateway

| Result | Files | PREREG row |
|---|---|---|
| C engine parity with the Python bound | `results/engine/` | engine parity |
| Live ESP32 tests (sensor kept online within budget, blocked over budget; cost; multi-device benign) | `results/live/` | LIVE E1–E5, live #4 |

The earlier ARLHNIDS / OnePass-IDS results are indexed in `archive/RESULTS_onepass.md`.
