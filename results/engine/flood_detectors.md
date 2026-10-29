# Spoof-induced false alarms across detectors (PREREG §9 ENG-X + amendment)

Python twin of the engine flood (`scripts/flood_detectors.py`): same flood streams as §9 ENG (IoT-23 parity stream,
seeded), exact host context (unbounded state), detectors of nf_iot23 s1 held Okiru, P / A at 1% from calibration hosts.
215 benign host-days. One seed, one config, one stream: an existence result, not a rate.

**Run 1: harness check failed** (retrained mlp_ce at k = 1, N = 10: A 0.9%, vs engine 99.1%), so by the rule no claim.
**Amendment (pre-registered, commit 5c09b74):** `engine_w` = exact engine network, reference and thresholds in Python.
**Run 2: harness check passes.** engine_w reproduces the engine (A 1.4 / 3.7 / 99.1% at k = 1, N = 0 / 1 / 10; engine
1.4 / 4.2 / 99.1%; k = 10, N = 10: 7.4% vs 6.5%). Retrained detectors are deterministic: run 2 = run 1.

Benign host-days alarmed (%), k = 1:

| detector | A, N = 0 / 1 / 10 | P, N = 0 / 1 / 10 | claim rule (≥ 10% and ≥ 5× N = 0 at N = 10), A / P |
|---|---|---|---|
| engine_w (int8 MLP) | 1.4 / 3.7 / 99.1 | 0.9 / 0.9 / 98.1 | affected / affected |
| mlp_ce (retrained) | 0.0 / 0.0 / 0.9 | 0.9 / 0.9 / 98.1 | not / affected |
| mlp_sn_gda (DDU) | 0.0 / 0.0 / 0.5 | 0.5 / 98.1 / 98.1 | not / affected |
| kitnet | 0.0 / 0.0 / 0.0 | 0.9 / 97.7 / 97.7 | not / affected |
| rf | 1.4 / 1.4 / 1.4 | 0.0 / 0.0 / 0.5 | not / not |
| xgb | 0.5 / 0.5 / 0.5 | 0.9 / 0.9 / 1.4 | not / not |
| iforest | 0.9 / 0.9 / 0.9 | 0.5 / 0.5 / 0.9 | not / not |

k = 10 (dilated Mirai, so spoof bursts spread out): DDU P 81.9%, KitNET P 83.7%, all others ≤ 7.4%.

**Reading.** Spoofing breaks the neural and KitNET detectors and leaves the tree models and iForest untouched. A on
the retrained networks did not alarm, but A on the engine network did, so A's resistance is not structural.
Hypothesis, not tested: Prep does not clamp features (by design: extreme values of an unseen family are signal), so
networks extrapolate on the inflated destination counts; trees and iForest saturate outside the training range.
Attacker host-day detection is unchanged or higher under the flood for every detector.
