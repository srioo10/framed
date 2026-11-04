# Per-host conformal CUSUM kill test — nf_cic host, xgb, seed 0

Host-day TPR (share of attacker host-days of the held-out family with ≥ 1 alarm on a held-out flow) at a 1% benign host-day alarm budget calibrated on disjoint hosts. Clean = clean-history subset.

| family | hosts | host-days | clean | P | E | W | A | A clean | P clean | A-V | flow TPR A / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 8.0 | 8.0 | 3.0 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.802 / 0.206 |
| BruteForce | 1.0 | 1.0 | 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.919 / 0.000 |
| DDoS | 10.0 | 20.0 | 20.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 / 0.515 |
| DoS | 3.0 | 3.0 | 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 / 0.945 |
| Infiltration | 125.0 | 149.0 | 149.0 | 0.054 | 0.107 | 0.067 | 0.094 | 0.094 | 0.054 | 0.309 | 0.182 / 0.041 |
| Web | 53.0 | 53.0 | 52.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 |

macro: P 0.655, E 0.685, W 0.678, A 0.682
realised test benign host-day alarm rate: P 0.0134, E 0.0140, W 0.0117, A 0.0141, AV 0.0499; A-V per-flow benign alarm rate 0.1006
A − P: macro +0.028 [95% host-bootstrap CI +0.000, +0.073]
A − E: macro -0.002 [95% host-bootstrap CI -0.006, +0.000]
gates vs P: G1 FAIL, G2 FAIL, G3 PASS, G4 PASS
gates vs E: G1 FAIL, G2 FAIL, G3 PASS, G4 PASS
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
