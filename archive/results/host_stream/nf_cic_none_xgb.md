# Per-host conformal CUSUM kill test — nf_cic none, xgb, seed 0

Host-day TPR (share of attacker host-days of the held-out family with ≥ 1 alarm on a held-out flow) at a 1% benign host-day alarm budget calibrated on disjoint hosts. Clean = clean-history subset.

| family | hosts | host-days | clean | P | E | W | A | A clean | P clean | A-V | flow TPR A / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 8.0 | 8.0 | 3.0 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.606 / 0.141 |
| BruteForce | 1.0 | 1.0 | 1.0 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 / 0.000 |
| DDoS | 10.0 | 20.0 | 20.0 | 0.000 | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 | 0.500 | 0.439 / 0.000 |
| DoS | 3.0 | 3.0 | 3.0 | 0.000 | 0.333 | 0.000 | 0.333 | 0.333 | 0.000 | 0.333 | 0.001 / 0.000 |
| Infiltration | 125.0 | 149.0 | 149.0 | 0.201 | 0.067 | 0.174 | 0.060 | 0.060 | 0.201 | 0.295 | 0.178 / 0.009 |
| Web | 53.0 | 53.0 | 52.0 | 0.226 | 0.094 | 0.434 | 0.132 | 0.135 | 0.231 | 0.604 | 0.311 / 0.285 |

macro: P 0.238, E 0.499, W 0.101, A 0.504
realised test benign host-day alarm rate: P 0.0118, E 0.0110, W 0.0112, A 0.0108, AV 0.0523; A-V per-flow benign alarm rate 0.0315
A − P: macro +0.266 [95% host-bootstrap CI +0.194, +0.374]
A − E: macro +0.005 [95% host-bootstrap CI -0.002, +0.016]
gates vs P: G1 PASS, G2 FAIL, G3 FAIL, G4 PASS
gates vs E: G1 FAIL, G2 FAIL, G3 PASS, G4 PASS
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
