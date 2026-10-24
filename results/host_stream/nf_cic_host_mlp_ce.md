# Per-host conformal CUSUM kill test — nf_cic host, mlp_ce, seed 0

Host-day TPR (share of attacker host-days of the held-out family with ≥ 1 alarm on a held-out flow) at a 1% benign host-day alarm budget calibrated on disjoint hosts. Clean = clean-history subset.

| family | hosts | host-days | clean | P | E | W | A | A clean | P clean | A-V | flow TPR A / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 8.0 | 8.0 | 3.0 | 0.750 | 0.875 | 0.750 | 0.875 | 0.667 | 0.667 | 0.875 | 0.572 / 0.046 |
| BruteForce | 1.0 | 1.0 | 1.0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.917 / 0.000 |
| DDoS | 10.0 | 20.0 | 20.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 / 0.996 |
| DoS | 3.0 | 3.0 | 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 / 0.994 |
| Infiltration | 125.0 | 149.0 | 149.0 | 0.027 | 0.054 | 0.020 | 0.060 | 0.060 | 0.027 | 0.148 | 0.178 / 0.007 |
| Web | 53.0 | 53.0 | 52.0 | 0.038 | 0.038 | 0.038 | 0.019 | 0.000 | 0.019 | 0.038 | 0.086 / 0.086 |

macro: P 0.469, E 0.494, W 0.468, A 0.659
realised test benign host-day alarm rate: P 0.0140, E 0.0148, W 0.0128, A 0.0149, AV 0.0547; A-V per-flow benign alarm rate 0.0759
A − P: macro +0.190 [95% host-bootstrap CI +0.162, +0.234]
A − E: macro +0.165 [95% host-bootstrap CI +0.157, +0.169]
gates vs P: G1 PASS, G2 FAIL, G3 PASS, G4 PASS
gates vs E: G1 PASS, G2 FAIL, G3 PASS, G4 PASS
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
