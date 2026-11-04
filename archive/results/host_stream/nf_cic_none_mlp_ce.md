# Per-host conformal CUSUM kill test — nf_cic none, mlp_ce, seed 0

Host-day TPR (share of attacker host-days of the held-out family with ≥ 1 alarm on a held-out flow) at a 1% benign host-day alarm budget calibrated on disjoint hosts. Clean = clean-history subset.

| family | hosts | host-days | clean | P | E | W | A | A clean | P clean | A-V | flow TPR A / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 8.0 | 8.0 | 3.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 |
| BruteForce | 1.0 | 1.0 | 1.0 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 / 0.000 |
| DDoS | 10.0 | 20.0 | 20.0 | 0.500 | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 | 1.000 | 0.487 / 0.004 |
| DoS | 3.0 | 3.0 | 3.0 | 0.333 | 0.333 | 0.333 | 0.667 | 0.667 | 0.333 | 0.667 | 0.230 / 0.009 |
| Infiltration | 125.0 | 149.0 | 149.0 | 0.054 | 0.081 | 0.081 | 0.081 | 0.081 | 0.054 | 0.289 | 0.173 / 0.008 |
| Web | 53.0 | 53.0 | 52.0 | 0.057 | 0.038 | 0.151 | 0.038 | 0.038 | 0.058 | 0.226 | 0.245 / 0.172 |

macro: P 0.157, E 0.409, W 0.344, A 0.464
realised test benign host-day alarm rate: P 0.0118, E 0.0127, W 0.0099, A 0.0129, AV 0.0601; A-V per-flow benign alarm rate 0.0281
A − P: macro +0.307 [95% host-bootstrap CI +0.243, +0.416]
A − E: macro +0.056 [95% host-bootstrap CI +0.000, +0.167]
gates vs P: G1 PASS, G2 FAIL, G3 PASS, G4 PASS
gates vs E: G1 FAIL, G2 FAIL, G3 PASS, G4 PASS
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
