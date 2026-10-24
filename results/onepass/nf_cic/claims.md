# OnePass-IDS claims — nf_cic, seeds [1, 2, 3, 4]

| method | Bot | BruteForce | DDoS | DoS | Infiltration | Web | macro |
|---|---|---|---|---|---|---|---|
| attacker hosts / host-days | 8 / 32 | 1 / 4 | 10 / 80 | 3 / 8 | 260 / 525 | 126 / 210 | |
| OnePass (host, mlp_ce, A) | 0.406 | 0.750 | 1.000 | 1.000 | 0.069 | 0.010 | 0.539 |
| per-flow MLP (none, P) | 0.219 | 0.000 | 0.613 | 0.250 | 0.069 | 0.081 | 0.205 |
| rf (none, P) | 0.000 | 0.000 | 0.000 | 0.000 | 0.242 | 0.290 | 0.089 |
| xgb (none, P) | 0.594 | 0.000 | 0.325 | 0.375 | 0.225 | 0.333 | 0.309 |
| mcde (none, P) | 0.031 | 0.250 | 0.988 | 0.500 | 0.152 | 0.076 | 0.333 |
| rf (host, A) upper ref | 0.969 | 1.000 | 1.000 | 1.000 | 0.128 | 0.005 | 0.684 |
| xgb (host, A) upper ref | 0.938 | 1.000 | 1.000 | 1.000 | 0.095 | 0.000 | 0.672 |
| mlp_ce_int8 (host, A) | 0.406 | 0.750 | 1.000 | 1.000 | 0.065 | 0.010 | 0.538 |
| mlp_sn_gda (host, A) | 0.969 | 1.000 | 1.000 | 1.000 | 0.051 | 0.071 | 0.682 |
| host_perm (mlp_ce, A) | 0.344 | 0.000 | 0.750 | 0.375 | 0.109 | 0.071 | 0.275 |

C1 OnePass − per-flow MLP: +0.334 [+0.256, +0.405] → PASS
C2 OnePass − best large per-flow baseline (mcde): +0.206 [+0.166, +0.280] → PASS
C3 host gain +0.102, perm gain -0.162 → PASS
C4 realised benign host-day alarm rate 0.0112 → PASS
C5 int8 loss +0.001 → PASS
C6 DDU − OnePass +0.143 [+0.110, +0.169] → PASS
C7 back-off accuracy (leaf case: ['BruteForce', 'DDoS', 'DoS', 'Web']) 0.508, chance 0.250 → FAIL
   per family: BruteForce 0.000 (n 52972), DDoS 0.897 (n 79495), DoS 0.415 (n 42750), Web 0.000 (n 35)
underpowered (< 5 attacker hosts): ['BruteForce', 'DoS']
