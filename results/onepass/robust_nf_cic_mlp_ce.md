# Host-level budget robustness — nf_cic, mlp_ce, seeds [1, 2, 3, 4]

benign test host-days per fold (mean): 19261; alarming: 216.3 (0.0112)
benign test hosts per fold (mean): 14568; hosts with >= 1 alarm: 127.2
max alarming host-days on one host (max over folds): 10
share of alarm host-days from the top 10% alarming hosts (mean): 0.265
alarm host-days per alarming host, distribution: {1: 1943, 2: 579, 3: 253, 4: 145, 5: 87, 6: 25, 7: 11, 8: 5, 9: 1, 10: 4}
lag-1 autocorrelation of benign conformal p-values within host-day streams (mean): 0.652
Ville rule realised per-flow benign alarm rate: 0.0720 (exchangeable target <= 0.01)
