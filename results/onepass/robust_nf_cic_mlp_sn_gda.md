# Host-level budget robustness — nf_cic, mlp_sn_gda, seeds [1, 2, 3, 4]

benign test host-days per fold (mean): 19261; alarming: 249.8 (0.0129)
benign test hosts per fold (mean): 14568; hosts with >= 1 alarm: 136.5
max alarming host-days on one host (max over folds): 10
share of alarm host-days from the top 10% alarming hosts (mean): 0.341
alarm host-days per alarming host, distribution: {1: 2203, 2: 502, 3: 273, 4: 87, 5: 25, 6: 42, 7: 29, 8: 45, 9: 10, 10: 59}
lag-1 autocorrelation of benign conformal p-values within host-day streams (mean): 0.534
Ville rule realised per-flow benign alarm rate: 0.0628 (exchangeable target <= 0.01)
