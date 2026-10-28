# Discovery harness and blind-spot kill test (PREREG §9 DISC)

## nf_ctu13

Seeds [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; detectors per fold: [np.int64(5)]; attacker host-days (summed over seeds) 167; all-missed 36

| family | attacker host-days | all-missed share |
|---|---|---|
| Menti | 4 | 0.000 |
| Murlo | 8 | 0.000 |
| NSIS | 27 | 0.667 |
| Neris | 48 | 0.000 |
| Rbot | 70 | 0.214 |
| Sogou | 1 | 1.000 |
| Virut | 9 | 0.222 |

**All-missed share (macro over families, mean over seeds): 0.205** (B1 needs >= 0.15)

| detector | method | realised benign host-day FPR | macro host-day TPR |
|---|---|---|---|
| iforest | A | 0.0383 | 0.348 |
| iforest | P | 0.0026 | 0.032 |
| kitnet | A | 0.0187 | 0.267 |
| kitnet | P | 0.0144 | 0.071 |
| mlp_sn_gda | A | 0.0398 | 0.315 |
| mlp_sn_gda | P | 0.0576 | 0.225 |
| rf | A | 0.0106 | 0.546 |
| rf | P | 0.0147 | 0.695 |
| xgb | A | 0.0124 | 0.586 |
| xgb | P | 0.0128 | 0.587 |

| descriptor | host-grouped CV AUC (all-missed vs caught) |
|---|---|
| rate | 0.972 |
| port_entropy | 0.842 |
| iat_cv | 0.970 |
| fanout | 0.935 |
| med_bytes | 0.736 |
| enc_share | 0.873 |
| med_dur | 0.861 |

## nf_iot23

Seeds [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; detectors per fold: [np.int64(5)]; attacker host-days (summed over seeds) 66; all-missed 0

| family | attacker host-days | all-missed share |
|---|---|---|
| Gagfyt | 4 | 0.000 |
| Hajime | 3 | 0.000 |
| Hakai | 8 | 0.000 |
| HideAndSeek | 8 | 0.000 |
| IRCBot | 1 | 0.000 |
| Kenjiro | 3 | 0.000 |
| Mirai | 17 | 0.000 |
| Muhstik | 12 | 0.000 |
| Okiru | 4 | 0.000 |
| Torii | 5 | 0.000 |
| Trojan | 1 | 0.000 |

**All-missed share (macro over families, mean over seeds): 0.000** (B1 needs >= 0.15)

| detector | method | realised benign host-day FPR | macro host-day TPR |
|---|---|---|---|
| iforest | A | 0.0418 | 0.127 |
| iforest | P | 0.0358 | 0.123 |
| kitnet | A | 0.0688 | 0.067 |
| kitnet | P | 0.0658 | 0.067 |
| mlp_sn_gda | A | 0.0843 | 0.654 |
| mlp_sn_gda | P | 0.0797 | 0.581 |
| rf | A | 0.0644 | 0.967 |
| rf | P | 0.0561 | 0.967 |
| xgb | A | 0.0507 | 0.980 |
| xgb | P | 0.0575 | 0.908 |

B2 not evaluable: 0 all-missed, 66 caught host-days (rule: B2 fails for this dataset)

## nf_cic

Seeds [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; detectors per fold: [np.int64(4)]; attacker host-days (summed over seeds) 132; all-missed 0

| family | attacker host-days | all-missed share |
|---|---|---|
| Bot | 32 | 0.000 |
| BruteForce | 4 | 0.000 |
| DDoS | 80 | 0.000 |
| DoS | 8 | 0.000 |
| Infiltration | 4 | 0.000 |
| Web | 4 | 0.000 |

**All-missed share (macro over families, mean over seeds): 0.000** (B1 needs >= 0.15)

| detector | method | realised benign host-day FPR | macro host-day TPR |
|---|---|---|---|
| iforest | A | 0.0113 | 0.540 |
| iforest | P | 0.0113 | 0.173 |
| mlp_sn_gda | A | 0.0129 | 0.995 |
| mlp_sn_gda | P | 0.0115 | 0.358 |
| rf | A | 0.0120 | 0.870 |
| rf | P | 0.0118 | 0.403 |
| xgb | A | 0.0117 | 0.823 |
| xgb | P | 0.0112 | 0.703 |

B2 not evaluable: 0 all-missed, 132 caught host-days (rule: B2 fails for this dataset)

**B1 FAIL**: all-missed share nf_ctu13 0.205, nf_iot23 0.000, nf_cic 0.000

**B2 FAIL**: descriptors with AUC >= 0.80 per dataset: {"rate": ["nf_ctu13"], "port_entropy": ["nf_ctu13"], "iat_cv": ["nf_ctu13"], "fanout": ["nf_ctu13"], "enc_share": ["nf_ctu13"], "med_dur": ["nf_ctu13"]}

**DISC verdict: stop: no algorithm claim**
