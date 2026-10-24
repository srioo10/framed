# unsw_raw leak-column controls, TPR@1%FPR on the held-out family (mean [95% t-CI] over seeds)

best single feature = oracle pick on the test rows; `orig:` = the creators' ct_*, plain ct_* = causal.

| held-out | n | TTL-only RF | IP-only RF | best single feature (most common) | its TPR | best non-ct_* feature | its TPR |
|---|---|---|---|---|---|---|---|
| Analysis+Backdoor | 132 | 0.700 [0.539, 0.861] | 0.000 [0.000, 0.000] | sbytes | 0.805 [0.691, 0.918] | sbytes | 0.805 [0.691, 0.918] |
| DoS | 690 | 0.633 [0.391, 0.874] | 0.000 [0.000, 0.000] | tcprtt | 0.765 [0.663, 0.868] | tcprtt | 0.765 [0.663, 0.868] |
| Exploits | 4980 | 0.282 [-0.247, 0.812] | 0.200 [-0.355, 0.755] | tcprtt | 0.812 [0.695, 0.929] | tcprtt | 0.812 [0.695, 0.929] |
| Fuzzers | 3590 | 0.535 [0.017, 1.053] | 0.200 [-0.355, 0.755] | tcprtt | 0.597 [0.488, 0.705] | tcprtt | 0.597 [0.488, 0.705] |
| Generic | 3243 | 0.279 [-0.223, 0.781] | 0.600 [-0.080, 1.280] | sbytes | 0.850 [0.725, 0.976] | sbytes | 0.773 [0.436, 1.109] |
| Recon | 1634 | 0.523 [-0.017, 1.064] | 0.600 [-0.080, 1.280] | dmeansz | 0.800 [0.675, 0.924] | dmeansz | 0.800 [0.675, 0.924] |
| Shellcode+Worms | 303 | 0.554 [0.245, 0.864] | 0.000 [0.000, 0.000] | ct_dst_ltm | 0.604 [0.567, 0.641] | dmeansz | 0.541 [0.477, 0.605] |

## Tie-robust readouts for the TTL and IP controls (mean over seeds)

TPR with >= at the cal threshold, the cal FPR that >= implies, and pAUC over FPR <= 1% (0.5 = chance).

| held-out | TTL TPR(>=) | TTL cal FPR(>=) | TTL pAUC | IP TPR(>=) | IP cal FPR(>=) | IP pAUC |
|---|---|---|---|---|---|---|
| Analysis+Backdoor | 0.970 | 0.015 | 0.810 | 0.800 | 0.018 | 0.678 |
| DoS | 0.960 | 0.015 | 0.729 | 0.600 | 0.018 | 0.678 |
| Exploits | 0.415 | 0.012 | 0.585 | 0.600 | 0.018 | 0.785 |
| Fuzzers | 0.683 | 0.015 | 0.663 | 1.000 | 0.018 | 0.778 |
| Generic | 0.502 | 0.013 | 0.552 | 1.000 | 0.018 | 0.785 |
| Recon | 0.838 | 0.015 | 0.656 | 1.000 | 0.018 | 0.775 |
| Shellcode+Worms | 0.888 | 0.015 | 0.694 | 0.800 | 0.018 | 0.679 |

realised benign-test FPR: TTL Analysis+Backdoor 0.0070, DoS 0.0070, Exploits 0.0076, Fuzzers 0.0088, Generic 0.0069, Recon 0.0088, Shellcode+Worms 0.0088

## Top 5 single features per held-out family (mean TPR@1% over seeds)

- Analysis+Backdoor: sbytes 0.764, tcprtt 0.752, ackdat 0.733, synack 0.676, res_bdy_len 0.502
- DoS: tcprtt 0.765, ackdat 0.713, synack 0.691, sbytes 0.615, ct_dst_ltm 0.581
- Exploits: tcprtt 0.812, ackdat 0.773, synack 0.749, sbytes 0.569, ct_dst_ltm 0.527
- Fuzzers: tcprtt 0.596, ackdat 0.580, synack 0.540, sbytes 0.323, dmeansz 0.320
- Generic: state 0.659, sbytes 0.589, orig:ct_dst_src_ltm 0.508, ct_src_dport_ltm 0.410, ct_dst_src_ltm 0.395
- Recon: tcprtt 0.693, ackdat 0.656, synack 0.629, ct_dst_ltm 0.571, dmeansz 0.556
- Shellcode+Worms: ct_dst_ltm 0.600, dmeansz 0.477, tcprtt 0.458, sbytes 0.445, ackdat 0.443
