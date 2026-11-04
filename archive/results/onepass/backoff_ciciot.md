# C7b — taxonomy back-off on CICIoT2023 (PREREG 2026-10-26)

Share of flagged unknown flows (own 1% cal threshold) given the correct parent; leaf cases only.

| held-out | parent | method | flagged | correct parent | confusion (pooled) |
|---|---|---|---|---|---|
| BruteForce | Access | ddu | 3398 | 0.020 | {'Recon': 3275, 'Spoofing': 54, 'Access': 69} |
| DDoS | Volumetric | ddu | 39942 | 0.107 | {'Access': 13820, 'Recon': 21653, 'Volumetric': 4257, 'Spoofing': 212} |
| DoS | Volumetric | ddu | 4579 | 0.972 | {'Volumetric': 4453, 'Recon': 69, 'Spoofing': 23, 'Access': 34} |
| Flood | Volumetric | ddu | 98731 | 0.014 | {'Recon': 67043, 'Spoofing': 22119, 'Access': 8147, 'Volumetric': 1422} |
| Mirai | Volumetric | ddu | 86961 | 0.985 | {'Volumetric': 85623, 'Recon': 764, 'Spoofing': 451, 'Access': 123} |
| Web | Access | ddu | 1559 | 0.156 | {'Recon': 496, 'Access': 243, 'Spoofing': 820} |
| BruteForce | Access | softmax | 3127 | 0.100 | {'Recon': 2687, 'Access': 314, 'Spoofing': 126} |
| DDoS | Volumetric | softmax | 76968 | 0.890 | {'Volumetric': 68526, 'Recon': 2939, 'Spoofing': 5320, 'Access': 183} |
| DoS | Volumetric | softmax | 23027 | 1.000 | {'Volumetric': 23022, 'Spoofing': 5} |
| Flood | Volumetric | softmax | 99807 | 0.686 | {'Volumetric': 68513, 'Recon': 23713, 'Spoofing': 7066, 'Access': 515} |
| Mirai | Volumetric | softmax | 100000 | 0.998 | {'Volumetric': 99808, 'Spoofing': 164, 'Access': 27, 'Recon': 1} |
| Web | Access | softmax | 4408 | 0.025 | {'Recon': 2307, 'Spoofing': 1988, 'Access': 112, 'Volumetric': 1} |

C7b (claim): ddu macro over leaf families 0.376, flow-weighted 0.409, chance 0.250 → FAIL

softmax control: softmax macro over leaf families 0.617, flow-weighted 0.847, chance 0.250 → PASS
