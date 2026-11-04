# OnePass deployment cost (d = 59, k = 6; single CPU thread, batch 1)

| model | serialized size | latency per flow |
|---|---|---|
| MLP fp32 | 26.6 KB | 30.6 µs |
| MLP int8 (OnePass) | 10.9 KB | 77.7 µs |
| DDU MLP + class Gaussians | 232.5 KB | 213.9 µs |
| RF (300 trees) | 287443.7 KB | 13073.9 µs |
| XGB | 14030.8 KB | 618.7 µs |

Host state: one float32 CUSUM value + day stamp per source host (≤ 10 bytes). Latency is PyTorch / sklearn on this CPU, not an MCU measurement.
