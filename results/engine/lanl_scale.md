# Engine scale on LANL 2015 flows (PREREG §9 ENG)

130.0M flows, 36.2 days, 11,154 distinct sources / 8,711 distinct destinations. Parity model (nf_iot23 s1, held Okiru):
scores carry no detection meaning here; only cost is measured. One core, `-O2`, laptop (Intel, 12 threads busy with nothing else on core 11).

| table slots | allocated state | max RSS | µs/flow | evictions |
|---|---|---|---|---|
| 1,024 | 2.2 MB | 4.9 MB | 2.68 | 34,409,521 |
| 8,192 | 17.8 MB | 20.1 MB | 2.77 | 101,667 |
| 65,536 | 142.6 MB | 58.9 MB | 2.64 | 0 |
| 262,144 | 570.4 MB | 71.2 MB | 2.60 | 0 |

- Per-flow cost is flat in table size (2.6–2.8 µs, ≈ 360k flows/s on one core).
- RSS < allocated state for large tables: untouched slots are never paged in, so RSS tracks the active hosts
  (~20k hosts × 2,176 B ≈ 43 MB plus input buffers), not the table size.
- At 1,024 slots the table thrashes (34M evictions for ~20k active hosts); 8,192 slots already cut evictions 340×.
  Raw log: `lanl_scale.txt`.
