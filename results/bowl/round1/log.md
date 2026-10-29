# Round 1 roster rationale
- tabm (tree bins, 4 members, KD 0.9, min-member head): the §9 SEL amendment-3 candidate, shrunk to fit 64 KB / speed.
- mlp periodic + SELU + label smoothing: RealMLP-lite defaults (Holzmüller et al. NeurIPS'24).
- node T32 D4: NODE-lite (Popov et al. ICLR'20); deployed form is hard oblivious splits.
- leafnet 16 trees x 16 leaves: GBDT-leaf -> net (He et al. 2014).
- mlp raw + spectral norm + Mahalanobis head: DDU (our Phase-1 detector).
- mlp quantile bins: ple_ce (§9 PLE).
- mlp tree bins + KD 0.9: distilled student.
FT-Transformer-tiny deferred: per-flow attention over ~60 tokens exceeds the speed limit at any useful width.
