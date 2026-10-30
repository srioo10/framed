# Round 3 autopsy
Champion: {'trial': 4, 'family': 'mlp', 'enc': 'raw', 'e': 2, 'w': 64, 'depth': 2, 'sn': 0, 'act': 'relu', 'members': 1, 'head': 'energy', 'kd': 0.5, 'syn': 0.5, 'ens': 1, 'ls': 0.0, 'focal': 2.0, 'lr': 0.003, 'fitness': 0.8019965696376851, 'unk': 0.6612516970471091, 'known': 0.942741442228261, 'size': 6277, 'ratio': 0.017895567567898722}
Bosses: {'cat': {'fitness': 0.8115574028550516, 'unk': 0.6400510098500125, 'known': 0.9830637958600908}, 'gang': {'fitness': 0.8168489457825386, 'unk': 0.650930023046699, 'known': 0.9827678685183785}, 'lgbm': {'fitness': 0.799050798854633, 'unk': 0.6156479809844261, 'known': 0.9824536167248397}, 'rf': {'fitness': 0.818983747475001, 'unk': 0.6569239122007551, 'known': 0.9810435827492471}, 'xgb': {'fitness': 0.7711709155700186, 'unk': 0.5625003772321931, 'known': 0.979841453907844}, 'xgbx': {'fitness': 0.7965845966793222, 'unk': 0.6102713214024037, 'known': 0.9828978719562406}}

## nf_ctu13/pair_0 (R0=['Virut'], unknown=Menti): champion TPR 0.999, boss XGB 1.000, boss RF 1.000
boss-only 0.001, champion-only 0.000

## nf_ctu13/pair_1 (R0=['Rbot'], unknown=Murlo): champion TPR 0.876, boss XGB 0.410, boss RF 0.280
boss-only 0.082, champion-only 0.541
```
|--- f19 <= 1.48
|   |--- f22 <= -0.04
|   |   |--- IN_PKTS <= 2.94
|   |   |   |--- class: False
|   |   |--- IN_PKTS >  2.94
|   |   |   |--- class: False
|   |--- f22 >  -0.04
|   |   |--- class: False
|--- f19 >  1.48
|   |--- f20 <= 1.99
|   |   |--- f17 <= -3.14
|   |   |   |--- class: True
|   |   |--- f17 >  -3.14
|   |   |   |--- class: False
|   |--- f20 >  1.99
|   |   |--- f7 <= -1.78
|   |   |   |--- class: False
|   |   |--- f7 >  -1.78
|   |   |   |--- class: False
```

## nf_ctu13/pair_2 (R0=['Rbot'], unknown=NSIS): champion TPR 0.900, boss XGB 0.944, boss RF 0.914
boss-only 0.064, champion-only 0.006
```
|--- f18 <= -0.89
|   |--- IN_BYTES <= -0.24
|   |   |--- f19 <= 7.18
|   |   |   |--- class: False
|   |   |--- f19 >  7.18
|   |   |   |--- class: True
|   |--- IN_BYTES >  -0.24
|   |   |--- f26 <= 1.21
|   |   |   |--- class: False
|   |   |--- f26 >  1.21
|   |   |   |--- class: True
|--- f18 >  -0.89
|   |--- f17 <= -2.67
|   |   |--- FLOW_DURATION_MILLISECONDS <= -0.07
|   |   |   |--- class: True
|   |   |--- FLOW_DURATION_MILLISECONDS >  -0.07
|   |   |   |--- class: False
|   |--- f17 >  -2.67
|   |   |--- f8 <= -0.85
|   |   |   |--- class: False
|   |   |--- f8 >  -0.85
|   |   |   |--- class: False
```

## nf_ctu13/pair_3 (R0=['Murlo'], unknown=Neris): champion TPR 0.956, boss XGB 0.999, boss RF 0.998
boss-only 0.043, champion-only 0.000
```
|--- f18 <= -0.25
|   |--- f17 <= 1.87
|   |   |--- f7 <= 1.15
|   |   |   |--- class: False
|   |   |--- f7 >  1.15
|   |   |   |--- class: True
|   |--- f17 >  1.87
|   |   |--- f8 <= -0.06
|   |   |   |--- class: True
|   |   |--- f8 >  -0.06
|   |   |   |--- class: False
|--- f18 >  -0.25
|   |--- f8 <= -0.38
|   |   |--- f17 <= 1.49
|   |   |   |--- class: False
|   |   |--- f17 >  1.49
|   |   |   |--- class: False
|   |--- f8 >  -0.38
|   |   |--- f21 <= -1.09
|   |   |   |--- class: False
|   |   |--- f21 >  -1.09
|   |   |   |--- class: False
```

## nf_ctu13/pair_4 (R0=['Murlo'], unknown=Rbot): champion TPR 0.482, boss XGB 0.584, boss RF 0.991
boss-only 0.514, champion-only 0.001
```
|--- IN_BYTES <= 1.65
|   |--- f17 <= -2.82
|   |   |--- f25 <= -0.27
|   |   |   |--- class: False
|   |   |--- f25 >  -0.27
|   |   |   |--- class: True
|   |--- f17 >  -2.82
|   |   |--- f17 <= 0.32
|   |   |   |--- class: False
|   |   |--- f17 >  0.32
|   |   |   |--- class: True
|--- IN_BYTES >  1.65
|   |--- FLOW_DURATION_MILLISECONDS <= 2.36
|   |   |--- f20 <= 0.07
|   |   |   |--- class: True
|   |   |--- f20 >  0.07
|   |   |   |--- class: True
|   |--- FLOW_DURATION_MILLISECONDS >  2.36
|   |   |--- OUT_BYTES <= 1.27
|   |   |   |--- class: False
|   |   |--- OUT_BYTES >  1.27
|   |   |   |--- class: False
```

## nf_ctu13/pair_5 (R0=['Menti'], unknown=Virut): champion TPR 0.660, boss XGB 0.702, boss RF 0.748
boss-only 0.116, champion-only 0.014
```
|--- f18 <= -0.48
|   |--- f18 <= -0.71
|   |   |--- f8 <= -0.82
|   |   |   |--- class: False
|   |   |--- f8 >  -0.82
|   |   |   |--- class: False
|   |--- f18 >  -0.71
|   |   |--- f8 <= -0.32
|   |   |   |--- class: True
|   |   |--- f8 >  -0.32
|   |   |   |--- class: False
|--- f18 >  -0.48
|   |--- f18 <= -0.38
|   |   |--- FLOW_DURATION_MILLISECONDS <= -0.07
|   |   |   |--- class: True
|   |   |--- FLOW_DURATION_MILLISECONDS >  -0.07
|   |   |   |--- class: False
|   |--- f18 >  -0.38
|   |   |--- f18 <= -0.35
|   |   |   |--- class: False
|   |   |--- f18 >  -0.35
|   |   |   |--- class: False
```

## nf_ugr16/pair_0 (R0=['udpscan'], unknown=nerisbotnet): champion TPR 0.896, boss XGB 0.260, boss RF 0.900
boss-only 0.023, champion-only 0.019
```
|--- f19 <= -2.06
|   |--- f17 <= 0.73
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.19
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.19
|   |   |   |--- class: True
|   |--- f17 >  0.73
|   |   |--- PROTOCOL <= 3.37
|   |   |   |--- class: True
|   |   |--- PROTOCOL >  3.37
|   |   |   |--- class: False
|--- f19 >  -2.06
|   |--- f7 <= -0.21
|   |   |--- f22 <= 0.00
|   |   |   |--- class: False
|   |   |--- f22 >  0.00
|   |   |   |--- class: True
|   |--- f7 >  -0.21
|   |   |--- f17 <= -0.09
|   |   |   |--- class: False
|   |   |--- f17 >  -0.09
|   |   |   |--- class: False
```

## nf_ugr16/pair_1 (R0=['scan44'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000

## nf_ugr16/pair_2 (R0=['scan44'], unknown=nerisbotnet): champion TPR 0.894, boss XGB 0.820, boss RF 0.904
boss-only 0.026, champion-only 0.016
```
|--- f10 <= -0.27
|   |--- PROTOCOL <= 3.43
|   |   |--- f17 <= 0.05
|   |   |   |--- class: False
|   |   |--- f17 >  0.05
|   |   |   |--- class: True
|   |--- PROTOCOL >  3.43
|   |   |--- f17 <= -0.19
|   |   |   |--- class: False
|   |   |--- f17 >  -0.19
|   |   |   |--- class: False
|--- f10 >  -0.27
|   |--- f23 <= 7.52
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.12
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.12
|   |   |   |--- class: True
|   |--- f23 >  7.52
|   |   |--- f22 <= -0.14
|   |   |   |--- class: False
|   |   |--- f22 >  -0.14
|   |   |   |--- class: True
```

## nf_ugr16/pair_3 (R0=['nerisbotnet'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.010
boss-only 0.010, champion-only 0.000
```
|--- f7 <= 0.89
|   |--- f12 <= 1.03
|   |   |--- f17 <= 0.62
|   |   |   |--- class: False
|   |   |--- f17 >  0.62
|   |   |   |--- class: False
|   |--- f12 >  1.03
|   |   |--- class: True
|--- f7 >  0.89
|   |--- f17 <= 0.53
|   |   |--- class: True
|   |--- f17 >  0.53
|   |   |--- class: True
```

## nf_ugr16/pair_4 (R0=['dos'], unknown=nerisbotnet): champion TPR 0.527, boss XGB 0.858, boss RF 0.904
boss-only 0.385, champion-only 0.007
```
|--- f26 <= -0.61
|   |--- f16 <= -0.21
|   |   |--- f22 <= 1.67
|   |   |   |--- class: False
|   |   |--- f22 >  1.67
|   |   |   |--- class: False
|   |--- f16 >  -0.21
|   |   |--- f9 <= 0.63
|   |   |   |--- class: True
|   |   |--- f9 >  0.63
|   |   |   |--- class: False
|--- f26 >  -0.61
|   |--- IN_BYTES <= -1.81
|   |   |--- f7 <= 1.93
|   |   |   |--- class: False
|   |   |--- f7 >  1.93
|   |   |   |--- class: True
|   |--- IN_BYTES >  -1.81
|   |   |--- f23 <= 7.08
|   |   |   |--- class: False
|   |   |--- f23 >  7.08
|   |   |   |--- class: False
```

## nf_ugr16/pair_5 (R0=['dos'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000

## nf_cic/pair_0 (R0=['Web'], unknown=Bot): champion TPR 0.929, boss XGB 0.530, boss RF 0.872
boss-only 0.015, champion-only 0.072

## nf_cic/pair_1 (R0=['Infiltration'], unknown=BruteForce): champion TPR 0.998, boss XGB 1.000, boss RF 1.000
boss-only 0.002, champion-only 0.000
```
|--- f44 <= 0.08
|   |--- class: True
|--- f44 >  0.08
|   |--- f54 <= -0.41
|   |   |--- class: False
|   |--- f54 >  -0.41
|   |   |--- class: False
```

## nf_cic/pair_2 (R0=['DoS'], unknown=DDoS): champion TPR 0.999, boss XGB 0.985, boss RF 0.983
boss-only 0.000, champion-only 0.012

## nf_cic/pair_3 (R0=['BruteForce'], unknown=DoS): champion TPR 0.997, boss XGB 0.805, boss RF 0.986
boss-only 0.001, champion-only 0.011
```
|--- SRC_TO_DST_IAT_MIN <= 2.84
|   |--- f52 <= -0.59
|   |   |--- class: False
|   |--- f52 >  -0.59
|   |   |--- f51 <= -0.31
|   |   |   |--- class: False
|   |   |--- f51 >  -0.31
|   |   |   |--- class: False
|--- SRC_TO_DST_IAT_MIN >  2.84
|   |--- SRC_TO_DST_IAT_MAX <= 1.26
|   |   |--- class: True
|   |--- SRC_TO_DST_IAT_MAX >  1.26
|   |   |--- class: False
```

## nf_cic/pair_4 (R0=['BruteForce'], unknown=Infiltration): champion TPR 0.210, boss XGB 0.228, boss RF 0.330
boss-only 0.125, champion-only 0.001
```
|--- SRC_TO_DST_AVG_THROUGHPUT <= -1.48
|   |--- f58 <= -1.16
|   |   |--- SRC_TO_DST_AVG_THROUGHPUT <= -1.85
|   |   |   |--- class: True
|   |   |--- SRC_TO_DST_AVG_THROUGHPUT >  -1.85
|   |   |   |--- class: False
|   |--- f58 >  -1.16
|   |   |--- DST_TO_SRC_AVG_THROUGHPUT <= -1.68
|   |   |   |--- class: False
|   |   |--- DST_TO_SRC_AVG_THROUGHPUT >  -1.68
|   |   |   |--- class: True
|--- SRC_TO_DST_AVG_THROUGHPUT >  -1.48
|   |--- L7_PROTO <= 1.87
|   |   |--- TCP_FLAGS <= 0.88
|   |   |   |--- class: False
|   |   |--- TCP_FLAGS >  0.88
|   |   |   |--- class: False
|   |--- L7_PROTO >  1.87
|   |   |--- f54 <= -0.78
|   |   |   |--- class: True
|   |   |--- f54 >  -0.78
|   |   |   |--- class: False
```

## nf_cic/pair_5 (R0=['Bot'], unknown=Web): champion TPR 0.323, boss XGB 0.000, boss RF 0.003
boss-only 0.000, champion-only 0.320
