# Round 2 autopsy
Champion: {'trial': 63, 'family': 'mlp', 'enc': 'raw', 'e': 4, 'w': 64, 'depth': 2, 'sn': 0, 'act': 'selu', 'members': 1, 'head': 'nlog', 'kd': 0.5, 'teacher': 'xgb', 'syn': 0.5, 'ls': 0.0, 'focal': 0.0, 'lr': 0.003, 'fitness': 0.8099453114211738, 'unk': 0.6326432308559329, 'known': 0.9872473919864145, 'size': 6277, 'ratio': 0.04277569691646775}
Bosses: {'rf': {'fitness': 0.8155622907175295, 'unk': 0.637568102136724, 'known': 0.9935564792983348}, 'xgb': {'fitness': 0.7699979709543108, 'unk': 0.5480576520879585, 'known': 0.9919382898206633}}

## nf_ctu13/pair_0 (R0=['Virut'], unknown=Menti): champion TPR 0.999, boss XGB 1.000, boss RF 1.000
boss-only 0.001, champion-only 0.000

## nf_ctu13/pair_1 (R0=['Rbot'], unknown=Murlo): champion TPR 0.864, boss XGB 0.410, boss RF 0.280
boss-only 0.079, champion-only 0.525
```
|--- L4_DST_PORT <= -0.19
|   |--- FLOW_DURATION_MILLISECONDS <= -1.06
|   |   |--- class: False
|   |--- FLOW_DURATION_MILLISECONDS >  -1.06
|   |   |--- f17 <= -2.69
|   |   |   |--- class: True
|   |   |--- f17 >  -2.69
|   |   |   |--- class: False
|--- L4_DST_PORT >  -0.19
|   |--- OUT_BYTES <= 1.45
|   |   |--- f19 <= 2.06
|   |   |   |--- class: False
|   |   |--- f19 >  2.06
|   |   |   |--- class: False
|   |--- OUT_BYTES >  1.45
|   |   |--- f7 <= -1.78
|   |   |   |--- class: True
|   |   |--- f7 >  -1.78
|   |   |   |--- class: False
```

## nf_ctu13/pair_2 (R0=['Rbot'], unknown=NSIS): champion TPR 0.922, boss XGB 0.944, boss RF 0.914
boss-only 0.051, champion-only 0.015
```
|--- f22 <= 0.57
|   |--- f8 <= -0.85
|   |   |--- IN_BYTES <= 1.94
|   |   |   |--- class: False
|   |   |--- IN_BYTES >  1.94
|   |   |   |--- class: False
|   |--- f8 >  -0.85
|   |   |--- FLOW_DURATION_MILLISECONDS <= -1.37
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  -1.37
|   |   |   |--- class: False
|--- f22 >  0.57
|   |--- f17 <= -0.36
|   |   |--- f9 <= -0.47
|   |   |   |--- class: True
|   |   |--- f9 >  -0.47
|   |   |   |--- class: True
|   |--- f17 >  -0.36
|   |   |--- class: False
```

## nf_ctu13/pair_3 (R0=['Murlo'], unknown=Neris): champion TPR 0.998, boss XGB 0.999, boss RF 0.998
boss-only 0.001, champion-only 0.000
```
|--- f17 <= -1.42
|   |--- FLOW_DURATION_MILLISECONDS <= -0.08
|   |   |--- class: False
|   |--- FLOW_DURATION_MILLISECONDS >  -0.08
|   |   |--- f11 <= -0.16
|   |   |   |--- class: False
|   |   |--- f11 >  -0.16
|   |   |   |--- class: False
|--- f17 >  -1.42
|   |--- f21 <= -1.10
|   |   |--- class: False
|   |--- f21 >  -1.10
|   |   |--- f18 <= -0.42
|   |   |   |--- class: False
|   |   |--- f18 >  -0.42
|   |   |   |--- class: False
```

## nf_ctu13/pair_4 (R0=['Murlo'], unknown=Rbot): champion TPR 0.981, boss XGB 0.584, boss RF 0.991
boss-only 0.015, champion-only 0.001
```
|--- f18 <= -0.44
|   |--- FLOW_DURATION_MILLISECONDS <= 0.20
|   |   |--- f26 <= 0.02
|   |   |   |--- class: True
|   |   |--- f26 >  0.02
|   |   |   |--- class: True
|   |--- FLOW_DURATION_MILLISECONDS >  0.20
|   |   |--- L4_DST_PORT <= 1.81
|   |   |   |--- class: False
|   |   |--- L4_DST_PORT >  1.81
|   |   |   |--- class: True
|--- f18 >  -0.44
|   |--- L4_DST_PORT <= 0.27
|   |   |--- L4_DST_PORT <= 0.15
|   |   |   |--- class: False
|   |   |--- L4_DST_PORT >  0.15
|   |   |   |--- class: False
|   |--- L4_DST_PORT >  0.27
|   |   |--- FLOW_DURATION_MILLISECONDS <= 1.63
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  1.63
|   |   |   |--- class: False
```

## nf_ctu13/pair_5 (R0=['Menti'], unknown=Virut): champion TPR 0.706, boss XGB 0.702, boss RF 0.748
boss-only 0.069, champion-only 0.013
```
|--- f19 <= -0.12
|   |--- f18 <= -0.48
|   |   |--- f18 <= -0.76
|   |   |   |--- class: False
|   |   |--- f18 >  -0.76
|   |   |   |--- class: False
|   |--- f18 >  -0.48
|   |   |--- class: False
|--- f19 >  -0.12
|   |--- f18 <= -0.71
|   |   |--- f20 <= -0.25
|   |   |   |--- class: False
|   |   |--- f20 >  -0.25
|   |   |   |--- class: False
|   |--- f18 >  -0.71
|   |   |--- OUT_BYTES <= 0.20
|   |   |   |--- class: False
|   |   |--- OUT_BYTES >  0.20
|   |   |   |--- class: True
```

## nf_ugr16/pair_0 (R0=['udpscan'], unknown=nerisbotnet): champion TPR 0.877, boss XGB 0.260, boss RF 0.900
boss-only 0.038, champion-only 0.014
```
|--- f17 <= 0.28
|   |--- f22 <= 0.47
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.19
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.19
|   |   |   |--- class: True
|   |--- f22 >  0.47
|   |   |--- f7 <= 0.82
|   |   |   |--- class: True
|   |   |--- f7 >  0.82
|   |   |   |--- class: False
|--- f17 >  0.28
|   |--- PROTOCOL <= 3.37
|   |   |--- f8 <= 7.73
|   |   |   |--- class: True
|   |   |--- f8 >  7.73
|   |   |   |--- class: True
|   |--- PROTOCOL >  3.37
|   |   |--- f16 <= -0.22
|   |   |   |--- class: False
|   |   |--- f16 >  -0.22
|   |   |   |--- class: False
```

## nf_ugr16/pair_1 (R0=['scan44'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000

## nf_ugr16/pair_2 (R0=['scan44'], unknown=nerisbotnet): champion TPR 0.869, boss XGB 0.820, boss RF 0.904
boss-only 0.036, champion-only 0.001
```
|--- f17 <= 0.19
|   |--- f22 <= -0.45
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.19
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.19
|   |   |   |--- class: True
|   |--- f22 >  -0.45
|   |   |--- f24 <= 0.49
|   |   |   |--- class: False
|   |   |--- f24 >  0.49
|   |   |   |--- class: True
|--- f17 >  0.19
|   |--- PROTOCOL <= 3.43
|   |   |--- f12 <= -0.47
|   |   |   |--- class: True
|   |   |--- f12 >  -0.47
|   |   |   |--- class: False
|   |--- PROTOCOL >  3.43
|   |   |--- f8 <= 2.57
|   |   |   |--- class: False
|   |   |--- f8 >  2.57
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

## nf_ugr16/pair_4 (R0=['dos'], unknown=nerisbotnet): champion TPR 0.895, boss XGB 0.858, boss RF 0.904
boss-only 0.016, champion-only 0.007
```
|--- FLOW_DURATION_MILLISECONDS <= 2.54
|   |--- f15 <= -0.12
|   |   |--- f7 <= 1.17
|   |   |   |--- class: False
|   |   |--- f7 >  1.17
|   |   |   |--- class: False
|   |--- f15 >  -0.12
|   |   |--- f24 <= -1.73
|   |   |   |--- class: True
|   |   |--- f24 >  -1.73
|   |   |   |--- class: False
|--- FLOW_DURATION_MILLISECONDS >  2.54
|   |--- IN_PKTS <= 0.27
|   |   |--- class: True
|   |--- IN_PKTS >  0.27
|   |   |--- f9 <= -1.04
|   |   |   |--- class: True
|   |   |--- f9 >  -1.04
|   |   |   |--- class: True
```

## nf_ugr16/pair_5 (R0=['dos'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000
