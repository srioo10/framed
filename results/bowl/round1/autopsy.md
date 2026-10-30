# Round 1 autopsy
Champion: {'trial': 125, 'family': 'tabm', 'enc': 'quantile', 'e': 4, 'w': 32, 'depth': 1, 'sn': 0, 'act': 'relu', 'members': 4, 'head': 'min', 'kd': 0.5, 'ls': 0.0, 'focal': 0.0, 'lr': 0.003, 'fitness': 0.7952538494211817, 'unk': 0.6018938074213986, 'known': 0.9886138914209649, 'size': 9872, 'ratio': 0.2436618828461714}
Bosses: {'rf': {'fitness': 0.8155622907175295, 'unk': 0.637568102136724, 'known': 0.9935564792983348}, 'xgb': {'fitness': 0.7699979709543108, 'unk': 0.5480576520879585, 'known': 0.9919382898206633}}

## nf_ctu13/pair_0 (R0=['Virut'], unknown=Menti): champion TPR 0.999, boss XGB 1.000, boss RF 1.000
boss-only 0.001, champion-only 0.000

## nf_ctu13/pair_1 (R0=['Rbot'], unknown=Murlo): champion TPR 0.289, boss XGB 0.410, boss RF 0.280
boss-only 0.146, champion-only 0.017
```
|--- f19 <= 1.29
|   |--- FLOW_DURATION_MILLISECONDS <= 1.52
|   |   |--- IN_BYTES <= -0.08
|   |   |   |--- class: False
|   |   |--- IN_BYTES >  -0.08
|   |   |   |--- class: False
|   |--- FLOW_DURATION_MILLISECONDS >  1.52
|   |   |--- IN_PKTS <= 0.81
|   |   |   |--- class: False
|   |   |--- IN_PKTS >  0.81
|   |   |   |--- class: False
|--- f19 >  1.29
|   |--- f26 <= 0.02
|   |   |--- PROTOCOL <= 0.00
|   |   |   |--- class: False
|   |   |--- PROTOCOL >  0.00
|   |   |   |--- class: False
|   |--- f26 >  0.02
|   |   |--- f19 <= 2.56
|   |   |   |--- class: True
|   |   |--- f19 >  2.56
|   |   |   |--- class: True
```

## nf_ctu13/pair_2 (R0=['Rbot'], unknown=NSIS): champion TPR 0.882, boss XGB 0.944, boss RF 0.914
boss-only 0.085, champion-only 0.010
```
|--- f18 <= -0.73
|   |--- f20 <= 1.24
|   |   |--- f21 <= 1.64
|   |   |   |--- class: False
|   |   |--- f21 >  1.64
|   |   |   |--- class: True
|   |--- f20 >  1.24
|   |   |--- f22 <= -0.98
|   |   |   |--- class: True
|   |   |--- f22 >  -0.98
|   |   |   |--- class: True
|--- f18 >  -0.73
|   |--- f22 <= 0.60
|   |   |--- class: False
|   |--- f22 >  0.60
|   |   |--- class: False
```

## nf_ctu13/pair_3 (R0=['Murlo'], unknown=Neris): champion TPR 0.998, boss XGB 0.999, boss RF 0.998
boss-only 0.001, champion-only 0.000
```
|--- f18 <= -0.71
|   |--- f20 <= 1.00
|   |   |--- f9 <= -0.31
|   |   |   |--- class: False
|   |   |--- f9 >  -0.31
|   |   |   |--- class: False
|   |--- f20 >  1.00
|   |   |--- f20 <= 1.50
|   |   |   |--- class: True
|   |   |--- f20 >  1.50
|   |   |   |--- class: False
|--- f18 >  -0.71
|   |--- f21 <= -1.09
|   |   |--- f21 <= -1.09
|   |   |   |--- class: False
|   |   |--- f21 >  -1.09
|   |   |   |--- class: False
|   |--- f21 >  -1.09
|   |   |--- f20 <= -0.45
|   |   |   |--- class: False
|   |   |--- f20 >  -0.45
|   |   |   |--- class: False
```

## nf_ctu13/pair_4 (R0=['Murlo'], unknown=Rbot): champion TPR 0.766, boss XGB 0.584, boss RF 0.991
boss-only 0.229, champion-only 0.000
```
|--- L4_DST_PORT <= -0.57
|   |--- f15 <= -0.30
|   |   |--- f24 <= 1.10
|   |   |   |--- class: False
|   |   |--- f24 >  1.10
|   |   |   |--- class: False
|   |--- f15 >  -0.30
|   |   |--- f17 <= -0.90
|   |   |   |--- class: True
|   |   |--- f17 >  -0.90
|   |   |   |--- class: False
|--- L4_DST_PORT >  -0.57
|   |--- f16 <= 2.31
|   |   |--- f17 <= 0.26
|   |   |   |--- class: True
|   |   |--- f17 >  0.26
|   |   |   |--- class: False
|   |--- f16 >  2.31
|   |   |--- f17 <= 0.26
|   |   |   |--- class: True
|   |   |--- f17 >  0.26
|   |   |   |--- class: True
```

## nf_ctu13/pair_5 (R0=['Menti'], unknown=Virut): champion TPR 0.677, boss XGB 0.702, boss RF 0.748
boss-only 0.097, champion-only 0.012
```
|--- f8 <= -0.50
|   |--- f18 <= -0.70
|   |   |--- OUT_PKTS <= -0.67
|   |   |   |--- class: True
|   |   |--- OUT_PKTS >  -0.67
|   |   |   |--- class: False
|   |--- f18 >  -0.70
|   |   |--- f21 <= -0.51
|   |   |   |--- class: True
|   |   |--- f21 >  -0.51
|   |   |   |--- class: False
|--- f8 >  -0.50
|   |--- f18 <= -0.58
|   |   |--- FLOW_DURATION_MILLISECONDS <= 1.06
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  1.06
|   |   |   |--- class: True
|   |--- f18 >  -0.58
|   |   |--- class: False
```

## nf_ugr16/pair_0 (R0=['udpscan'], unknown=nerisbotnet): champion TPR 0.872, boss XGB 0.260, boss RF 0.900
boss-only 0.037, champion-only 0.009
```
|--- f17 <= 0.25
|   |--- f22 <= 0.43
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.19
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.19
|   |   |   |--- class: True
|   |--- f22 >  0.43
|   |   |--- f7 <= 0.58
|   |   |   |--- class: True
|   |   |--- f7 >  0.58
|   |   |   |--- class: False
|--- f17 >  0.25
|   |--- f26 <= 0.45
|   |   |--- IN_BYTES <= -0.71
|   |   |   |--- class: False
|   |   |--- IN_BYTES >  -0.71
|   |   |   |--- class: False
|   |--- f26 >  0.45
|   |   |--- f17 <= 1.15
|   |   |   |--- class: True
|   |   |--- f17 >  1.15
|   |   |   |--- class: False
```

## nf_ugr16/pair_1 (R0=['scan44'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000

## nf_ugr16/pair_2 (R0=['scan44'], unknown=nerisbotnet): champion TPR 0.863, boss XGB 0.820, boss RF 0.904
boss-only 0.042, champion-only 0.000
```
|--- f19 <= 0.48
|   |--- f7 <= 0.58
|   |   |--- f22 <= 0.26
|   |   |   |--- class: False
|   |   |--- f22 >  0.26
|   |   |   |--- class: True
|   |--- f7 >  0.58
|   |   |--- f12 <= -1.02
|   |   |   |--- class: True
|   |   |--- f12 >  -1.02
|   |   |   |--- class: False
|--- f19 >  0.48
|   |--- f12 <= -0.30
|   |   |--- f7 <= 0.21
|   |   |   |--- class: False
|   |   |--- f7 >  0.21
|   |   |   |--- class: True
|   |--- f12 >  -0.30
|   |   |--- f21 <= -0.62
|   |   |   |--- class: True
|   |   |--- f21 >  -0.62
|   |   |   |--- class: False
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

## nf_ugr16/pair_4 (R0=['dos'], unknown=nerisbotnet): champion TPR 0.897, boss XGB 0.858, boss RF 0.904
boss-only 0.014, champion-only 0.007
```
|--- f7 <= 1.18
|   |--- f22 <= 0.97
|   |   |--- FLOW_DURATION_MILLISECONDS <= 2.54
|   |   |   |--- class: False
|   |   |--- FLOW_DURATION_MILLISECONDS >  2.54
|   |   |   |--- class: True
|   |--- f22 >  0.97
|   |   |--- f7 <= 1.14
|   |   |   |--- class: True
|   |   |--- f7 >  1.14
|   |   |   |--- class: False
|--- f7 >  1.18
|   |--- f12 <= -0.62
|   |   |--- f19 <= 0.41
|   |   |   |--- class: False
|   |   |--- f19 >  0.41
|   |   |   |--- class: True
|   |--- f12 >  -0.62
|   |   |--- f10 <= 0.10
|   |   |   |--- class: False
|   |   |--- f10 >  0.10
|   |   |   |--- class: False
```

## nf_ugr16/pair_5 (R0=['dos'], unknown=spam): champion TPR 0.000, boss XGB 0.000, boss RF 0.000
boss-only 0.000, champion-only 0.000
