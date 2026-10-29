# tabm prototype gate (PREREG §9 TABM)

Seed 1. Macro host-day TPR over held-out families.

ds         nf_ctu13        nf_iot23        nf_ugr16       
method            A      P        A      P        A      P
detector                                                  
kitnet        0.485  0.008    0.000  0.000    0.557  0.520
mlp_sn_gda    0.697  0.352    0.444  0.167    0.429  0.354
rf            0.632  0.752    1.000  0.889    0.699  0.715
tabm          0.632  0.705    0.833  0.833    0.707  0.692
tabm_mi       0.703  0.625    0.889  0.833    0.586  0.383
tabm_min      0.656  0.713    1.000  0.833    0.715  0.702
tabm_raw      0.711  0.632    1.000  1.000    0.696  0.603
xgb           0.632  0.668    1.000  0.833    0.706  0.631

## Gate

      ds  tabm_tpr method  max_baseline_tpr  tpr_ok  fpr_ok  fwd_fpr_hi      fpr
nf_ugr16  0.706783      A          0.714724    True   False    0.016376      NaN
nf_ctu13  0.704762      P          0.752381    True    True         NaN 0.089701
nf_iot23  0.833333      A          1.000000   False    True         NaN 0.071958

**Gate: FAIL** (needs >= 2 of 3 datasets)


## nf_ugr16 forward benign FPR

  detector method  fwd_fpr  fwd_hi
    kitnet      A   0.0104  0.0159
    kitnet      P   0.0036  0.0059
mlp_sn_gda      A   0.0140  0.0196
mlp_sn_gda      P   0.0132  0.0174
        rf      A   0.0175  0.0247
        rf      P   0.0089  0.0133
      tabm      A   0.0113  0.0164
      tabm      P   0.0071  0.0102
   tabm_mi      A   0.0137  0.0192
   tabm_mi      P   0.0097  0.0134
  tabm_min      A   0.0131  0.0184
  tabm_min      P   0.0092  0.0127
  tabm_raw      A   0.0174  0.0241
  tabm_raw      P   0.0108  0.0161
       xgb      A   0.0158  0.0225
       xgb      P   0.0119  0.0169
