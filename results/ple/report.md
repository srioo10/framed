# ple prototype gate (PREREG §9 PLE)

Seed 1. Macro host-day TPR over held-out families.

ds         nf_ctu13        nf_iot23        nf_ugr16       
method            A      P        A      P        A      P
detector                                                  
clip_ddu      0.697  0.363    0.556  0.333    0.434  0.355
kitnet        0.485  0.008    0.000  0.000    0.557  0.520
mlp_sn_gda    0.697  0.352    0.444  0.167    0.429  0.354
ple_ce        0.689  0.705    0.722  0.722    0.694  0.660
ple_ddu       0.640  0.487    0.333  0.333    0.631  0.606
rf            0.632  0.752    1.000  0.889    0.699  0.715
xgb           0.632  0.668    1.000  0.833    0.706  0.631

## Gate

      ds  ple_ddu_tpr method  max_baseline_tpr  tpr_ok  fpr_ok  fwd_fpr_hi      fpr
nf_ugr16     0.630771      A          0.714724   False   False    0.018556      NaN
nf_ctu13     0.639683      A          0.752381   False    True         NaN 0.068106
nf_iot23     0.333333      A          1.000000   False   False         NaN 0.107937

**Gate: FAIL** (needs >= 2 of 3 datasets)


## nf_ugr16 forward benign FPR

  detector method  fwd_fpr  fwd_hi
  clip_ddu      A   0.0144  0.0200
  clip_ddu      P   0.0157  0.0204
    kitnet      A   0.0104  0.0159
    kitnet      P   0.0036  0.0059
mlp_sn_gda      A   0.0140  0.0196
mlp_sn_gda      P   0.0132  0.0174
    ple_ce      A   0.0129  0.0184
    ple_ce      P   0.0076  0.0106
   ple_ddu      A   0.0129  0.0186
   ple_ddu      P   0.0106  0.0145
        rf      A   0.0175  0.0247
        rf      P   0.0089  0.0133
       xgb      A   0.0158  0.0225
       xgb      P   0.0119  0.0169
