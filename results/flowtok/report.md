# FlowTok prototype gate (PREREG §9 TOK)

Seed 1. Macro host-day TPR over held-out families.

ds         nf_ctu13        nf_iot23        nf_ugr16       
method            A      P        A      P        A      P
detector                                                  
ft_hc         0.560  0.230    0.833  0.722    0.618  0.167
ft_mlp        0.673  0.339    0.889  0.722    0.436  0.381
ft_order0     0.697  0.483    0.574  0.333    0.540  0.401
ft_tree       0.697  0.303    0.611  0.500    0.503  0.198
kitnet        0.485  0.008    0.000  0.000    0.557  0.520
mlp_sn_gda    0.697  0.352    0.444  0.167    0.429  0.354
rf            0.632  0.752    1.000  0.889    0.699  0.715
xgb           0.632  0.668    1.000  0.833    0.706  0.631

## Gate

      ds  ft_tree_tpr method  max_baseline_tpr  tpr_ok  fpr_ok  fwd_fpr_hi      fpr
nf_ugr16     0.502598      A          0.714724   False   False     0.02291      NaN
nf_ctu13     0.696825      A          0.752381   False    True         NaN 0.084718
nf_iot23     0.611111      A          1.000000   False    True         NaN 0.020106

**Gate: FAIL** (needs >= 2 of 3 datasets)


## nf_ugr16 forward benign FPR

  detector method  fwd_fpr  fwd_hi
     ft_hc      A   0.0358  0.0425
     ft_hc      P   0.0105  0.0157
    ft_mlp      A   0.0200  0.0256
    ft_mlp      P   0.0226  0.0266
 ft_order0      A   0.0139  0.0197
 ft_order0      P   0.0104  0.0140
   ft_tree      A   0.0164  0.0229
   ft_tree      P   0.0161  0.0213
    kitnet      A   0.0104  0.0159
    kitnet      P   0.0036  0.0059
mlp_sn_gda      A   0.0140  0.0196
mlp_sn_gda      P   0.0132  0.0174
        rf      A   0.0175  0.0247
        rf      P   0.0089  0.0133
       xgb      A   0.0158  0.0225
       xgb      P   0.0119  0.0169
