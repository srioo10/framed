# Development selection (PREREG §9 SEL), seed 1

Flow TPR at a matched 1% benign flow FPR on test hosts (macro over held-out families):

ds           nf_cic  nf_ctu13  nf_ugr16
detector                               
kitnet        0.024     0.045     0.000
lxgb_hc       0.874     0.798     0.000
mlp_sn_gda    0.523     0.039     0.264
ple_ce        0.611     0.749     0.000
rf            0.999     0.765     0.000
tabm          0.948     0.770     0.000
tabm_kd       0.875     0.676     0.000
tabm_kd_min   0.899     0.683     0.000
tabm_min      0.964     0.791     0.000
tabm_raw      0.756     0.716     0.000
xgb           0.920     0.884     0.000

Host-day TPR, better of P / A:

ds           nf_cic  nf_ctu13  nf_ugr16
detector                               
kitnet        0.800     0.566     0.233
lxgb_hc       1.000     0.739     0.151
mlp_sn_gda    1.000     0.813     0.202
ple_ce        0.800     0.822     0.225
rf            1.000     0.878     0.283
tabm          1.000     0.822     0.256
tabm_kd       1.000     0.687     0.264
tabm_kd_min   1.000     0.720     0.266
tabm_min      1.000     0.831     0.278
tabm_raw      0.933     0.830     0.232
xgb           1.000     0.780     0.269

Deployable threshold (99th pct of calibration hosts): realised test flow FPR

ds           nf_cic  nf_ctu13  nf_ugr16
detector                               
kitnet       0.0229    0.0236    0.2652
lxgb_hc      0.0103    0.0309    0.2458
mlp_sn_gda   0.0152    0.0490    0.0380
ple_ce       0.0112    0.0384    0.2875
rf           0.0106    0.0289    0.2936
tabm         0.0112    0.0394    0.2895
tabm_kd      0.0109    0.0594    0.2902
tabm_kd_min  0.0104    0.0642    0.2903
tabm_min     0.0104    0.0395    0.2901
tabm_raw     0.0096    0.0323    0.2887
xgb          0.0102    0.0275    0.2020

Candidates:

             mean_gap  datasets   day_P   day_A  flood_affected
detector                                                       
lxgb_hc       -0.1586         3  0.5499  0.6295           False
ple_ce        -0.2624         3  0.6003  0.6096           False
tabm          -0.1431         3  0.5593  0.6637           False
tabm_kd       -0.1988         3  0.4964  0.6115           False
tabm_kd_min   -0.1886         3  0.6500  0.6121            True
tabm_min      -0.1310         3  0.7020  0.6809           False
tabm_raw      -0.2253         3  0.5295  0.6649            True

**Selected: tabm_min**, host-day method P

