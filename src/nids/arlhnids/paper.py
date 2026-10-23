"""Settings reported in ARLHNIDS-IoT (Srivastava & Sinha 2025), keyed by our loader names.
Table numbers refer to the paper. Nothing here is tuned by us."""

DATASETS = ["unsw_official", "bot_iot", "ciciot_paper"]

# Columns the paper drops before modelling (§3.3: "id" and "label"); everything else is a feature.
# For BoT-IoT that keeps `attack` and `subcategory`: Table 11 gives 18 feature dimensions, which is
# exactly the 10-best file's 19 columns minus the target `category`.
PAPER_DROP = {"unsw_official": ["id", "label"], "bot_iot": [], "ciciot_paper": []}

# Table 8: FCM hyperparameters (c, m, error, maxiter)
FCM = {
    "unsw_official": dict(c=166, m=1.013, error=0.0005, maxiter=1025),
    "bot_iot": dict(c=183, m=1.098, error=0.001, maxiter=1156),
    "ciciot_paper": dict(c=210, m=1.199, error=0.005, maxiter=1200),
}

# Table 10: class sizes after FCM undersampling (majority, Table 9) and GAMO oversampling (minority)
BALANCED = {
    "unsw_official": {"Normal": 80912, "Generic": 31204, "Exploits": 44525, "Fuzzers": 39455, "DoS": 37482,
                      "Reconnaissance": 36889, "Analysis": 34070, "Backdoor": 33976, "Shellcode": 33771,
                      "Worms": 33437},
    "bot_iot": {"DDoS": 92000, "DoS": 92000, "Reconnaissance": 91082, "Normal": 73026, "Theft": 72933},
    "ciciot_paper": {"DDoS": 30439, "Normal": 90002, "Spoofing": 59854, "DoS": 55977, "Mirai": 50872,
                     "Brute Force": 50358, "Web-Based": 48444, "Recon": 48277},
}

# Table 5: majority classes under Eq. 2
MAJORITY = {"unsw_official": {"Normal", "Generic"}, "bot_iot": {"DDoS", "DoS"}, "ciciot_paper": {"DDoS", "Normal"}}

# Table 11 (Algorithm 2 lists 50 wolves / 10 iterations; Table 11 is per dataset and is used here)
GWO = {
    "unsw_official": dict(pop=20, iters=86),
    "bot_iot": dict(pop=50, iters=95),
    "ciciot_paper": dict(pop=30, iters=112),
}
GWO_SEED, GWO_ALPHA, GWO_MAX_CORR = 40, 0.5, 0.6

# Table 12: number of features GWO kept
GWO_KEPT = {"unsw_official": 25, "bot_iot": 10, "ciciot_paper": 21}

# Table 13: tuned SIDS hyperparameters (paper's names; sids.py maps them to current library APIs)
SIDS = {
    "unsw_official": {
        "DT": dict(splitter="random", criterion="gini", max_depth=12, min_samples_split=10, min_samples_leaf=14,
                   min_weight_fraction_leaf=2.3e-04, min_impurity_decrease=7.28e-05, max_leaf_nodes=29),
        "RF": dict(n_estimators=128, max_depth=16, min_samples_split=24, min_samples_leaf=10),
        "ET": dict(n_estimators=950, criterion="gini", max_depth=12, min_samples_split=10, min_samples_leaf=3,
                   max_features="sqrt", bootstrap=False),
        "AB": dict(n_estimators=300, learning_rate=0.579, algorithm="SAMME"),
        "LGBM": dict(learning_rate=0.007, n_estimators=287, num_leaves=35, max_depth=5, min_data_in_leaf=47),
        "CB": dict(learning_rate=0.0069, iterations=284, depth=8, l2_leaf_reg=0.80908),
        "XGB": dict(booster="dart", reg_lambda=0.004, reg_alpha=2.6e-04, subsample=0.367, colsample_bytree=0.923,
                    early_stopping_rounds=12, n_estimators=16, max_depth=7, min_child_weight=10, eta=0.1163,
                    gamma=4.57e-05, grow_policy="lossguide"),
    },
    "bot_iot": {
        "DT": dict(splitter="best", criterion="gini", max_depth=5, min_samples_split=9, min_samples_leaf=14,
                   min_weight_fraction_leaf=1.76e-08, min_impurity_decrease=8.42e-06, max_leaf_nodes=11),
        "RF": dict(n_estimators=87, max_depth=13, min_samples_split=32, min_samples_leaf=9),
        "ET": dict(n_estimators=866, criterion="entropy", max_depth=19, min_samples_split=16, min_samples_leaf=4,
                   max_features="auto", bootstrap=False),
        "AB": dict(n_estimators=225, learning_rate=0.310, algorithm="SAMME.R"),
        "LGBM": dict(learning_rate=0.0015, n_estimators=929, num_leaves=40, max_depth=5, min_data_in_leaf=20),
        "CB": dict(learning_rate=0.03349, iterations=881, depth=7, l2_leaf_reg=0.2058),
        "XGB": dict(booster="gbtree", reg_lambda=2.95e-05, reg_alpha=2.80e-08, subsample=0.3378,
                    colsample_bytree=0.586, early_stopping_rounds=24, n_estimators=32, max_depth=5,
                    min_child_weight=7, eta=0.7282, gamma=0.0020, grow_policy="depthwise"),
    },
    "ciciot_paper": {
        "DT": dict(splitter="best", criterion="entropy", max_depth=14, min_samples_split=10, min_samples_leaf=16,
                   min_weight_fraction_leaf=1.98e-04, min_impurity_decrease=0.0038, max_leaf_nodes=8),
        "RF": dict(n_estimators=68, max_depth=6, min_samples_split=12, min_samples_leaf=7),
        "ET": dict(n_estimators=905, criterion="log_loss", max_depth=14, min_samples_split=20, min_samples_leaf=5,
                   max_features="log2", bootstrap=True),
        "AB": dict(n_estimators=427, learning_rate=0.110, algorithm="SAMME.R"),
        "LGBM": dict(learning_rate=0.373, n_estimators=779, num_leaves=36, max_depth=9, min_data_in_leaf=34),
        "CB": dict(learning_rate=0.0457, iterations=536, depth=9, l2_leaf_reg=2.3903),
        "XGB": dict(booster="gbtree", reg_lambda=1.76e-04, reg_alpha=1.47e-03, subsample=0.792,
                    colsample_bytree=0.937, early_stopping_rounds=22, n_estimators=64, max_depth=9,
                    min_child_weight=10, eta=1.19e-06, gamma=1.88e-07, grow_policy="depthwise"),
    },
}

# Headline SIDS accuracy (abstract, Tables 16/18/20)
SIDS_ACCURACY = {"unsw_official": 0.9289, "bot_iot": 0.9919, "ciciot_paper": 0.9831}
