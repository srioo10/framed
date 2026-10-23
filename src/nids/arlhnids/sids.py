"""Signature-based IDS (§3.7.1, Algorithm 3): seven tree models, hard majority vote.

Models are fitted one by one and voted by hand rather than through sklearn's VotingClassifier, because
XGBoost's early stopping (Table 13) needs an eval set that VotingClassifier cannot pass. The vote is the
same: most frequent class, ties to the lowest class index."""
import time

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build(params, seed):
    """Table 13 settings on current library versions:
    ET max_features "auto" (removed in sklearn 1.3) = "sqrt" for classifiers; AdaBoost "SAMME.R" was removed
    in sklearn 1.6, so every AdaBoost runs SAMME; LightGBM min_data_in_leaf = min_child_samples;
    XGBoost eta/lambda/alpha = learning_rate/reg_lambda/reg_alpha."""
    p = {k: dict(v) for k, v in params.items()}
    if p["ET"].get("max_features") == "auto":
        p["ET"]["max_features"] = "sqrt"
    p["AB"].pop("algorithm", None)
    p["LGBM"]["min_child_samples"] = p["LGBM"].pop("min_data_in_leaf")
    p["XGB"]["learning_rate"] = p["XGB"].pop("eta")
    return {
        "DT": DecisionTreeClassifier(random_state=seed, **p["DT"]),
        "RF": RandomForestClassifier(random_state=seed, n_jobs=-1, **p["RF"]),
        "ET": ExtraTreesClassifier(random_state=seed, n_jobs=-1, **p["ET"]),
        "AB": AdaBoostClassifier(random_state=seed, **p["AB"]),
        "LGBM": LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **p["LGBM"]),
        "CB": CatBoostClassifier(random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False, **p["CB"]),
        "XGB": XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **p["XGB"]),
    }


def fit_predict(models, Xtr, ytr, Xte, Xes, yes, k):
    """Fit every model, predict Xte, hard-vote. (Xes, yes) is XGBoost's early-stopping set.
    Returns (vote, {model: predictions}, {model: fit seconds})."""
    preds, secs = {}, {}
    for name, m in models.items():
        t = time.time()
        if name == "XGB":
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
        else:
            m.fit(Xtr, ytr)
        secs[name] = time.time() - t
        preds[name] = np.asarray(m.predict(Xte)).reshape(-1).astype(np.int64)
    votes = np.zeros((len(Xte), k), dtype=np.int32)
    for p in preds.values():
        votes[np.arange(len(p)), p] += 1
    return votes.argmax(1), preds, secs
