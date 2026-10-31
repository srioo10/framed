"""BSAN (PREREG §9): threshold-free sanity check of the published baselines on every CONF pair, training seed 2.

Per pair and model: ROC AUC of test benign vs unknown (pseudo) rows and vs known attack rows, TPR at 1 / 5 / 10%
calibration FPR, plus the RF / XGB boss AUCs. Writes results/conf/baseline_sanity.csv.
Usage: .venv/bin/python scripts/baseline_sanity.py [--workers 4]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import bowl
import conf_baselines as cb

SEED = 2
MODELS = cb.MODELS + ["kitnet_online", "horuseye_online"]


def evaluate(s_cal, s_unk, s_es, y_es, bi):
    ben, att = s_es[y_es == bi], s_es[y_es != bi]
    auc = lambda a: roc_auc_score(np.r_[np.zeros(len(ben)), np.ones(len(a))], np.r_[ben, a])  # noqa: E731
    r = {"auc_unk": auc(s_unk), "auc_known": auc(att)}
    for q in (1, 5, 10):
        tau = np.quantile(s_cal, 1 - q / 100)
        r[f"tpr_unk@{q}"], r[f"tpr_known@{q}"] = (s_unk > tau).mean(), (att > tau).mean()
    return r


def job(key):
    torch.set_num_threads(1)
    ds, name = key.split("/")
    bowl.DATASETS = (ds,)
    z = next(p for p in bowl.load_pairs(bowl.ARENA, cheap=False) if p["name"] == name)
    out = [{"pair": key, "model": f"{b}(boss)", **evaluate(*(z[f"boss_{b}s{SEED}_{k}"] for k in ("cal", "pseudo", "es")),
                                                            z["y_es"], z["bi"])} for b in ("rf", "xgb")]
    for m in MODELS:
        torch.manual_seed(SEED)
        f = cb.fit(m, z, SEED)
        s = [np.asarray(f(z[f"X_{k}"].astype(np.float32)), np.float64) for k in ("cal", "pseudo", "es")]
        out.append({"pair": key, "model": m, **evaluate(*s, z["y_es"], z["bi"])})
        print(out[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arena", type=Path, default=bowl.ROOT / "data" / "conf")
    a = ap.parse_args()
    bowl.ARENA = a.arena
    keys = [f"{p['ds']}/{p['name']}" for p in bowl.load_pairs(bowl.ARENA, cheap=False)]
    with ProcessPoolExecutor(a.workers) as ex:
        R = pd.DataFrame([r for rs in ex.map(job, keys) for r in rs])
    assert R[["auc_unk", "auc_known"]].stack().between(0, 1).all()
    R.to_csv(bowl.ROOT / "results" / "conf" / "baseline_sanity.csv", index=False)
    R["ds"] = R.pair.str.split("/").str[0]
    print(R.groupby(["model", "ds"])[["auc_unk", "auc_known", "tpr_unk@1", "tpr_known@1"]].mean().round(3).to_string())
    print((R.assign(lt=R.auc_unk < 0.5).groupby("model").lt.sum()).to_string())


if __name__ == "__main__":
    main()
