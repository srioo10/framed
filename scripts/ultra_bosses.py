"""Ultra bosses (PREREG §9 BOWL amendment): heavier reference detectors than the arena's RF/XGB, never fighters.
Per arena pair, fit on the pair's fit rows only and write data/arena/{ds}/ultra_{i}.npz with boss_{name}_{es,pseudo,cal}
(-log P(benign), same convention as arena.py), which bowl.load_pairs merges into the pair.
  cat   CatBoost 1000 iters, depth 8           lgbm  LightGBM 1000 trees, 255 leaves
  xgbx  XGB 1000 trees, depth 10, lr 0.05      gang  mean P(benign) of rf, xgb, cat, lgbm, xgbx
Runs on Kaggle as its own kernel (GPU used when present) or locally.
Usage: .venv/bin/python scripts/ultra_bosses.py [--arena data/arena] [--out data/arena]
"""
import argparse
import glob
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("nf_ctu13", "nf_ugr16", "nf_cic")


def models(gpu, seed=1):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {"cat": CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.1, random_seed=seed, verbose=0,
                                      task_type="GPU" if gpu else "CPU", thread_count=-1),
            "lgbm": LGBMClassifier(n_estimators=1000, num_leaves=255, learning_rate=0.05, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1),
            "xgbx": XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", device="cuda" if gpu else "cpu",
                                  n_jobs=-1, random_state=seed)}


def nlog(p):
    return -np.log(np.maximum(p, 1e-45)).astype(np.float32)


def run(arena, out, gpu):
    for ds in (d for d in DATASETS if (arena / d).exists()):
        (out / ds).mkdir(parents=True, exist_ok=True)
        for f in sorted((arena / ds).glob("pair_*.npz"), key=lambda f: int(f.stem.split("_")[1])):
            dest = out / ds / f"ultra_{f.stem.split('_')[1]}.npz"
            if dest.exists():
                continue
            t0 = time.time()
            z = np.load(f)
            bi, res = int(z["bi"]), {}
            for name, m in models(gpu).items():
                m.fit(z["X_fit"], z["y_fit"])
                for k in ("es", "pseudo", "cal"):
                    res[f"boss_{name}_{k}"] = nlog(m.predict_proba(z[f"X_{k}"])[:, bi])
            for k in ("es", "pseudo", "cal"):
                P = [np.exp(-z[f"boss_{b}_{k}"].astype(np.float64)) for b in ("rf", "xgb")] + \
                    [np.exp(-res[f"boss_{b}_{k}"].astype(np.float64)) for b in ("cat", "lgbm", "xgbx")]
                res[f"boss_gang_{k}"] = nlog(np.mean(P, axis=0))
            np.savez_compressed(dest, **res)
            print(f"{ds} {f.stem}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    kaggle = Path("/kaggle/input").exists()
    if a.arena is None:  # on Kaggle the arena sits inside the bigp-arena dataset mount
        a.arena = Path(glob.glob("/kaggle/input/**/arena/nf_ctu13", recursive=True)[0]).parent if kaggle \
            else ROOT / "data" / "arena"
    a.out = a.out or (Path("/kaggle/working/ultra") if kaggle else a.arena)
    run(a.arena, a.out, gpu=shutil.which("nvidia-smi") is not None)
