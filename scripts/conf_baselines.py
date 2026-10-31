"""CONF C5 (PREREG §9 CONF amendment 4): published-IDS baselines on the CONF test pairs, training seeds 2-4, same
calibration rows and 1%-FPR threshold as conf_clean.py. Scores are "higher = attack". Early stopping of the deep
baselines uses a 10% split of the fit rows (never test rows).
kitnet: nids.kitnet.KitNET on benign fit rows. iforest: models.iforest on benign fit rows, score = -score_samples.
horuseye: HorusEye-style two stages (reimplementation): iForest on benign fit rows flags a flow when its score exceeds
the 90th percentile of benign fit scores; flagged flows score 1 + RMSE of an autoencoder (KitNET's _AE, all features)
trained on benign fit rows, unflagged flows score their iForest percentile in [0, 1), so every flagged flow ranks above
every unflagged one. deep_sad: nids.deep_baselines.deep_sad with the fit rows' attack labels. cade: deep_baselines.cade.
Appends results/conf/baselines.jsonl. Usage: BOWL_DATASETS=nf_ctu13 .venv/bin/python scripts/conf_baselines.py
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import bowl
from conf_clean import SEEDS, metrics
from nids import deep_baselines, kitnet, models

MODELS = ["kitnet", "iforest", "horuseye", "deep_sad", "cade"]


def fit(model, z, seed):
    Xf, yf, bi = z["X_fit"].astype(np.float32), z["y_fit"], z["bi"]
    ben = Xf[yf == bi]
    if model == "kitnet":
        return kitnet.KitNET(seed).fit(ben).score
    if model == "iforest":
        f = models.iforest(ben, seed)
        return lambda X: -f.score_samples(X)
    if model == "horuseye":
        f = models.iforest(ben, seed)
        ref = np.sort(-f.score_samples(ben))
        q = ref[int(0.9 * len(ref))]
        rng = np.random.default_rng(seed)
        sub = ben[rng.choice(len(ben), min(len(ben), 200_000), replace=False)].astype(np.float64)
        ae = kitnet._AE(sub.shape[1], rng).fit(sub, rng)

        def score(X):
            s = -f.score_samples(X)
            out = np.searchsorted(ref, s) / (len(ref) + 1.0)
            hit = s > q
            out[hit] = 1.0 + ae.rmse(np.asarray(X[hit], np.float64))
            return out
        return score
    i = np.random.default_rng(seed).permutation(len(Xf))
    tr, va = i[len(i) // 10:], i[:len(i) // 10]
    if model == "deep_sad":
        return deep_baselines.deep_sad(Xf[tr], yf[tr] != bi, Xf[va], yf[va] != bi, seed)
    return deep_baselines.cade(Xf[tr], yf[tr], Xf[va], yf[va], z["k"], seed)


def job(args):
    key, model, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ds, name = key.split("/")
    bowl.DATASETS = (ds,)
    z = next(p for p in bowl.load_pairs(bowl.ARENA, cheap=False) if p["name"] == name)
    t0 = time.time()
    f = fit(model, z, seed)
    t_fit = time.time() - t0
    s = {k: np.asarray(f(z[f"X_{k}"].astype(np.float32)), np.float64) for k in ("cal", "pseudo", "es")}
    Xt = z["X_cal"][:4096].astype(np.float32)
    t0 = time.perf_counter()
    f(Xt)
    out = {"pair": key, "model": model, "seed": seed, **metrics(z, s), "fit_s": t_fit,
           "us_per_flow": 1e6 * (time.perf_counter() - t0) / len(Xt)}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "fam_tpr"}),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=Path, default=bowl.ROOT / "data" / "conf")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="baselines.jsonl")
    a = ap.parse_args()
    bowl.ARENA = a.arena  # forked workers inherit it
    keys = a.pairs or [f"{p['ds']}/{p['name']}" for p in bowl.load_pairs(bowl.ARENA, cheap=False)]
    out = bowl.ROOT / "results" / "conf"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, mo, sd) for k in keys for mo in a.models for sd in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex, open(out / a.out, "a") as f:
        for fu in as_completed([ex.submit(job, j) for j in jobs]):  # saved as each job ends
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
