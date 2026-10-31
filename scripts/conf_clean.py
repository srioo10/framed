"""CONF clean and robust numbers (PREREG §9 CONF, amendment 1): champions A and B on the CONF test pairs (data/conf),
training seeds 2-4, all rows. Per pair x model x seed: fitness / unknown / known TPR at the clean 1%-FPR threshold of
calibration rows, realised FPR on benign test rows, per-family known TPR. A: headline without the input clip, plus
with the cert-mode host-context clip at inference (its robust numbers need it); B: bytes context zeroed as trained.
Robust fitness (IBP over the Theorem 1' deletion box) at m = 1 / 5 / 10 / 50 for B and clipped A. RF / XGB from the
scores conf_pairs.py saved per training seed. Appends results/conf/clean.jsonl as each job ends.
Usage: BOWL_DATASETS=nf_ctu13 .venv/bin/python scripts/conf_clean.py [--arena data/conf] [--workers 4] [--device cpu]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import arena
import bowl

A = {"family": "mlp", "enc": "raw", "e": 2, "w": 64, "depth": 2, "sn": 0, "act": "relu", "members": 1, "head": "energy",
     "kd": 0.5, "syn": 0.5, "ens": 1, "ls": 0.0, "focal": 2.0, "lr": 0.003}
B = {"family": "tabm", "enc": "raw", "e": 4, "w": 32, "depth": 2, "sn": 0, "act": "relu", "members": 8, "head": "nlog",
     "kd": 0.9, "syn": 0.5, "clip": 1, "cert": 4, "gate": 1, "rdet": 2, "ls": 0.0, "focal": 0.0, "lr": 0.003}
MS, SEEDS, EPOCHS = (1, 5, 10, 50), (2, 3, 4), 30


def metrics(z, s):
    """Clean fitness, realised test FPR and per-family known TPR from scores s (cal / pseudo / es)."""
    f, u, k = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
    tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
    ben = z["y_es"] == z["bi"]
    fam = {int(c): float((s["es"][z["y_es"] == c] > tau).mean()) for c in np.unique(z["y_es"][~ben])}
    return {"fitness": f, "unk": u, "known": k, "test_fpr": float((s["es"][ben] > tau).mean()), "fam_tpr": fam}


def job(args):
    key, model, seed, device = args
    torch.set_num_threads(1)
    ds, name = key.split("/")
    bowl.DATASETS = (ds,)
    z = next(p for p in bowl.load_pairs(bowl.ARENA, cheap=False) if p["name"] == name)
    out = {"pair": key, "model": model, "seed": seed}
    if model in ("rf", "xgb"):
        out |= metrics(z, {k: z[f"boss_{model}s{seed}_{k}"] for k in ("cal", "pseudo", "es")})
    elif model == "B":
        z = bowl.drop_bytes([z], {"drop_bytes": 1})[0]
        m, h, s = bowl.run_pair(B, z, device, seed, EPOCHS)
        m.eval()
        out |= metrics(z, s)
        out["robust"] = {mm: bowl.robust_fitness(B, m, z, s["cal"], device, mm) for mm in MS}
    else:  # A, trained without the clip and with the bytes context, as in round 3
        m, h, s = bowl.run_pair(A, z, device, seed, EPOCHS)
        m.eval()
        out |= metrics(z, s)
        lo, hi = bowl.hc_clip(z)  # amendment 1: cert-mode host-context clip at inference only
        m.register_buffer("clo", torch.as_tensor(lo, dtype=torch.float32, device=next(m.parameters()).device))
        m.register_buffer("chi", torch.as_tensor(hi, dtype=torch.float32, device=next(m.parameters()).device))
        sc = {k: h.score(m, z[f"X_{k}"], device) for k in ("cal", "pseudo", "es")}
        out["clip"] = metrics(z, sc)
        out["robust"] = {mm: bowl.robust_fitness(A, m, z, sc["cal"], device, mm) for mm in MS}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "fam_tpr"}),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=Path, default=bowl.ROOT / "data" / "conf")
    ap.add_argument("--models", nargs="+", default=["A", "B", "rf", "xgb"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="clean.jsonl")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    bowl.ARENA = a.arena  # forked workers inherit it
    keys = a.pairs or [f"{p['ds']}/{p['name']}" for p in bowl.load_pairs(bowl.ARENA, cheap=False)]
    out = bowl.ROOT / "results" / "conf"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, mo, sd, a.device) for k in keys for mo in a.models for sd in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex, open(out / a.out, "a") as f:
        for fu in as_completed([ex.submit(job, j) for j in jobs]):  # saved as each job ends: a kill keeps the rest
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
