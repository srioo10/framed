"""K5b (PREREG §9 K5b): champion B checked with branch and bound (fighters.bab_block) on the same 200 rows per pair as
K5. B is trained per arena pair as the death match (seed 1, 30 epochs, bytes context zeroed). Per row: plain alarm,
IBP block, BaB block (proof rule: undecided = not blocked; K5 convention: undecided = blocked).
Writes results/cert/{out} (.csv, and .jsonl as each pair ends).
Usage: .venv/bin/python scripts/k5b_net.py [--pairs nf_ctu13/pair_0 ...] [--workers 4] [--arena DIR] [--out k5b_net.csv]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import bowl
import fighters as fx

B = {"family": "tabm", "enc": "raw", "e": 4, "w": 32, "depth": 2, "sn": 0, "act": "relu", "members": 8, "head": "nlog",
     "kd": 0.9, "syn": 0.5, "clip": 1, "cert": 4, "gate": 1, "rdet": 2, "ls": 0.0, "focal": 0.0, "lr": 0.003}
M, SEED = 10, 1


def job(args):
    key, nrows, time_s, q8, seed, maxb = args
    torch.set_num_threads(1)
    z = next(p for p in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1})
             if f"{p['ds']}/{p['name']}" == key)
    bi = z["bi"]
    rng = np.random.default_rng(0)  # the K5 rows (k5_trees.job draws the same way)
    att = z["y_es"] != bi
    sub = lambda X: X[np.sort(rng.choice(len(X), min(nrows, len(X)), replace=False))]  # noqa: E731
    Xu, Xk = sub(z["X_pseudo"]), sub(z["X_es"][att]) if att.any() else None
    m, h, s = bowl.run_pair(B, z, "cpu", seed, 30)
    m.eval()
    if q8:  # PREREG INT8-CONF: the deployed int8 weights (same function as the export), tau recalibrated on them
        from engine_export_b import quantise
        quantise(m)
        s = {"cal": h.score(m, z["X_cal"], "cpu")}
    tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
    out = {"pair": key, "seed": seed, "q8": bool(q8), "tau": float(tau), "robust_full": bowl.robust_fitness(B, m, z, s["cal"], "cpu", M)[0]}
    for part, X in (("unk", Xu), ("known", Xk)):
        if X is None:
            continue
        lo, hi = bowl.dbox(z, X, M)
        plain = h.score(m, X, "cpu") > tau
        ibp = fx.score_lower(m, lo, hi, bi, B["head"]) > tau
        blk, und = ibp.copy(), np.zeros(len(X), bool)
        for i in np.flatnonzero(plain & ~ibp):
            blk[i], decided = fx.bab_block(m, h, lo[i], hi[i], bi, B["head"], tau, time_s=time_s, max_boxes=maxb)
            und[i] = not decided
        out |= {f"plain_{part}": plain.mean(), f"ibp_{part}": ibp.mean(), f"bab_{part}": blk.mean(),
                f"babk5_{part}": (blk | und).mean(), f"undecided_{part}": int(und.sum()), f"n_{part}": len(X)}
    if Xk is None:  # no known attack rows in this pair: known = unknown, as in k5_trees
        for q in ("plain", "ibp", "bab", "babk5"):
            out[f"{q}_known"] = out[f"{q}_unk"]
    for q in ("plain", "ibp", "bab", "babk5"):
        out[q] = (out[f"{q}_unk"] + out[f"{q}_known"]) / 2
    print(json.dumps({q: (round(v, 3) if isinstance(v, float) else v) for q, v in out.items()}), flush=True)
    return out


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--max-boxes", type=float, default=8192)  # LONGMILP: 1e12 so the time limit binds
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--out", default="k5b_net.csv")
    ap.add_argument("--arena", type=Path, default=bowl.ARENA)
    ap.add_argument("--seed", type=int, default=SEED)  # CONF: training seeds 2-4; rows stay RNG seed 0
    ap.add_argument("--q8", action="store_true")  # score the int8-rounded model, as deployed
    a = ap.parse_args()
    SEED = a.seed
    bowl.ARENA = a.arena  # forked workers inherit it
    keys = a.pairs or [f"{p['ds']}/{p['name']}" for p in bowl.load_pairs(bowl.ARENA, cheap=False)]
    out = bowl.ROOT / "results" / "cert"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex, open(out / a.out.replace(".csv", ".jsonl"), "a") as f:
        for fu in as_completed([ex.submit(job, (k, a.rows, a.time, a.q8, a.seed, int(a.max_boxes))) for k in keys]):
            rows.append(fu.result())
            f.write(json.dumps(rows[-1]) + "\n")
            f.flush()
    R = pd.DataFrame(rows)
    R.to_csv(out / a.out, index=False)
    print(R[["robust_full", "plain", "ibp", "bab", "babk5", "undecided_unk", "undecided_known"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
