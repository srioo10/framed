"""Checking cost (PREREG §9 item 2): per-flow wall time, one row at a time on one core, to decide "certified block or
not" over the Theorem 1' deletion box at m = 10: gated net (champion B, IBP), RF / XGB per-tree bounds, RF / XGB
exact MILP (k5_trees, 20 s limit). Rows: pair --pair of each dataset, unknown + known attack rows (seed 0).
Writes results/cert/check_cost.csv.
Batch time (all rows in one call / rows) is reported too, since the tree bounds are vectorised Python.
Usage: OMP_NUM_THREADS=1 taskset -c 7 .venv/bin/python scripts/check_cost.py [--rows 1000] [--milp-rows 100]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import bowl
import cert_trees as ct
import fighters as fx
import k5_trees as k5
from nids import models

B = {"family": "tabm", "enc": "raw", "e": 4, "w": 32, "depth": 2, "sn": 0, "act": "relu", "members": 8, "head": "nlog",
     "kd": 0.9, "syn": 0.5, "clip": 1, "cert": 4, "gate": 1, "rdet": 2, "ls": 0.0, "focal": 0.0, "lr": 0.003}


def per_row(f, n):
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        f(i)
        t.append(time.perf_counter() - t0)
    return np.array(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--milp-rows", type=int, default=100)
    ap.add_argument("--pair", default="pair_1")
    ap.add_argument("--ds", default=None)  # one dataset per process (one core each)
    a = ap.parse_args()
    torch.set_num_threads(1)
    res = []
    for z in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1}):
        if z["name"] != a.pair or (a.ds and z["ds"] != a.ds):
            continue
        bi, k, rng = z["bi"], z["k"], np.random.default_rng(0)
        att = z["y_es"] != bi
        X = np.vstack([z["X_pseudo"], z["X_es"][att]])
        X = X[np.sort(rng.choice(len(X), min(a.rows, len(X)), replace=False))]
        lo, hi = bowl.dbox(z, X, 10)
        m, _, _ = bowl.run_pair(B, z, "cpu", 1, 30)
        m.eval()
        rf = models.rf(1).set_params(n_jobs=1).fit(z["X_fit"], z["y_fit"])
        xg = models.xgb(1).set_params(n_jobs=1).fit(z["X_fit"], z["y_fit"])
        cr, cx = list(rf.classes_).index(bi), list(xg.classes_).index(bi)
        n = len(X)
        with torch.no_grad():
            r1 = lambda i: slice(i, i + 1) if isinstance(i, int) else i  # noqa: E731
            fn = {"net_ibp": lambda i: fx.score_lower(m, lo[r1(i)], hi[r1(i)], bi, "nlog"),
                  "rf_pertree": lambda i: ct.rf_lower(rf, lo[r1(i)], hi[r1(i)], cr),
                  "xgb_pertree": lambda i: ct.xgb_upper(xg, lo[r1(i)], hi[r1(i)], cx, k, lower=True)}
            for name, f in fn.items():
                t = per_row(f, n)
                t0 = time.perf_counter()
                f(slice(None))  # whole batch at once: per-flow cost when vectorised (fairer to the Python tree bounds)
                res.append({"ds": z["ds"], "pair": z["name"], "method": name, "rows": n, "median_ms": 1e3 * np.median(t),
                            "p99_ms": 1e3 * np.quantile(t, 0.99), "batch_ms": 1e3 * (time.perf_counter() - t0) / n,
                            "undecided": 0.0})
                print(json.dumps(res[-1]), flush=True)
        for name, mod, col in (("rf_milp", rf, cr), ("xgb_milp", xg, cx)):
            T, off = k5.trees(mod, k, col)
            nm, und = min(a.milp_rows, n), []
            t = per_row(lambda i: und.append(not k5.milp_bound(T, off, lo[i], hi[i], X[i], col, k, name == "rf_milp")[2]), nm)
            res.append({"ds": z["ds"], "pair": z["name"], "method": name, "rows": nm, "median_ms": 1e3 * np.median(t),
                        "p99_ms": 1e3 * np.quantile(t, 0.99), "undecided": float(np.mean(und))})
            print(json.dumps(res[-1]), flush=True)
        pd.DataFrame(res).to_csv(bowl.ROOT / "results" / "cert" / f"check_cost{'_' + a.ds if a.ds else ''}.csv", index=False)


if __name__ == "__main__":
    main()
