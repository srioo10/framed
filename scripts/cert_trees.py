"""Box certificate for the tree bosses (PREREG §9 CERT K3): the same Theorem 1 boxes as the networks, bounded per tree.

Per tree, the leaves reachable from a box [lo, hi] (XGB: go left if lo_f < thr, right if hi_f >= thr; sklearn: <= / >); trees are bounded
independently, which is sound (looser than a joint bound). XGB: upper(z_c - z_bi) = sum over class-c trees of the max
reachable leaf - sum over benign trees of the min. RF: P(benign) >= mean over trees of the min reachable leaf share.
Both give an upper bound on s = -log P(benign). Bosses are refit per arena pair with arena.py's configs and checked
against the stored boss scores before use.
Usage: .venv/bin/python scripts/cert_trees.py [--rows 5000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import sensitivity  # noqa: E402

MS = (1, 5, 10, 50, 100)
OUT = ROOT / "results" / "cert"


VISITS = None  # PREREG §9 K8: set to np.zeros(n_rows) to count node visits per row (one comparison each)


def reach(left, right, feat, thr, leaf_val, lo, hi, le=False):
    """Per row: (min, max) of leaf_val over the leaves reachable from the box. Nodes with left < 0 are leaves.
    le: the tree sends x <= thr left (sklearn); otherwise x < thr (XGB)."""
    n = len(lo)
    mn, mx = np.full(n, np.inf), np.full(n, -np.inf)
    stack = [(0, np.arange(n))]
    while stack:
        node, r = stack.pop()
        if VISITS is not None:
            VISITS[r] += 1
        if left[node] < 0:
            mn[r] = np.minimum(mn[r], leaf_val[node])
            mx[r] = np.maximum(mx[r], leaf_val[node])
            continue
        f, t = feat[node], thr[node]
        a, b = (r[lo[r, f] <= t], r[hi[r, f] > t]) if le else (r[lo[r, f] < t], r[hi[r, f] >= t])
        if len(a):
            stack.append((left[node], a))
        if len(b):
            stack.append((right[node], b))
    return mn, mx


def rf_upper(rf, lo, hi, bi):
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)  # sklearn compares in float32
    pmin = np.zeros(len(lo))
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        share = v[:, bi] / v.sum(1)
        pmin += reach(t.children_left, t.children_right, t.feature, t.threshold, share, lo32, hi32, le=True)[0]
    return -np.log(np.maximum(pmin / len(rf.estimators_), 1e-45))


def rf_lower(rf, lo, hi, bi):
    """Lower bound of s: P(benign) <= mean over trees of the max reachable leaf share (robust-alarm gating)."""
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    pmax = np.zeros(len(lo))
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        pmax += reach(t.children_left, t.children_right, t.feature, t.threshold, v[:, bi] / v.sum(1), lo32, hi32,
                      le=True)[1]
    return -np.log(np.maximum(pmax / len(rf.estimators_), 1e-45))


def xgb_upper(xg, lo, hi, bi, k, lower=False):
    df = xg.get_booster().trees_to_dataframe()
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    lo_m, hi_m = np.zeros((len(lo), k)), np.zeros((len(lo), k))
    for tid, T in df.groupby("Tree", sort=True):
        T = T.reset_index(drop=True)
        pos = {i: j for j, i in enumerate(T.ID)}
        leaf = T.Feature.to_numpy() == "Leaf"
        left = np.array([-1 if l else pos[y] for l, y in zip(leaf, T.Yes)])
        right = np.array([-1 if l else pos[n] for l, n in zip(leaf, T.No)])
        feat = np.array([0 if l else int(f[1:]) for l, f in zip(leaf, T.Feature)])
        thr = np.where(leaf, 0, T.Split.fillna(0)).astype(np.float32)
        mn, mx = reach(left, right, feat, thr, T.Gain.to_numpy(), lo32, hi32)
        c = tid % k
        lo_m[:, c] += mn
        hi_m[:, c] += mx
    x0 = np.zeros((1, lo.shape[1]), np.float32)
    off = np.atleast_1d(xg.predict(x0, output_margin=True)[0]) - leaf_sums(df, x0, k)  # per-class base margins
    if len(off) == 1:  # binary: one margin m for class 1, P(1) = sigmoid(m)
        mn, mx = lo_m.sum(1) + off[0], hi_m.sum(1) + off[0]  # trees were spread over columns tid % k
        if lower:
            return np.logaddexp(0, mn) if bi == 0 else np.logaddexp(0, -mx)
        return np.logaddexp(0, mx) if bi == 0 else np.logaddexp(0, -mn)
    d = (lo_m + off - (hi_m[:, [bi]] + off[bi])) if lower else (hi_m + off - (lo_m[:, [bi]] + off[bi]))
    d[:, bi] = 0
    return np.logaddexp.reduce(d, axis=1)  # -log P_lo(benign) = log sum_c exp(upper(z_c - z_bi))


def leaf_sums(df, x, k):
    """Per-class sum of leaf values reached by the single point x (1 entry for a binary model)."""
    nt = df.Tree.max() + 1
    kk = k if nt % k == 0 and k > 2 else 1
    tot = np.zeros(kk)
    for tid, T in df.groupby("Tree", sort=True):
        T = T.reset_index(drop=True)
        pos = {i: j for j, i in enumerate(T.ID)}
        j = 0
        while T.Feature[j] != "Leaf":
            j = pos[T.Yes[j]] if np.float32(x[0, int(T.Feature[j][1:])]) < np.float32(T.Split[j]) else pos[T.No[j]]
        tot[tid % kk] += T.Gain[j]
    return tot


def cert_fpr(s_clean, s_up):
    """Threshold: smallest score with clean share above it <= 1%; certified FPR = share of upper bounds above it."""
    tau = np.sort(s_clean)[::-1][int(np.floor(0.01 * len(s_clean)))]
    return float((s_clean > tau).mean()), float((s_up > tau).mean()), tau


def main():
    import arena
    from nids import models
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    a = ap.parse_args()
    rows = []
    for ds in ("nf_ctu13", "nf_ugr16", "nf_cic"):
        for f in sorted((arena.OUT / ds).glob("pair_*.npz"), key=lambda f: int(f.stem.split("_")[1])):
            z, meta = np.load(f), json.loads(f.with_suffix(".json").read_text())
            bi, k = int(z["bi"]), int(z["teacher"].shape[1])
            i = np.sort(np.random.default_rng(0).choice(len(z["X_cal"]), min(a.rows, len(z["X_cal"])), replace=False))
            Xc = z["X_cal"][i]
            for name, m in (("rf", models.rf(arena.SEED)), ("xgb", models.xgb(arena.SEED))):
                m.fit(z["X_fit"], z["y_fit"])
                s = -np.log(np.maximum(m.predict_proba(Xc)[:, bi], 1e-45))
                assert np.allclose(s, z[f"boss_{name}_cal"][i], atol=1e-3), f"{ds} {f.stem} {name}: refit differs"
                up = (lambda lo, hi: rf_upper(m, lo, hi, bi)) if name == "rf" else \
                    (lambda lo, hi: xgb_upper(m, lo, hi, bi, k))
                assert np.allclose(up(Xc, Xc), s, atol=1e-3), f"{name}: zero-width box != exact score"
                for mm in MS:
                    lo, hi = sensitivity.box(Xc, meta["cols"], np.array(meta["mu"]), np.array(meta["sd"]), mm)
                    clean, cert, tau = cert_fpr(s, up(lo, hi))
                    rows.append({"ds": ds, "pair": f.stem, "model": name, "m": mm, "fpr_clean": clean,
                                 "fpr_cert": cert, "tau": tau, "rows": len(Xc)})
                    print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "k3_trees.csv", index=False)
    print(R.groupby(["model", "ds", "m"]).fpr_cert.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
