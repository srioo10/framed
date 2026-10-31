"""K5 (PREREG §9 CERT K5 row): is the gated-tree comparison fair? Gated RF / XGB under an exact MILP verifier
(Kantchelian et al. 2016 style) over the Theorem 1' deletion box, plain and robust-trained (fit rows plus 2 uniform
samples per row from its box at m: benign rows the injection box, attack rows the deletion box).

Per row: block if the proven lower bound of s = -log P(benign) over the box > tau; not block if a point realised from
the MILP solution scores <= tau on the real model; otherwise undecided, counted as blocked (favours the trees).
RF and binary XGB: max P(benign) is exact. Multiclass XGB: min over the box of max_c (z_c - z_bi), so
s >= log1p(exp(.)) (sound, not exact). MILP only where the plain alarm fires and the per-tree bound does not block.
Writes results/cert/k5_trees.csv.
Usage: .venv/bin/python scripts/k5_trees.py [--rows 100] [--workers 8] [--pairs nf_ctu13/pair_0 ...]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bowl  # noqa: E402
import cert_trees as ct  # noqa: E402
from nids import models  # noqa: E402

M, SEED, LIMIT = 10, 1, 20.0


def trees(m, k, bi):
    """Common form: per tree (left, right, feat, thr, value, cls, le). RF value = benign leaf share."""
    if hasattr(m, "estimators_"):
        out = []
        for est in m.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]
            out.append((t.children_left, t.children_right, t.feature, t.threshold.astype(np.float32),
                        v[:, bi] / v.sum(1), 0, True))
        return out, None
    df = m.get_booster().trees_to_dataframe()
    out = []
    for tid, T in df.groupby("Tree", sort=True):
        T = T.reset_index(drop=True)
        pos = {i: j for j, i in enumerate(T.ID)}
        leaf = T.Feature.to_numpy() == "Leaf"
        left = np.array([-1 if lf else pos[y] for lf, y in zip(leaf, T.Yes)])
        right = np.array([-1 if lf else pos[n] for lf, n in zip(leaf, T.No)])
        feat = np.array([0 if lf else int(f[1:]) for lf, f in zip(leaf, T.Feature)])
        thr = np.where(leaf, 0, T.Split.fillna(0)).astype(np.float32)
        out.append((left, right, feat, thr, T.Gain.to_numpy(), tid % k if k > 2 else 0, False))
    x0 = np.zeros((1, m.n_features_in_), np.float32)
    off = np.atleast_1d(m.predict(x0, output_margin=True)[0]) - ct.leaf_sums(df, x0, k)
    return out, off


def leaves(tr, lo, hi):
    """Reachable leaves of one tree from the box with the active split conditions (feature, thr, goes_left)."""
    left, right, feat, thr, _, _, le = tr
    out, stack = [], [(0, ())]
    while stack:
        node, cond = stack.pop()
        if left[node] < 0:
            out.append((node, cond))
            continue
        f, t = feat[node], thr[node]
        a, b = (lo[f] <= t, hi[f] > t) if le else (lo[f] < t, hi[f] >= t)
        if a and b:
            stack += [(left[node], cond + ((f, t, True),)), (right[node], cond + ((f, t, False),))]
        elif a:
            stack.append((left[node], cond))
        else:
            stack.append((right[node], cond))
    return out


def realise(x, preds, y, le):
    """A point in the box consistent with the predicate values y (pred = 1: x < thr, or <= thr if le)."""
    x = x.copy()
    for f in {f for f, _ in preds}:
        ts = sorted((t, y[i]) for i, (g, t) in enumerate(preds) if g == f)
        ones = [t for t, v in ts if v > 0.5]
        zeros = [t for t, v in ts if v <= 0.5]
        a, b = (max(zeros) if zeros else -np.inf), (min(ones) if ones else np.inf)
        if le:  # need a < x <= b
            x[f] = b if np.isfinite(b) else np.nextafter(np.float32(a), np.float32(np.inf))
        else:  # need a <= x < b
            x[f] = a if np.isfinite(a) else np.nextafter(np.float32(b), np.float32(-np.inf))
    return x


def milp_bound(T, off, lo, hi, x, bi, k, rf):
    """(lower bound of s over the box, s at a realised point or None)."""
    L = [(i, node, cond) for i, tr in enumerate(T) for node, cond in leaves(tr, lo, hi)]
    preds = sorted({(f, t) for _, _, cond in L for f, t, _ in cond})
    pid = {p: j for j, p in enumerate(preds)}
    nP, nL = len(preds), len(L)
    multi = not rf and k > 2
    nv = nP + nL + (1 if multi else 0)
    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(entries, lo_, hi_):
        r = len(lb)
        for c, v in entries:
            rows.append(r), cols.append(c), vals.append(v)
        lb.append(lo_), ub.append(hi_)
    per = {}
    for j, (ti, _, _) in enumerate(L):
        per.setdefault(ti, []).append((nP + j, 1.0))
    for e in per.values():  # one leaf per tree
        add(e, 1, 1)
    for j, (_, _, cond) in enumerate(L):  # leaf j only if its path conditions hold
        for f, t, goes_left in cond:
            add([(nP + j, 1.0), (pid[(f, t)], -1.0 if goes_left else 1.0)], -np.inf, 0 if goes_left else 1)
    for a, b in zip(preds, preds[1:]):  # x < t1 implies x < t2 for t1 < t2
        if a[0] == b[0]:
            add([(pid[a], 1.0), (pid[b], -1.0)], -np.inf, 0)
    val = np.array([T[i][4][node] for i, node, _ in L])
    cls = np.array([T[i][5] for i, _, _ in L])
    c = np.zeros(nv)
    if rf:
        c[nP:nP + nL] = -val / len(T)  # min -P(benign)
    elif not multi:
        c[nP:nP + nL] = val if bi == 0 else -val  # bi 0: s increasing in margin -> min margin
    else:
        c[-1] = 1.0  # min t, t >= z_c - z_bi for every c != bi
        for cc in range(k):
            if cc != bi:
                e = [(nP + j, val[j] * (int(cls[j] == cc) - int(cls[j] == bi))) for j in range(nL) if cls[j] in (cc, bi)]
                add(e + [(nv - 1, -1.0)], -np.inf, -(off[cc] - off[bi]))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), nv))
    integ = np.r_[np.ones(nP), np.zeros(nL), np.zeros(1 if multi else 0)]
    bnd = (np.r_[np.zeros(nP + nL), [-np.inf] if multi else []], np.r_[np.ones(nP + nL), [np.inf] if multi else []])
    r = milp(c, integrality=integ, bounds=Bounds(*bnd), constraints=LinearConstraint(A, lb, ub),
             options={"time_limit": LIMIT, "disp": False})
    dual = getattr(r, "mip_dual_bound", None)
    if dual is None or not np.isfinite(dual):
        dual = r.fun if r.status == 0 else -np.inf
    if rf:
        s_lo = -np.log(np.clip(-dual, 1e-45, 1.0))
    elif not multi:
        mrg = dual + off[0] if bi == 0 else -dual + off[0]  # bi 0: min margin; bi 1: max margin
        s_lo = np.logaddexp(0, mrg) if bi == 0 else np.logaddexp(0, -mrg)
    else:
        s_lo = np.logaddexp(0, dual)
    xr = realise(x, preds, r.x[:nP], T[0][6]) if r.x is not None else None
    return float(s_lo), xr


def augment(z, rng):
    """Fit rows plus 2 uniform samples per row from its box at M (benign: injection box, attack: deletion box)."""
    X, y = z["X_fit"], z["y_fit"]
    fmin, fmax = X.min(0), X.max(0)
    b = y == z["bi"]
    lo, hi = np.empty_like(X), np.empty_like(X)
    lo[b], hi[b] = bowl.box(z, X[b], M)
    lo[~b], hi[~b] = bowl.dbox(z, X[~b], M)
    lo, hi = np.clip(lo, fmin, fmax), np.clip(hi, fmin, fmax)
    S = [lo + rng.random(X.shape, dtype=np.float32) * (hi - lo) for _ in range(2)]
    return np.vstack([X, *S]), np.r_[y, y, y]


def job(args):
    key, name, robust, nrows = args
    z = next(p for p in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1})
             if f"{p['ds']}/{p['name']}" == key)
    bi, k = z["bi"], z["k"]
    rng = np.random.default_rng(0)
    att = z["y_es"] != bi
    sub = lambda X: X[np.sort(rng.choice(len(X), min(nrows, len(X)), replace=False))]  # noqa: E731
    Xu, Xk = sub(z["X_pseudo"]), sub(z["X_es"][att]) if att.any() else None
    Xf, yf = augment(z, np.random.default_rng(SEED)) if robust else (z["X_fit"], z["y_fit"])
    m = (models.rf(SEED) if name == "rf" else models.xgb(SEED)).set_params(n_jobs=1).fit(Xf, yf)
    col = list(m.classes_).index(bi)
    s = lambda X: -np.log(np.maximum(m.predict_proba(X)[:, col], 1e-45))  # noqa: E731
    s_cal = s(z["X_cal"])
    tau = np.sort(s_cal)[::-1][int(np.floor(0.01 * len(s_cal)))]
    rf = name == "rf"
    low = (lambda lo, hi: ct.rf_lower(m, lo, hi, col)) if rf else \
        (lambda lo, hi: ct.xgb_upper(m, lo, hi, col, k, lower=True))  # noqa: E731
    T, off = trees(m, k, col)
    out = {"pair": key, "model": name, "robust_trained": int(robust), "tau": float(tau)}
    for part, X in (("unk", Xu), ("known", Xk)):
        if X is None:
            continue
        lo, hi = bowl.dbox(z, X, M)
        plain, pt = s(X) > tau, low(lo, hi) > tau
        blk, und = pt.copy(), 0
        for i in np.flatnonzero(plain & ~pt):
            s_lo, xr = milp_bound(T, off, lo[i], hi[i], X[i], col, k, rf)
            if s_lo > tau:
                blk[i] = True
            elif xr is not None and s(xr[None])[0] <= tau:
                pass  # attackable: a point in the box drops the score below tau
            else:
                blk[i], und = True, und + 1  # undecided counts as blocked (favours the trees)
        out |= {f"plain_{part}": plain.mean(), f"pertree_{part}": pt.mean(), f"milp_{part}": blk.mean(),
                f"undecided_{part}": und, f"n_{part}": len(X)}
        if part == "unk":  # soundness check on 20 rows
            for i in range(min(20, len(X))):
                s_lo, _ = milp_bound(T, off, lo[i], hi[i], X[i], col, k, rf)
                top = np.maximum(np.minimum(hi[i], z["X_fit"].max(0)), lo[i])
                pts = lo[i] + np.random.default_rng(i).random((200, X.shape[1]), dtype=np.float32) * (top - lo[i])
                lt = low(lo[i:i + 1], hi[i:i + 1])[0]
                if rf or k == 2:  # multiclass XGB: the MILP bound is a different sound bound, not a tighter one
                    assert s_lo >= lt - 1e-4, f"{key} {name}: MILP bound {s_lo} < per-tree bound {lt}"
                assert s_lo <= s(pts).min() + 1e-4 and s_lo <= s(X[i:i + 1])[0] + 1e-4, f"{key} {name}: unsound"
    if Xk is None:  # no known attack rows in this pair: known = unknown, as in gate_bosses
        for q in ("plain", "pertree", "milp"):
            out[f"{q}_known"] = out[f"{q}_unk"]
    for q in ("plain", "pertree", "milp"):
        out[q] = (out[f"{q}_unk"] + out[f"{q}_known"]) / 2
    print(json.dumps({q: (round(v, 3) if isinstance(v, float) else v) for q, v in out.items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"])
    ap.add_argument("--out", default="k5_trees.csv")
    a = ap.parse_args()
    keys = a.pairs or [f"{p['ds']}/{p['name']}" for p in bowl.load_pairs(bowl.ARENA, cheap=False)]
    jobs = [(kk, n, r, a.rows) for kk in keys for n in a.models for r in (0, 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        R = pd.DataFrame(list(ex.map(job, jobs)))
    R.to_csv(bowl.ROOT / "results" / "cert" / a.out, index=False)
    print(R.groupby(["model", "robust_trained"])[["plain", "pertree", "milp", "undecided_unk", "undecided_known"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
