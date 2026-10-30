"""Exploratory probe (bytes host-context columns dropped): 'robust alarm' gating. Block only if the alarm survives removing ANY <= m flows from each context
window (removal box around the observed row). A benign flow's clean row lies in the removal box of its attacked row,
so blocked-benign FPR under <= m injections is <= the clean alarm FPR (1%) by construction. Cost = detection lost.
Measures robust-alarm unknown / known TPR at tau (clean 1% FPR) for cert 0 and cert 4 nets, m in {1, 5, 10}."""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/srioo/Desktop/Projects/root/BigP/scripts")
import arena  # noqa: E402
import bowl  # noqa: E402
import fighters as fx  # noqa: E402

torch.set_num_threads(4)
G = {"family": "mlp", "enc": "raw", "e": 2, "w": 64, "depth": 2, "sn": 0, "act": "relu", "members": 1, "head": "nlog",
     "kd": 0.5, "syn": 0.5, "ls": 0.0, "focal": 2.0, "lr": 0.003, "clip": 1}


def removal_box(X, cols, mu, sd, m):
    """Per-group box of every row reachable by deleting j <= m flows of the window (product over groups; sound)."""
    X = np.asarray(X, np.float64)
    lo, hi = X.copy(), X.copy()
    raw = lambda j: np.expm1(np.maximum(X[:, j] * sd[j] + mu[j], 0))  # noqa: E731
    back = lambda j, v: (np.log1p(v) - mu[j]) / sd[j]  # noqa: E731
    for key in ("src", "dst"):
        for w in ("10s", "120s"):
            idx = {s: cols.index(f"hc_{key}_{s}_{w}") for s in ("n", "fail", "newport", "newpeer", "bytes")}
            n = np.rint(raw(idx["n"]))
            r = np.maximum(n - m, 0)  # fewest flows left
            lo[:, idx["n"]] = back(idx["n"], r)
            empty = r < 1
            d = np.maximum(r, 1)
            for s in ("fail", "newport", "newpeer"):
                c = np.rint(raw(idx[s]) * np.maximum(n, 1))
                c_hi = c if s == "fail" else c + m
                lo_s, hi_s = np.maximum(c - m, 0) / d, np.minimum(c_hi, d) / d
                lo_s, hi_s = np.where(empty, 0, np.minimum(lo_s, raw(idx[s]))), np.where(empty, 1, np.maximum(hi_s, raw(idx[s])))
                lo[:, idx[s]], hi[:, idx[s]] = back(idx[s], lo_s), back(idx[s], np.minimum(hi_s, 1))
            pass  # bytes context dropped (constant), nothing to bound
    hc = np.array([c.startswith("hc_") for c in cols])
    lo[:, hc] = np.minimum(lo[:, hc], X[:, hc]) - 1e-4
    hi[:, hc] = np.maximum(hi[:, hc], X[:, hc]) + 1e-4
    return lo.astype(np.float32), hi.astype(np.float32)


def score_lower(m, lo, hi, bi):
    """Lower bound of -log P(benign) = logsumexp_i (z_i - z_bi): margins' lower bounds via a negated head."""
    neg = copy.deepcopy(m)
    with torch.no_grad():
        neg.head.weight.neg_(), neg.head.bias.neg_()
        d = -fx.margins_upper(neg, torch.as_tensor(lo), torch.as_tensor(hi), bi)[:, 0]
        d[:, bi] = 0
        return torch.logsumexp(d.double(), -1).numpy()


pairs = [z for z in bowl.load_pairs(bowl.ARENA, cheap=True) if z["name"] in ("pair_0", "pair_1")]
for z in pairs:
    j = [i for i, c in enumerate(z["cols"]) if c.startswith("hc_") and "_bytes_" in c]  # bytes context dropped
    for k in [k for k in z if k.startswith("X_")]:
        z[k] = z[k].copy()
        z[k][:, j] = 0
    for cert in (4, 8):
        g = G | {"cert": cert}
        m, h, s = bowl.run_pair(g, z, "cpu", 0, 15)
        m.eval()
        f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
        tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
        # sanity: lower bound <= plain score
        lo, hi = removal_box(z["X_cal"][:2000], z["cols"], z["mu"], z["sd"], 1)
        assert (score_lower(m, lo, hi, z["bi"]) <= s["cal"][:2000] + 1e-3).all()
        r = {"pair": f"{z['ds']}/{z['name']}", "cert": cert, "fit": round(f, 3), "unk": round(u, 3), "known": round(k_, 3)}
        att = z["y_es"] != z["bi"]
        for mm in (1, 5, 10):
            ru = (score_lower(m, *removal_box(z["X_pseudo"], z["cols"], z["mu"], z["sd"], mm), z["bi"]) > tau).mean()
            Xa = z["X_es"][att]
            rk = (score_lower(m, *removal_box(Xa, z["cols"], z["mu"], z["sd"], mm), z["bi"]) > tau).mean()
            r[f"robust{mm}"] = [round(float((ru + rk) / 2), 3), round(float(ru), 3), round(float(rk), 3)]
        print(json.dumps(r), flush=True)
