"""Exploratory probe (not a bowl run): where does the certified FPR(10) of a cert-trained net come from, and which
round-5 fix could move it. Pairs 0-1 of each dataset, cheap arena, seed 0, the round-4 seed genome (r3 body, nlog, cert 4).
Variants: full features | src 120 s context dropped | all src context dropped (constant column, not boxed)."""
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
     "kd": 0.5, "syn": 0.5, "ls": 0.0, "focal": 2.0, "lr": 0.003, "cert": 4, "clip": 1}
M = 10
variant = sys.argv[1]
pairs = [z for z in bowl.load_pairs(bowl.ARENA, cheap=True) if z["name"] in ("pair_0", "pair_1")]


def drop(z, pref):
    """Column -> constant 0 (the standardized mean) and renamed so the box leaves it alone."""
    j = [i for i, c in enumerate(z["cols"]) if any(c.startswith(p) for p in pref)]
    for k in [k for k in z if k.startswith("X_")]:
        z[k] = z[k].copy()
        z[k][:, j] = 0
    z["cols"] = [("x" + c if i in j else c) for i, c in enumerate(z["cols"])]


def cert_rate(m, z, lo, hi, tau):
    return float((fx.score_upper(m, lo, hi, z["bi"], "nlog") > tau).mean())


def pgd(m, z, X, lo, hi, tau, steps=40):
    """Empirical: any point of the box found above tau (lower bound of what the box allows)."""
    clo, chi = m.clo.numpy(), m.chi.numpy()
    lo, hi = np.maximum(lo, np.minimum(clo, X)), np.minimum(hi, np.maximum(chi, X))
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    hit = np.zeros(len(X), bool)
    for start in (X, hi, (lo + hi) / 2):
        x = torch.tensor(np.clip(start, lo, hi), dtype=torch.float32, requires_grad=True)
        for _ in range(steps):
            s = -torch.log_softmax(m(x)[0], -1)[:, z["bi"]]
            g, = torch.autograd.grad(s.sum(), x)
            with torch.no_grad():
                x += 0.1 * (hi_t - lo_t) * g.sign()
                x.copy_(torch.maximum(torch.minimum(x, hi_t), lo_t))
        with torch.no_grad():
            hit |= (-torch.log_softmax(m(x)[0], -1)[:, z["bi"]]).numpy() > tau
    return float(hit.mean())


out = []
for z in pairs:
    if variant == "nosrc120":
        drop(z, ("hc_src_n_120", "hc_src_fail_120", "hc_src_newport_120", "hc_src_newpeer_120", "hc_src_bytes_120"))
    elif variant == "nosrc":
        drop(z, ("hc_src_",))
    m, h, s = bowl.run_pair(G, z, "cpu", 0, 15)
    m.eval()
    f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
    tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
    i = np.sort(np.random.default_rng(0).choice(len(s["cal"]), min(5000, len(s["cal"])), replace=False))
    X = z["X_cal"][i]
    lo, hi = (v.astype(np.float32) for v in bowl.box(z, X, M))
    cols = z["cols"]
    r = {"pair": f"{z['ds']}/{z['name']}", "variant": variant, "fit": f, "unk": u, "known": k_,
         "cert": cert_rate(m, z, lo, hi, tau)}
    groups = {"dst": lambda c: c.startswith("hc_dst"), "src": lambda c: c.startswith("hc_src"),
              "src10": lambda c: c.startswith("hc_src") and c.endswith("_10s"),
              "src120": lambda c: c.startswith("hc_src") and c.endswith("_120s")}
    groups |= {f"src120_{st}": (lambda c, st=st: c == f"hc_src_{st}_120s")
               for st in ("n", "fail", "newport", "newpeer", "bytes")}
    for name, sel in groups.items():
        o = np.array([sel(c) for c in cols])
        if not o.any():
            continue
        l2, h2 = X.copy(), X.copy()
        l2[:, o], h2[:, o] = lo[:, o], hi[:, o]
        r[f"only_{name}"] = cert_rate(m, z, l2, h2, tau)
    if r["cert"] > 0:
        r["pgd"] = pgd(m, z, X, lo, hi, tau)
    if "hc_src_n_120s" in cols:  # failure rate by the flow's real src 120 s count
        j = cols.index("hc_src_n_120s")
        n = np.rint(np.expm1(np.maximum(X[:, j] * z["sd"][j] + z["mu"][j], 0)))
        bad = fx.score_upper(m, lo, hi, z["bi"], "nlog") > tau
        r["by_n"] = {f"{a}-{b}": [int(((n >= a) & (n <= b)).sum()), float(bad[(n >= a) & (n <= b)].mean())
                                  if ((n >= a) & (n <= b)).any() else None]
                     for a, b in ((0, 0), (1, 2), (3, 9), (10, 99), (100, 10**12))}
    print(json.dumps(r), flush=True)
    out.append(r)
