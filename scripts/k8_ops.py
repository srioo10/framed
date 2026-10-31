"""K8 (PREREG §9 K8): implementation-independent cost of a certified-block decision per flow, same rows as item 2
(check_cost.py: pair_1, 1,000 unknown + known rows, seed 0, deletion box at m = 10). Net: 2 x weights of champion B
(interval bound propagation = mid-point and radius passes). Trees: node visits of cert_trees.reach per row.
Writes results/cert/k8_ops.csv.
Usage: .venv/bin/python scripts/k8_ops.py
"""
import numpy as np
import pandas as pd

import bowl
import cert_trees as ct
from check_cost import B
from nids import models


def main():
    res = []
    for z in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1}):
        if z["name"] != "pair_1":
            continue
        bi, k, rng = z["bi"], z["k"], np.random.default_rng(0)
        att = z["y_es"] != bi
        X = np.vstack([z["X_pseudo"], z["X_es"][att]])
        X = X[np.sort(rng.choice(len(X), min(1000, len(X)), replace=False))]
        lo, hi = bowl.dbox(z, X, 10)
        net = 2 * sum(p.numel() for p in bowl.build(B, z, 1).parameters())
        rf = models.rf(1).fit(z["X_fit"], z["y_fit"])
        xg = models.xgb(1).fit(z["X_fit"], z["y_fit"])
        for name, f in (("rf", lambda: ct.rf_lower(rf, lo, hi, list(rf.classes_).index(bi))),
                        ("xgb", lambda: ct.xgb_upper(xg, lo, hi, list(xg.classes_).index(bi), k, lower=True))):
            ct.VISITS = np.zeros(len(X))
            f()
            v, ct.VISITS = ct.VISITS, None
            res.append({"ds": z["ds"], "tree": name, "net_madds": net, "visits_median": np.median(v),
                        "visits_p99": np.quantile(v, 0.99), "ratio_median": np.median(v) / net})
            print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(bowl.ROOT / "results" / "cert" / "k8_ops.csv", index=False)


if __name__ == "__main__":
    main()
