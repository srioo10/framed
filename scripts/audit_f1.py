"""PREREG §9 AUDIT F1: block decisions of champion B with float32 vs float64 (+ outward margin) IBP bounds, pair_1
of each dataset, all unknown + known-attack rows, deletion box at m = 10. Writes results/cert/audit_f1.csv.
Usage: .venv/bin/python scripts/audit_f1.py
"""
import numpy as np
import pandas as pd

import bowl
import fighters as fx
from check_cost import B


def main():
    res = []
    for z in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1}):
        if z["name"] != "pair_1":
            continue
        m, h, s = bowl.run_pair(B, z, "cpu", 1, 30)
        tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
        X = np.vstack([z["X_pseudo"], z["X_es"][z["y_es"] != z["bi"]]])
        lo, hi = bowl.dbox(z, X, 10)
        b32 = fx.score_lower(m, lo, hi, z["bi"], B["head"], f64=False)
        b64 = fx.score_lower(m, lo, hi, z["bi"], B["head"])
        exact = h.score(m, X, "cpu")
        res.append({"ds": z["ds"], "rows": len(X), "flips": int(((b32 > tau) != (b64 > tau)).sum()),
                    "max_abs_diff": float(np.abs(b32 - b64).max()), "b64_above_exact": int((b64 > exact + 1e-9).sum()),
                    "block32": float((b32 > tau).mean()), "block64": float((b64 > tau).mean())})
        print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(bowl.ROOT / "results" / "cert" / "audit_f1.csv", index=False)


if __name__ == "__main__":
    main()
