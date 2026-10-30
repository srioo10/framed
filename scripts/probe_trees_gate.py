"""Exploratory probe (dev pairs 0-1 per dataset, cheap arena, host-context bytes zeroed like the nets; disclosed):
robust-alarm gating applied to the RF / XGB bosses with the same deletion box and their sound tree lower bounds."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import arena  # noqa: E402
import bowl  # noqa: E402
import cert_trees as ct  # noqa: E402
from nids import models  # noqa: E402

rng = np.random.default_rng(0)
for z in [z for z in bowl.load_pairs(bowl.ARENA, cheap=True) if z["name"] in ("pair_0", "pair_1")]:
    j = [i for i, c in enumerate(z["cols"]) if c.startswith("hc_") and "_bytes_" in c]
    for k in [k for k in z if k.startswith("X_")]:
        z[k] = z[k].copy()
        z[k][:, j] = 0
    bi, k = z["bi"], z["k"]
    a = z["y_es"] != bi
    sub = lambda X: X[np.sort(rng.choice(len(X), min(5000, len(X)), replace=False))]  # noqa: E731
    Xu, Xk = sub(z["X_pseudo"]), sub(z["X_es"][a])
    for name, m in (("rf", models.rf(0)), ("xgb", models.xgb(0))):
        m.fit(z["X_fit"], z["y_fit"])
        col = list(m.classes_).index(bi)
        s = {q: -np.log(np.maximum(m.predict_proba(z[f"X_{q}"])[:, col], 1e-45)) for q in ("cal", "pseudo", "es")}
        f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], bi)
        tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
        low = (lambda lo, hi: ct.rf_lower(m, lo, hi, col)) if name == "rf" else \
            (lambda lo, hi: ct.xgb_upper(m, lo, hi, col, k, lower=True))
        assert np.allclose(low(Xu[:200], Xu[:200]), -np.log(np.maximum(m.predict_proba(Xu[:200])[:, col], 1e-45)),
                           atol=1e-3), "zero-width lower bound != exact score"
        r = {"pair": f"{z['ds']}/{z['name']}", "model": name, "plain": [round(f, 3), round(u, 3), round(k_, 3)]}
        for mm in (1, 5, 10):
            ru = (low(*bowl.dbox(z, Xu, mm)) > tau).mean()
            rk = (low(*bowl.dbox(z, Xk, mm)) > tau).mean()
            r[f"rob{mm}"] = [round(float((ru + rk) / 2), 3), round(float(ru), 3), round(float(rk), 3)]
        print(json.dumps(r), flush=True)
