"""Robust-alarm gating for the tree bosses on the full arena (PREREG §9 CERT gated-bosses row): RF / XGB (arena
configs) and the tuned boss (results/bowl/tuned_boss.json), refit per pair with the host-context bytes columns zeroed
like the gated nets, seeds 1 and 2 like the death match. Plain fitness on all rows; robust fitness (alarm counts only
if the sound per-tree lower bound over the Theorem 1' deletion box clears the clean 1%-FPR threshold) on up to
--rows sampled unknown / known-attack rows. Writes results/cert/gate_bosses_{models}.csv.
Usage: .venv/bin/python scripts/gate_bosses.py [--models rf xgb tuned] [--rows 5000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena  # noqa: E402
import boss_bowl  # noqa: E402
import bowl  # noqa: E402
import cert_trees as ct  # noqa: E402
from nids import models  # noqa: E402


def build(name, seed):
    if name == "rf":
        return models.rf(seed)
    if name == "xgb":
        return models.xgb(seed)
    return boss_bowl.model(json.loads((bowl.OUT / "tuned_boss.json").read_text())["tuned_boss"], seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "tuned"])
    ap.add_argument("--rows", type=int, default=5000)
    a = ap.parse_args()
    rows = []
    for z in bowl.drop_bytes(bowl.load_pairs(bowl.ARENA, cheap=False), {"drop_bytes": 1}):
        bi, k = z["bi"], z["k"]
        att = z["y_es"] != bi
        rng = np.random.default_rng(0)
        sub = lambda X: X[np.sort(rng.choice(len(X), min(a.rows, len(X)), replace=False))]  # noqa: E731
        Xu, Xk = sub(z["X_pseudo"]), sub(z["X_es"][att])
        for name in a.models:
            for seed in (1, 2):
                m = build(name, seed)
                m.fit(z["X_fit"], z["y_fit"])
                col = list(m.classes_).index(bi)
                s = {q: -np.log(np.maximum(m.predict_proba(z[f"X_{q}"])[:, col], 1e-45)) for q in ("cal", "pseudo", "es")}
                f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], bi)
                tau = np.sort(s["cal"])[::-1][int(np.floor(0.01 * len(s["cal"])))]
                if hasattr(m, "estimators_"):  # RF
                    low = lambda lo, hi: ct.rf_lower(m, lo, hi, col)  # noqa: E731
                elif hasattr(m, "get_booster"):  # XGB
                    low = lambda lo, hi: ct.xgb_upper(m, lo, hi, col, k, lower=True)  # noqa: E731
                else:  # a LightGBM tuned boss has no tree lower bound here: plain fitness only (disclosed)
                    low = None
                r = {"ds": z["ds"], "pair": z["name"], "model": name, "seed": seed, "fit": f, "unk": u, "known": k_}
                if low is not None:
                    ex = -np.log(np.maximum(m.predict_proba(Xu[:200])[:, col], 1e-45))
                    assert np.allclose(low(Xu[:200], Xu[:200]), ex, atol=1e-3), "zero-width lower bound != exact"
                    for mm in bowl.CERT_MS:
                        ru = (low(*bowl.dbox(z, Xu, mm)) > tau).mean()
                        rk = (low(*bowl.dbox(z, Xk, mm)) > tau).mean() if att.any() else ru
                        r |= {f"rob{mm}": (ru + rk) / 2, f"rob{mm}_unk": ru, f"rob{mm}_known": rk}
                rows.append(r)
                print(json.dumps({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
    R = pd.DataFrame(rows)
    (bowl.ROOT / "results" / "cert").mkdir(parents=True, exist_ok=True)
    R.to_csv(bowl.ROOT / "results" / "cert" / f"gate_bosses_{'_'.join(a.models)}.csv", index=False)
    print(R.groupby("model")[["fit"] + [f"rob{m}" for m in bowl.CERT_MS if f"rob{m}" in R]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
