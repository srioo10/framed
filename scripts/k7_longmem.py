"""K7 kill test (PREREG §9 K7): does a long-memory host profile (hostctx statistics over 1 h and 24 h, `hl_*`) help
the boss RF / XGB on the existing arena pairs? Same rows and roles as data/arena; Prep and both bosses are refit with
and without hl_*. Harness check: the without-arm reproduces the saved arena X. Writes results/k7/k7_{ds}.csv.
Leakage controls (K7 amendment 1): --control perm (IPs shuffled within 10-min blocks) | global (all hosts one key).
Usage: .venv/bin/python scripts/k7_longmem.py nf_ctu13 [--pairs 0 1 2] [--control perm|global]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import arena  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import hostctx, load, models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--pairs", type=int, nargs="*")
    ap.add_argument("--control", choices=["perm", "global"])
    a = ap.parse_args()
    t0 = time.time()
    bank = sb.Bank(a.dataset, arena.SEED, "host", 40000, 200000)
    host, cache = bank.host, {}
    if a.control == "perm":
        host = hostctx.HostFlows(load.LOADERS[a.dataset](), permute_seed=1)
        host.keep = bank.kept
    elif a.control == "global":
        raw = load.LOADERS[a.dataset]()
        src, dst = raw["IPV4_SRC_ADDR"].astype(str).to_numpy(), raw["IPV4_DST_ADDR"].astype(str).to_numpy()
        host = copy.copy(bank.host)
        z0 = np.zeros(len(src), np.int64)
        host.key = {"src": z0, "dst": z0}
        host.pair_port = {k: hostctx._codes(raw["L4_DST_PORT"].to_numpy()) for k in ("src", "dst")}
        host.pair_peer = {"src": hostctx._codes(dst), "dst": hostctx._codes(src)}
        del raw

    def long(rows, R):  # same population rule as HostFlows.removal, long windows
        R = tuple(sorted(R))
        if R not in cache:
            cache[R] = host.features(~np.isin(host.family, list(R)), hostctx.LONG_MS)[host.keep]
        return pd.DataFrame(cache[R][rows], columns=hostctx.LONG_COLS)

    out, res = ROOT / "results" / "k7", []
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((arena.OUT / a.dataset).glob("pair_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
    for f in files:
        i = int(f.stem.split("_")[1])
        if a.pairs is not None and i not in a.pairs:
            continue
        z, meta = np.load(f), json.loads(f.with_suffix(".json").read_text())
        R0, P = tuple(meta["R0"]), tuple(sorted(set(meta["R0"]) | {meta["j"]}))
        roles = {"fit": sb.role_removal(P, "fit"), "es": sb.role_removal(P, "es"),
                 "pseudo": sb.role_removal(P, "pseudo", R0), "cal": sb.role_removal(P, "cal")}
        D = {k: bank.frame(z[f"rows_{k}"], R) for k, R in roles.items()}
        L = {k: long(z[f"rows_{k}"], R) for k, R in roles.items()}
        y, bi = z["y_fit"], int(z["bi"])
        for arm in ("without", "with"):
            F = D if arm == "without" else {k: pd.concat([D[k], L[k]], axis=1) for k in D}
            prep = models.Prep().fit(F["fit"])
            X = {k: prep.transform(v) for k, v in F.items()}
            if arm == "without":
                for k in X:
                    assert np.allclose(X[k], z[f"X_{k}"], atol=1e-4), f"{f}: {k} differs from the saved arena"
            for name in ("rf", "xgb"):
                m = getattr(models, name)(arena.SEED).fit(X["fit"], y)
                s = {k: -np.log(np.maximum(m.predict_proba(X[k])[:, bi], 1e-45)) for k in ("cal", "pseudo", "es")}
                fit, unk, known = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], bi)
                r = {"ds": a.dataset, "pair": i, "arm": arm, "model": name, "fit": fit, "unk": unk, "known": known}
                if arm == "without":
                    r["saved_diff"] = float(np.abs(s["cal"] - z[f"boss_{name}_cal"]).max())
                res.append(r)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}),
                      f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(res).to_csv(out / f"k7_{a.dataset}{'_' + a.control if a.control else ''}.csv", index=False)  # saved after every pair
    R = pd.DataFrame(res).pivot_table(index="model", columns="arm", values=["fit", "unk", "known"])
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
