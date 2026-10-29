"""Bounded-encoding DDU (PREREG §9 PLE): Prep -> bounded piecewise-linear encoding (16 quantile bins per feature,
every component clipped to [0, 1]) -> SN-MLP -> FeatureGDA. Arms: ple_ddu, ple_ce, clip_ddu.
Writes runs/bank/{ds}_host_cap40000_s{seed}_ple/config=outer_{held}/part.parquet (rows identical to the base bank).
Usage: .venv/bin/python scripts/ple.py nf_ctu13 --seed 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import score_bank as sb  # noqa: E402
from nids import models  # noqa: E402

BINS = 16


class Encoder:
    """Bounded piecewise-linear encoding: (n, d) -> (n, d * BINS) in [0, 1]; a zero-width bin is a step at its edge."""

    def fit(self, X):
        self.b = np.quantile(X, np.linspace(0, 1, BINS + 1), axis=0).T.astype(np.float32)  # (d, BINS + 1)
        return self

    def transform(self, X):
        lo, hi = self.b[None, :, :-1], self.b[None, :, 1:]
        w = hi - lo
        x = X[:, :, None]
        E = np.where(w > 0, np.clip((x - lo) / np.where(w > 0, w, 1), 0, 1), (x >= hi).astype(np.float32))
        return E.reshape(len(X), -1).astype(np.float32)


class Clip:
    def fit(self, X):
        self.lo, self.hi = np.quantile(X, [0.001, 0.999], axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo, self.hi).astype(np.float32)


def fit_models(Xf, yf, Xe, ye, k, bi, seed, **_):
    """{arm: score(X prepped)} for the three arms."""
    out = {}
    for name, enc in (("ple", Encoder()), ("clip", Clip())):
        enc.fit(Xf)
        Ef, Ee = enc.transform(Xf), enc.transform(Xe)
        snm = models.train_mlp(Ef, yf, Ee, ye, k, seed=seed, arch=models.SNMLP).eval()
        gda = models.FeatureGDA().fit(snm, Ef, yf, k)
        out[f"{name}_ddu"] = lambda X, e=enc, m=snm, g=gda: g.score(m, e.transform(X))
        if name == "ple":
            mlp = models.train_mlp(Ef, yf, Ee, ye, k, seed=seed).eval()
            out["ple_ce"] = lambda X, e=enc, m=mlp: 1 - models.mlp_proba(m, e.transform(X))[:, bi]
    return out


def run_config(bank, held, base_part, fit_models=fit_models):
    P = (held,)
    fit, es = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    dfit = bank.frame(fit, sb.role_removal(P, "fit"))
    prep = models.Prep().fit(dfit)
    Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit]])
    es = es[np.isin(bank.fam[es], known)]
    Xe, ye = prep.transform(bank.frame(es, sb.role_removal(P, "es"))), np.array([code[f] for f in bank.fam[es]])
    arms = fit_models(Xf, yf, Xe, ye, len(known), code["Benign"], bank.seed, bank=bank, P=P)
    roles = {"rank": bank.rank, "cal": bank.cal, "test_benign": bank.test_benign,
             "test_unknown": np.sort(np.concatenate([r[bank.removed(r, P)] for r in bank.test.values()])),
             "test_known": np.sort(np.concatenate([r[~bank.removed(r, P)] for r in bank.test.values()]))}
    parts = []
    for role, rows in roles.items():
        if len(rows) == 0:
            continue
        raw = bank.frame(rows, sb.role_removal(P, role))
        X = prep.transform(raw)
        d = {"row_id": rows.astype(np.int64), "role": role, "family": bank.fam[rows]}
        for a, f in arms.items():
            src = raw if getattr(f, "raw", False) else X  # raw arms (KitNET) take the unprepped frame
            d[a] = np.concatenate([f(src[i:i + 100_000]) for i in range(0, len(src), 100_000)]).astype(np.float32)
        parts.append(pd.DataFrame(d))
    b = base_part.groupby("role").row_id.apply(np.sort)
    for role, rows in roles.items():
        assert np.array_equal(np.sort(rows), b.get(role, np.empty(0, np.int64))), f"{held} {role}: rows differ"
    return pd.concat(parts, ignore_index=True)


def main(tag="ple", fit_models=fit_models):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    base = sb.OUT / f"{a.dataset}_host_cap40000_s{a.seed}"
    out = sb.OUT / f"{a.dataset}_host_cap40000_s{a.seed}_{tag}"
    bank = sb.Bank(a.dataset, a.seed, "host", 40000, 200000)
    for p in sorted(base.glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        df = run_config(bank, held, sb.read_part(p, ["row_id", "role"]), fit_models)
        dest = out / f"config=outer_{held}"
        dest.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest / "part.parquet", index=False)
        print(f"{held}: {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    (out / "manifest_outer.json").write_text(json.dumps({"dataset": a.dataset, "seed": a.seed,
                                                         "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
