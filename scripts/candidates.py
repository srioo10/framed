"""PREREG §9 SEL candidates in one run: tabm, tabm_min, tabm_mi, tabm_raw (tabm.py), ple_ce (ple.py), tabm_kd(_min), lxgb_hc (light
XGB on per-flow + hc_*), plus kitnet refit as score_bank.run_config where the base bank lacks it.
Writes runs/bank/{ds}_host_cap40000_s{seed}_cand/... via ple.main.
Usage: .venv/bin/python scripts/candidates.py nf_ton --seed 1
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flowtok  # noqa: E402
import ple  # noqa: E402
import score_bank as sb  # noqa: E402
import tabm  # noqa: E402
from nids import kitnet, mcde, models  # noqa: E402


def kd_models(Xf, yf, Xe, ye, k, bi, seed):
    """PREREG §9 SEL amendment 3: TabM-mini on tree-chosen bin edges, distilled from 3-fold OOF XGB (300 x depth 8)."""
    enc = ple.TreeEncoder().fit(Xf, yf, seed)
    m = tabm.train(enc.transform(Xf), yf, enc.transform(Xe), ye, k, seed=seed, soft=models.oof_proba(Xf, yf, seed))
    return {f"tabm_kd{s}": (lambda X, s=s: tabm.scores(tabm.member_proba(m, enc.transform(X)), bi)[s]) for s in ("", "_min")}


def fit_models(Xf, yf, Xe, ye, k, bi, seed, bank, P):
    out = tabm.fit_models(Xf, yf, Xe, ye, k, bi, seed)
    enc = ple.Encoder().fit(Xf)
    mlp = models.train_mlp(enc.transform(Xf), yf, enc.transform(Xe), ye, k, seed=seed).eval()
    out["ple_ce"] = lambda X: tabm.nlog(models.mlp_proba(mlp, enc.transform(X))[:, bi])
    out |= kd_models(Xf, yf, Xe, ye, k, bi, seed)
    xg = flowtok.light_xgb(seed).fit(Xf, yf)
    out["lxgb_hc"] = lambda X: tabm.nlog(xg.predict_proba(X)[:, bi])
    base = next((sb.OUT / f"{bank.ds}_host_cap40000_s{seed}").glob("config=outer_*/part.parquet"))
    if "kitnet" not in pq.read_schema(base).names:
        dben = bank.frame(bank.bsample, sb.role_removal(P, "bsample"))
        bprep = mcde.RobustPrep().fit(dben)
        kit = kitnet.KitNET(seed).fit(bprep.transform(dben))
        out["kitnet"] = lambda d: kit.score(bprep.transform(d))
        out["kitnet"].raw = True
    return out


if __name__ == "__main__":
    ple.main("cand", fit_models)
