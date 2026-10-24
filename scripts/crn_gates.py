"""CRN gates D1 and D2 (PREREG §9, 2026-10-24 CRN row). One process per (dataset, seed); --report merges.

D1: Gaussian conditional residual on benign-train only (bsample; never rank / cal). z = Gaussian-quantile transform,
    Λ = Ledoit-Wolf precision, r_j = (Λ(z - mean))_j / sqrt(Λ_jj), s_G = max_j |r_j|; also Mahalanobis.
    Compared with every existing bank member, alone and as rank-min with MLP-CE (min of tail p-values vs rank).
D2: seen-family ceiling: XGB fitted with removal set {} (all families known), TPR@1% on each family's test rows.
Usage: .venv/bin/python scripts/crn_gates.py ciciot_csv none --seed 0 --out results/crn/gates
       .venv/bin/python scripts/crn_gates.py ciciot_csv none --report --out results/crn/gates
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc, models  # noqa: E402
import bank_report as br  # noqa: E402
import score_bank as sb  # noqa: E402

MEMBERS = ["rf", "xgb", "mlp_ce", "iforest", "clust", "mcde"]
NEW = ["s_G", "mahal", "s_G+mlp_ce", "mahal+mlp_ce"]
HARD_CIC = ["BruteForce", "Recon", "Spoofing", "Web"]


def tpr(cal, unk):
    return float((unk > fc.threshold(cal, 0.01)).mean())


def rank_min(*cols):
    """Two-score rank-min: -min of tail p-values vs rank (larger = more anomalous)."""
    R = np.column_stack([c[0] for c in cols])
    return lambda i: -fc.tail_pvalues(R, np.column_stack([c[1][i] for c in cols])).min(1)


def one_seed(ds, variant, seed):
    bank = sb.Bank(ds, seed, variant, 40000, 200000)
    run = sb.OUT / f"{ds}_{variant}_cap40000_s{seed}"
    alpha_min = 0.01 if ds == "unsw_official" else 0.001
    out = []
    # D2: one closed-set XGB per seed
    fit, _ = bank.fit_rows(())
    dfit = bank.frame(fit, sb.role_removal((), "fit"))
    prep = models.Prep().fit(dfit)
    known = sorted(set(bank.fam[fit]))
    bi = known.index("Benign")
    closed = models.xgb(seed).fit(prep.transform(dfit), np.array([known.index(f) for f in bank.fam[fit]]))
    p_attack = lambda idx: 1 - closed.predict_proba(prep.transform(bank.frame(idx, ())))[:, bi]  # noqa: E731
    c_cal = p_attack(bank.cal)
    for k in bank.families:
        P = (k,)
        part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet")
        rows = {r: part[part.role == r] for r in ("rank", "cal", "test_benign", "test_unknown")}
        res = {"seed": seed, "fold": k, "n": len(rows["test_unknown"])}
        for m in MEMBERS:
            s = {r: (br.mcde_score(rows["rank"], d, alpha_min) if m == "mcde" else d[m].to_numpy(np.float64))
                 for r, d in rows.items()}
            res[m] = tpr(s["cal"], s["test_unknown"])
        # D1 on benign-train rows only
        dben = bank.frame(bank.bsample, sb.role_removal(P, "bsample"))
        qprep = models.Prep().fit(dben)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                                 random_state=seed).fit(qprep.transform(dben))
        Zb = qt.transform(qprep.transform(dben))
        keep = Zb.std(0) > 1e-6
        mu = Zb[:, keep].mean(0)
        lam = LedoitWolf().fit(Zb[:, keep] - mu).precision_
        dg = np.sqrt(np.diag(lam))

        def feats(r):
            idx = rows[r].row_id.to_numpy()
            Z = qt.transform(qprep.transform(bank.frame(idx, sb.role_removal(P, r)))) [:, keep] - mu
            resid = (Z @ lam) / dg
            return np.abs(resid).max(1), np.einsum("ij,ij->i", Z @ lam, Z)
        sc = {r: feats(r) for r in rows}
        for j, name in enumerate(("s_G", "mahal")):
            res[name] = tpr(sc["cal"][j], sc["test_unknown"][j])
            ref = (sc["rank"][j], {r: sc[r][j] for r in rows})
            mlp = (rows["rank"].mlp_ce.to_numpy(np.float64), {r: rows[r].mlp_ce.to_numpy(np.float64) for r in rows})
            f = rank_min(ref, mlp)
            res[f"{name}+mlp_ce"] = tpr(f("cal"), f("test_unknown"))
        # D2: closed-set ceiling on this family's test rows
        res["seen_ceiling"] = tpr(c_cal, p_attack(rows["test_unknown"].row_id.to_numpy()))
        out.append(res)
        print(f"s{seed} {k:16s} best member {max(res[m] for m in MEMBERS):.3f} | s_G {res['s_G']:.3f} "
              f"s_G+mlp {res['s_G+mlp_ce']:.3f} mahal {res['mahal']:.3f} | seen ceiling {res['seen_ceiling']:.3f}",
              flush=True)
    return out


def report(ds, out):
    res = pd.DataFrame([r for f in sorted(out.parent.glob(f"{out.name}_{ds}_s*.json")) for r in json.loads(f.read_text())])
    fam = res.groupby("fold")[MEMBERS + NEW + ["seen_ceiling", "n"]].mean()
    fam["best_member"] = fam[MEMBERS].max(1)
    fam["best_name"] = fam[MEMBERS].idxmax(1)
    hard = HARD_CIC if ds.startswith("ciciot") else [k for k in fam.index if fam.loc[k, "best_member"] < 0.5]
    hard = [k for k in hard if k in fam.index and fam.loc[k, "n"] >= br.MIN_ROWS]
    f = lambda c: br.fmt(*br.ci(c))  # noqa: E731
    L = [f"# CRN gates D1 / D2 — {ds} (seeds {sorted(res.seed.unique())})\n", "TPR@1%FPR, 5-seed mean [t-CI]\n",
         "| family | n | best member | " + " | ".join(NEW) + " | seen ceiling |", "|---|---|---|" + "---|" * (len(NEW) + 1)]
    for k, g in res.groupby("fold"):
        L.append(f"| {k}{' (hard)' if k in hard else ''} | {int(g.n.min())} | {fam.loc[k, 'best_member']:.3f} "
                 f"({fam.loc[k, 'best_name']}) | " + " | ".join(f(g[c]) for c in NEW) + f" | {f(g.seen_ceiling)} |")
    gain = {k: max(fam.loc[k, "s_G"], fam.loc[k, "s_G+mlp_ce"]) - fam.loc[k, "best_member"] for k in hard}
    n_gain = sum(v >= 0.03 for v in gain.values())
    elig = [k for k in fam.index if fam.loc[k, "n"] >= br.MIN_ROWS]
    macro = res[res.fold.isin(elig)].groupby("seed")["s_G+mlp_ce"].mean()
    L.append(f"\nhard families: {hard}; gain of best s_G variant over best member: " +
             ", ".join(f"{k} {v:+.3f}" for k, v in gain.items()))
    L.append(f"families with gain >= 0.03: {n_gain}; macro s_G+mlp_ce over eligible families: {f(macro)}")
    text = "\n".join(L)
    print(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_{ds}.md").write_text(text + "\n")
    return {"n_gain": n_gain, "macro": float(macro.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("variant")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(a.ds, a.out)
        return
    Path(f"{a.out}_{a.ds}_s{a.seed}.json").write_text(json.dumps(one_seed(a.ds, a.variant, a.seed), indent=1,
                                                                 default=float))


if __name__ == "__main__":
    main()
