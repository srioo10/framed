"""Runner-up gate (PREREG §9 CRN row): reference-class softmax, one MLP, one pass.

Per outer fold k: the bank's MLP (same fit / early-stop rows, same Prep) gets one extra class, "reference":
column-shuffled benign fit rows (each feature permuted independently, so the marginals stay benign and the joint
structure is broken) plus the same number of rows drawn uniformly in each feature's fit range. Scores:
refratio = log p_ref - log p_benign (a density ratio reference / benign), notbenign = 1 - p_benign, and each as
rank-min with the bank's MLP-CE. Same comparison and PASS rule as crn_gates.py.
Usage: .venv/bin/python scripts/crn_gate_refclass.py ciciot_csv none --seed 0 --out results/crn/refclass
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import models  # noqa: E402
import bank_report as br  # noqa: E402
import crn_gates as cg  # noqa: E402
import score_bank as sb  # noqa: E402

NEW = ["refratio", "notbenign", "refratio+mlp_ce", "notbenign+mlp_ce"]


def reference(Xb, n, rng):
    """n column-shuffled benign rows + n uniform rows in the benign range."""
    shuf = np.column_stack([rng.choice(Xb[:, j], n) for j in range(Xb.shape[1])])
    lo, hi = Xb.min(0), Xb.max(0)
    return np.vstack([shuf, rng.uniform(lo, hi, (n, Xb.shape[1]))]).astype(np.float32)


def one_seed(ds, variant, seed):
    bank = sb.Bank(ds, seed, variant, 40000, 200000)
    run = sb.OUT / f"{ds}_{variant}_cap40000_s{seed}"
    alpha_min = 0.01 if ds == "unsw_official" else 0.001
    out = []
    for k in bank.families:
        P = (k,)
        rng = np.random.default_rng([seed, 7])
        fit, es = bank.fit_rows(P)
        known = sorted(set(bank.fam[fit]))
        code = {f: i for i, f in enumerate(known)}
        bi, ri = code["Benign"], len(known)
        dfit = bank.frame(fit, sb.role_removal(P, "fit"))
        prep = models.Prep().fit(dfit)
        Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit]])
        Xe = prep.transform(bank.frame(es, sb.role_removal(P, "es")))
        ye = np.array([code[f] for f in bank.fam[es]])
        nb, nbe = (yf == bi).sum(), (ye == bi).sum()
        Rf, Re = reference(Xf[yf == bi], nb // 2, rng), reference(Xf[yf == bi], max(nbe // 2, 1), rng)
        m = models.train_mlp(np.vstack([Xf, Rf]), np.r_[yf, np.full(len(Rf), ri)],
                             np.vstack([Xe, Re]), np.r_[ye, np.full(len(Re), ri)], ri + 1, seed=seed)
        part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet")
        rows = {r: part[part.role == r] for r in ("rank", "cal", "test_benign", "test_unknown")}
        res = {"seed": seed, "fold": k, "n": len(rows["test_unknown"])}
        for mem in cg.MEMBERS:
            s = {r: (br.mcde_score(rows["rank"], d, alpha_min) if mem == "mcde" else d[mem].to_numpy(np.float64))
                 for r, d in rows.items()}
            res[mem] = cg.tpr(s["cal"], s["test_unknown"])
        sc = {}
        for r, d in rows.items():
            p = models.mlp_proba(m, prep.transform(bank.frame(d.row_id.to_numpy(), sb.role_removal(P, r))))
            sc[r] = {"refratio": np.log(p[:, ri] + 1e-12) - np.log(p[:, bi] + 1e-12), "notbenign": 1 - p[:, bi]}
        mlp = (rows["rank"].mlp_ce.to_numpy(np.float64), {r: d.mlp_ce.to_numpy(np.float64) for r, d in rows.items()})
        for name in ("refratio", "notbenign"):
            res[name] = cg.tpr(sc["cal"][name], sc["test_unknown"][name])
            f = cg.rank_min((sc["rank"][name], {r: sc[r][name] for r in rows}), mlp)
            res[f"{name}+mlp_ce"] = cg.tpr(f("cal"), f("test_unknown"))
        out.append(res)
        print(f"s{seed} {k:16s} best member {max(res[x] for x in cg.MEMBERS):.3f} | " +
              " ".join(f"{x} {res[x]:.3f}" for x in NEW), flush=True)
    return out


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
        res = pd.DataFrame([r for f in sorted(a.out.parent.glob(f"{a.out.name}_{a.ds}_s*.json"))
                            for r in json.loads(f.read_text())])
        fam = res.groupby("fold")[cg.MEMBERS + NEW + ["n"]].mean()
        fam["best"] = fam[cg.MEMBERS].max(1)
        L = [f"# Runner-up gate (reference-class softmax) — {a.ds}\n",
             "| family | best member | " + " | ".join(NEW) + " |", "|---|---|" + "---|" * len(NEW)]
        for k, g in res.groupby("fold"):
            L.append(f"| {k} | {fam.loc[k, 'best']:.3f} | " + " | ".join(br.fmt(*br.ci(g[c])) for c in NEW) + " |")
        hard = cg.HARD_CIC if a.ds.startswith("ciciot") else [k for k in fam.index if fam.loc[k, "best"] < 0.5]
        gain = {k: fam.loc[k, NEW].max() - fam.loc[k, "best"] for k in hard if k in fam.index}
        elig = [k for k in fam.index if fam.loc[k, "n"] >= br.MIN_ROWS]
        macro = {c: br.fmt(*br.ci(res[res.fold.isin(elig)].groupby("seed")[c].mean())) for c in NEW}
        L.append("\ngain of best variant over best member: " + ", ".join(f"{k} {v:+.3f}" for k, v in gain.items()))
        L.append(f"families with gain >= 0.03: {sum(v >= 0.03 for v in gain.values())}; macro: {macro}")
        text = "\n".join(L)
        print(text)
        Path(f"{a.out}_{a.ds}.md").write_text(text + "\n")
        return
    Path(f"{a.out}_{a.ds}_s{a.seed}.json").write_text(json.dumps(one_seed(a.ds, a.variant, a.seed), indent=1,
                                                                 default=float))


if __name__ == "__main__":
    main()
