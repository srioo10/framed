"""unsw_raw fingerprint ablation (PREREG §9, 2026-10-24 row): RF, XGB and MLP-CE refit on the headline variant
(ct_causal) with feature sets dropped, same folds / capped rows / early-stopping split / cal threshold as the bank.
Baseline = the bank's stored scores (identical rows and fits). TPR@1%FPR on each held-out family.
Usage: .venv/bin/python scripts/fingerprint_ablation.py --seeds 0 1 2 3 4 --out results/fusion/fingerprint_unsw_raw
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
from nids import fusion_checks as fc, models  # noqa: E402
import bank_report as br  # noqa: E402
import score_bank as sb  # noqa: E402

SETS = {"timing": ["tcprtt", "synack", "ackdat"], "seqwin": ["stcpb", "dtcpb", "swin", "dwin"]}
SETS["both"] = SETS["timing"] + SETS["seqwin"]
MEMBERS = ["rf", "xgb", "mlp_ce"]


def tpr(cal, unk, tb):
    tau = fc.threshold(cal, 0.01)
    return float((unk > tau).mean()), float((tb > tau).mean())


def one_seed(seed):
    bank = sb.Bank("unsw_raw", seed, "ct_causal", 40000, 200000)
    run = sb.OUT / f"unsw_raw_ct_causal_cap40000_s{seed}"
    out = []
    for k in bank.families:
        P = (k,)
        part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet", columns=["role", "row_id", *MEMBERS])
        rows = {r: part.row_id[part.role == r].to_numpy() for r in ("cal", "test_unknown", "test_benign")}
        base = {"seed": seed, "heldout": k, "drop": "none", "n": len(rows["test_unknown"])}
        for m in MEMBERS:
            base[m], base[f"{m}_test_fpr"] = tpr(*(part[m][part.role == r].to_numpy(np.float64)
                                                   for r in ("cal", "test_unknown", "test_benign")))
        out.append(base)
        fit, es = bank.fit_rows(P)
        known = sorted(set(bank.fam[fit]))
        code = {f: i for i, f in enumerate(known)}
        bi = code["Benign"]
        frames = {"fit": bank.frame(fit, sb.role_removal(P, "fit")), "es": bank.frame(es, sb.role_removal(P, "es")),
                  **{r: bank.frame(idx, sb.role_removal(P, r)) for r, idx in rows.items()}}
        yf, ye = (np.array([code[f] for f in bank.fam[idx]]) for idx in (fit, es))
        for name, cols in SETS.items():
            d = {r: f.drop(columns=cols) for r, f in frames.items()}
            prep = models.Prep().fit(d["fit"])
            X = {r: prep.transform(f) for r, f in d.items()}
            sc = {"rf": models.rf(seed).fit(X["fit"], yf), "xgb": models.xgb(seed).fit(X["fit"], yf),
                  "mlp_ce": models.train_mlp(X["fit"], yf, X["es"], ye, len(known), seed=seed)}
            res = {"seed": seed, "heldout": k, "drop": name, "n": base["n"]}
            for m, mdl in sc.items():
                p = {r: 1 - (models.mlp_proba(mdl, X[r]) if m == "mlp_ce" else mdl.predict_proba(X[r]))[:, bi]
                     for r in rows}
                res[m], res[f"{m}_test_fpr"] = tpr(p["cal"], p["test_unknown"], p["test_benign"])
            out.append(res)
            print(f"s{seed} {k:18s} drop {name:6s} " + " ".join(f"{m} {res[m]:.3f} (base {base[m]:.3f})"
                                                              for m in MEMBERS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = pd.DataFrame([r for s in a.seeds for r in one_seed(s)])
    a.out.with_suffix(".json").write_text(json.dumps(res.to_dict("records"), indent=1, default=float))
    res["best"] = res[MEMBERS].max(axis=1)
    n_min = res.groupby("heldout").n.min()
    elig = [k for k in n_min.index if n_min[k] >= br.MIN_ROWS]
    L = ["# unsw_raw fingerprint ablation (ct_causal), TPR@1%FPR on the held-out family\n",
         f"5-seed mean [95% t-CI]; eligible families (>= {br.MIN_ROWS} test rows): {elig}\n"]
    for name in ["none", *SETS]:
        g = res[res["drop"] == name]
        L.append(f"## drop = {name}{'' if name == 'none' else ': ' + ', '.join(SETS[name])}\n")
        L.append("| held-out | n | " + " | ".join(MEMBERS) + " | best of three |\n|---|---|" + "---|" * 4)
        for k, gk in g.groupby("heldout"):
            L.append(f"| {k} | {int(gk.n.min())} | " + " | ".join(br.fmt(*br.ci(gk[c])) for c in [*MEMBERS, "best"]) + " |")
        L.append("")
    L.append("## Decision (PREREG §9 rule): drop in best-of-three TPR@1% vs none\n")
    L.append("| set | macro drop (eligible) | 5-seed t-CI | worst family drop | treat as fingerprint |\n|---|---|---|---|---|")
    base = res[res["drop"] == "none"].set_index(["seed", "heldout"]).best
    verdict = {}
    for name in SETS:
        d = base - res[res["drop"] == name].set_index(["seed", "heldout"]).best
        d = d[d.index.get_level_values(1).isin(elig)]
        m, lo, hi = br.ci(d.groupby(level=0).mean())
        fam = d.groupby(level=1).mean()
        hit = (m >= 0.05 and lo > 0) or fam.max() >= 0.10
        verdict[name] = hit
        L.append(f"| {name} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {fam.max():+.3f} ({fam.idxmax()}) | "
                 f"{'**YES**' if hit else 'no'} |")
    lc = ROOT / "results/fusion/leak_controls_unsw_raw.json"
    if lc.exists():
        single = pd.DataFrame([{**r["single"], "heldout": r["heldout"]} for r in json.loads(lc.read_text())])
        cols = SETS["both"]
        L.append("\n## Single-feature controls for these 7 features (decision tree, TPR@1%, mean over seeds)\n")
        L.append("| held-out | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for k, g in single.groupby("heldout"):
            L.append(f"| {k} | " + " | ".join(f"{g[c].mean():.3f}" for c in cols) + " |")
    text = "\n".join(L)
    print(text)
    a.out.with_suffix(".md").write_text(text + "\n")


if __name__ == "__main__":
    main()
