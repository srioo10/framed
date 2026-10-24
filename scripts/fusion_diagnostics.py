"""Post hoc CICIoT fusion diagnostics on the existing banks (PREREG §9, 2026-10-24 labels row):
(i) fixed alpha 0.9 and equal shares beside the rehearsed alpha; (ii) H5 differences with each family dropped;
(iii) leave-one-member-out fusion (alpha re-rehearsed per reduced set) and MCDE's branches plus MLP-CE.
Usage: .venv/bin/python scripts/fusion_diagnostics.py "RUN_GLOB" --stage5 results/fusion/stage5_NAME.json
       --out results/fusion/diagnostics_NAME [--boot 2000]
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc  # noqa: E402
import bank_report as br  # noqa: E402
import stage5 as s5  # noqa: E402

NAMES = s5.NAMES
BASE = ["xgb", "mcde", "mlp_ce"]


def fused(rank, d, names, alpha):
    R = rank[names].to_numpy(np.float64)
    return fc.budget_fusion_score(fc.tail_pvalues(R, d[names].to_numpy(np.float64)),
                                  fc.shares_from_alpha(alpha, names))


def mcde_plus_mlp(rank, d, alpha_min, w=0.9):
    """MCDE with MLP-CE added to its supervised branch: HGB and MLP-CE split w equally."""
    u = fc.tail_pvalues(rank[["mcde_sup", "mlp_ce"]].to_numpy(np.float64), d[["mcde_sup", "mlp_ce"]].to_numpy(np.float64))
    dc, mc = ("mcde_dist", "mcde_mode") if "mcde_dist" in rank else ("clust", "mode")
    um = fc.tail_pvalues_by_mode(rank[dc].to_numpy(np.float64), rank[mc].to_numpy(), d[dc].to_numpy(np.float64),
                                 d[mc].to_numpy(), math.ceil(5 / alpha_min))
    return -np.minimum(np.minimum(u[:, 0], u[:, 1]) / (w / 2), um / (1 - w))


def variants(run, alpha_rehearsed):
    """fold -> {variant: alpha or None} and the reduced-set alphas re-rehearsed on the pair models."""
    inner = s5.rehearsal(run)
    out = {}
    for fold, entries in inner.items():
        if fold not in alpha_rehearsed:  # extra hold-outs are not part of these diagnostics
            continue
        v = {"rehearsed": (NAMES, alpha_rehearsed[fold]), "fixed 0.90": (NAMES, 0.9), "equal (0.60)": (NAMES, 0.6)}
        for m in NAMES:
            keep = [c for c in NAMES if c != m]
            idx = [NAMES.index(c) for c in keep]
            a, *_ = fc.rehearse_alpha([(r[:, idx], c[:, idx], x[:, idx]) for _, r, c, x in entries], keep, s5.GRID)
            v[f"without {m}"] = (keep, a)
        out[fold] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--stage5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    st = json.loads(a.stage5.read_text())
    alphas = {(p["seed"], p["fold"]): p["alpha"] for p in st["picks"]}
    main_folds = st["main_folds"]
    rows, store, picked = [], {}, []
    names = None
    for run in sorted(Path(p) for p in glob.glob(a.runs)):
        s = run.name.rsplit("_s", 1)[1]
        alpha_min = 0.01 if "unsw_official" in run.name else 0.001
        v = variants(run, {f: alphas[s, f] for f in main_folds})
        for fold in main_folds:
            df = pd.read_parquet(run / f"config=outer_{fold}" / "part.parquet")
            rank = df[df.role == "rank"]
            sc = {}
            for role in ("cal", "test_benign", "test_unknown"):
                d = df[df.role == role]
                cols = {name: fused(rank, d, keep, al) for name, (keep, al) in v[fold].items()}
                cols["MCDE + MLP-CE (w 0.9)"] = mcde_plus_mlp(rank, d, alpha_min)
                cols["xgb"], cols["mlp_ce"] = d.xgb.to_numpy(np.float64), d.mlp_ce.to_numpy(np.float64)
                cols["mcde"] = br.mcde_score(rank, d, alpha_min)
                names = list(cols)
                sc[role] = np.vstack(list(cols.values()))
            tau = fc.threshold(sc["cal"].T, 0.01)
            r = {"seed": s, "fold": fold}
            for i, m in enumerate(names):
                r[m] = float((sc["test_unknown"][i] > tau[i]).mean())
                r[f"{m}|fpr"] = float((sc["test_benign"][i] > tau[i]).mean())
            rows.append(r)
            store[s, fold] = (sc["cal"], sc["test_unknown"])
            picked.append({"seed": s, "fold": fold, **{k: al for k, (_, al) in v[fold].items() if k.startswith("without")}})
        print("done", run.name, flush=True)

    res = pd.DataFrame(rows)
    seeds = sorted(res.seed.unique())
    fus = [n for n in names if n not in BASE]
    s5.METHODS = names  # boot() labels rows by METHODS
    bs = s5.boot(store, seeds, main_folds, a.boot)
    macro = res.groupby("seed")[names].mean()
    L = [f"# Post hoc fusion diagnostics — {a.runs}\n", f"TPR@1%FPR; eligible main families {main_folds}; "
         "CI used = wider of the 5-seed t-CI and the two-level bootstrap (PREREG §9 Stage 5 rule).\n"]

    def table(title, variants_):
        L.append(f"## {title}\n")
        L.append("| variant | macro TPR@1% | minus XGB | minus MCDE | minus MLP-CE |\n|---|---|---|---|---|")
        for f in variants_:
            cells = []
            for b in BASE:
                d = macro[f] - macro[b]
                m, lo, hi = br.ci(d)
                bd = bs[:, names.index(f)] - bs[:, names.index(b)]
                blo, bhi = np.percentile(bd, [2.5, 97.5])
                ulo, uhi = (lo, hi) if hi - lo >= bhi - blo else (blo, bhi)
                cells.append(f"{m:+.3f} [{ulo:+.3f}, {uhi:+.3f}]{' ✔' if ulo > 0 else ''}")
            L.append(f"| {f} | {br.fmt(*br.ci(macro[f]))} | " + " | ".join(cells) + " |")
        L.append("")

    table("(i) Alpha: rehearsed vs fixed", ["rehearsed", "fixed 0.90", "equal (0.60)"])
    table("(iii) Leave-one-member-out (alpha re-rehearsed) and MCDE + MLP-CE",
          ["rehearsed"] + [f for f in fus if f.startswith("without")] + ["MCDE + MLP-CE (w 0.9)"])
    fam = res.groupby("fold")[names].mean()
    L.append("### Per family, TPR@1% (5-seed mean)\n")
    L.append("| variant | " + " | ".join(main_folds) + " |\n|---|" + "---|" * len(main_folds))
    for f in names:
        L.append(f"| {f} | " + " | ".join(f"{fam.loc[k, f]:.3f}" for k in main_folds) + " |")
    pa = pd.DataFrame(picked)
    L.append("\nre-rehearsed alpha per reduced set (mean over seeds and folds): " +
             ", ".join(f"{c} {pa[c].mean():.2f}" for c in pa.columns if c.startswith("without")))
    L.append("\nrealised benign-test FPR at 1% (mean over seeds and folds): " +
             ", ".join(f"{n} {res[f'{n}|fpr'].mean():.4f}" for n in names))

    L.append("\n## (ii) H5 with each family dropped in turn (rehearsed fusion; 5-seed t-CI)\n")
    L.append("| dropped family | minus XGB | minus MCDE | minus MLP-CE |\n|---|---|---|---|")
    for k in ["(none)"] + main_folds:
        g = res[res.fold != k].groupby("seed")[names].mean()
        L.append(f"| {k} | " + " | ".join(br.fmt(*br.ci(g["rehearsed"] - g[b])) for b in BASE) + " |")
    text = "\n".join(L)
    print(text)
    a.out.with_suffix(".md").write_text(text + "\n")
    a.out.with_suffix(".json").write_text(json.dumps({"outer": res.to_dict("records"), "alphas": picked},
                                                     indent=1, default=float))


if __name__ == "__main__":
    main()
