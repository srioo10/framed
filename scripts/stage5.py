"""Stages 4-5 (PREREG §9 H5): rehearsed alpha per outer fold, fused score on the outer test rows, H5 decision.

Per seed and outer fold k: alpha_sup from `rehearse_alpha` on fold k's pair models (eligible inner families only),
the rehearsal-selected single member from the same pair models, then fusion / XGB / MCDE / MLP-CE / selected
member scored on fold k's outer rows (thresholds on benign_cal). Also the Stage 2 best-alpha diagnostic.
Usage: .venv/bin/python scripts/stage5.py "RUN_GLOB" --out results/fusion/stage5_NAME [--boot 2000]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc  # noqa: E402
import bank_report as br  # noqa: E402

NAMES = br.FUSION
GRID = np.round(np.arange(0.30, 1.0001, 0.05), 2)
EQUAL = 0.6  # alpha_sup giving every one of the 5 members a 1/5 share
METHODS = ["fusion", "xgb", "mcde", "mlp_ce", "single", "rf"]
VERSUS = ["xgb", "mcde", "mlp_ce", "single"]
LABEL = {"Flood__Mirai": "unseen-family, joint hold-out (alpha tuned on BruteForce/Recon/Spoofing/Web)",
         "DoS": "partial-family (attack-type) hold-out", "DDoS": "partial-family (attack-type) hold-out"}


def mat(d):
    return d[NAMES].to_numpy(np.float64)


def rehearsal(run):
    """fold -> [(inner family j, b_rank, b_cal, pseudo-unknown rows of j)] over eligible j."""
    inner = {}
    for d in sorted(run.glob("config=pair_*")):
        df = pd.read_parquet(d / "part.parquet", columns=["role", "served", "family", *NAMES])
        rank, cal = mat(df[df.role == "rank"]), mat(df[df.role == "cal"])
        for fold, ps in df[df.role == "pseudo"].groupby("served"):
            assert ps.family.nunique() == 1, (d, fold)
            if len(ps) >= br.MIN_ROWS:
                inner.setdefault(fold, []).append((ps.family.iloc[0], rank, cal, mat(ps)))
    return inner


def choose(entries):
    """alpha, macro curve, flat flag, selected single member, and per-family numbers for the Stage 2 diagnostic."""
    alpha, curve, per_family, info = fc.rehearse_alpha([e[1:] for e in entries], NAMES, GRID)
    tpr = np.array([[(a[:, m] > fc.threshold(c[:, m], 0.01)).mean() for _, _, c, a in entries]
                    for m in range(len(NAMES))])
    pick, _ = fc.pick_alpha(np.arange(len(NAMES)), tpr, prior=0)  # last tie-break = member order
    # best tree with the threshold on rank + cal, as in union_ceiling_check (the Stage 2 gate)
    tree = [max((a[:, i] > fc.threshold(np.vstack([r, c])[:, i], 0.01)).mean() for i in (0, 1))
            for _, r, c, a in entries]
    return {"alpha": alpha, "curve": curve.tolist(), "flat": info["flat"], "single": NAMES[int(pick)],
            "inner": [e[0] for e in entries], "fused": per_family, "tree": np.array(tree),
            "single_macro": dict(zip(NAMES, tpr.mean(1).round(4).tolist()))}


def outer_scores(run, fold, alpha, single, alpha_min):
    df = pd.read_parquet(run / f"config=outer_{fold}" / "part.parquet")
    rank = df[df.role == "rank"]
    w, R = fc.shares_from_alpha(alpha, NAMES), mat(rank)
    out = {}
    for role in ("cal", "test_benign", "test_unknown"):
        d = df[df.role == role]
        out[role] = np.vstack([fc.budget_fusion_score(fc.tail_pvalues(R, mat(d)), w),
                               d.xgb.to_numpy(np.float64), br.mcde_score(rank, d, alpha_min),
                               d.mlp_ce.to_numpy(np.float64), d[single].to_numpy(np.float64),
                               d.rf.to_numpy(np.float64)])  # rows follow METHODS
    return out


def metrics(sc):
    r = {}
    for f in br.FPRS:
        if sc["cal"].shape[1] * f < 10:
            continue
        tau = fc.threshold(sc["cal"].T, f)
        assert ((sc["cal"] > tau[:, None]).mean(1) <= f).all()  # hard gate
        for i, m in enumerate(METHODS):
            r[(m, f"tpr@{f:g}")] = float((sc["test_unknown"][i] > tau[i]).mean())
            r[(m, f"test_fpr@{f:g}")] = float((sc["test_benign"][i] > tau[i]).mean())
    return r


def boot(store, seeds, folds, B, seed=0):
    """Two-level bootstrap of macro TPR@1% per method: seeds, then unknown and cal rows inside each seed."""
    rng = np.random.default_rng(seed)
    res = np.empty((B, len(METHODS)))
    for b in range(B):
        acc = np.zeros(len(METHODS))
        for s in rng.choice(seeds, len(seeds)):
            for f in folds:
                cal, unk = store[s, f]
                tau = np.quantile(cal[:, rng.integers(cal.shape[1], size=cal.shape[1])], 0.99, axis=1,
                                  method="higher")
                acc += (unk[:, rng.integers(unk.shape[1], size=unk.shape[1])] > tau[:, None]).mean(1)
        res[b] = acc / (len(seeds) * len(folds))
    return res


def fmt(x):
    return br.fmt(*br.ci(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    runs = sorted(Path(p) for p in glob.glob(a.runs))
    rows, picks, diag, store = [], [], [], {}
    for run in runs:
        s = run.name.rsplit("_s", 1)[1]
        alpha_min = 0.01 if "unsw_official" in run.name else 0.001
        for fold, entries in sorted(rehearsal(run).items()):
            c = choose(entries)
            sc = outer_scores(run, fold, c["alpha"], c["single"], alpha_min)
            picks.append({"seed": s, "fold": fold, "alpha": c["alpha"], "flat": c["flat"], "single": c["single"],
                          "curve": c["curve"], "inner": c["inner"], "single_macro": c["single_macro"]})
            gi = list(GRID).index(EQUAL)
            ai = list(GRID).index(c["alpha"])
            mbest = int(c["fused"].mean(1).argmax())
            for k, j in enumerate(c["inner"]):
                diag.append({"seed": s, "fold": fold, "inner": j, "equal": c["fused"][gi, k] - c["tree"][k],
                             "oracle": c["fused"][:, k].max() - c["tree"][k],
                             "fold_best": c["fused"][mbest, k] - c["tree"][k],
                             "rehearsed": c["fused"][ai, k] - c["tree"][k]})
            rows.append({"seed": s, "fold": fold, "n": sc["test_unknown"].shape[1], **metrics(sc)})
            store[s, fold] = (sc["cal"], sc["test_unknown"])
        print("done", run.name, flush=True)

    res = pd.DataFrame(rows)
    seeds = sorted(res.seed.unique())
    n_min = res.groupby("fold").n.min()
    main_folds = [f for f in n_min.index if f not in LABEL and n_min[f] >= br.MIN_ROWS]
    L = [f"# Stage 5 — {a.runs}\n", f"seeds {seeds}; eligible main families {main_folds}; "
         f"excluded (< {br.MIN_ROWS} test rows): {[f for f in n_min.index if f not in LABEL and f not in main_folds]}\n"]

    # ---- alpha, curve, flat, selected member ---------------------------------------------------------------
    pk = pd.DataFrame(picks)
    L.append("## Rehearsed alpha per outer fold\n")
    L.append("| fold | alpha per seed | flat (seeds) | selected single member per seed | inner families |")
    L.append("|---|---|---|---|---|")
    for fold, g in pk.groupby("fold"):
        L.append(f"| {fold.replace('__', '+')} | {', '.join(f'{x:.2f}' for x in g.alpha)} | {int(g.flat.sum())}/{len(g)} "
                 f"| {', '.join(g.single)} | {', '.join(g.inner.iloc[0])} |")
    L.append("\n### Macro rehearsal curve (TPR@1% on inner families, mean over seeds)\n")
    L.append("| fold | " + " | ".join(f"{x:.2f}" for x in GRID) + " |")
    L.append("|---|" + "---|" * len(GRID))
    for fold, g in pk.groupby("fold"):
        cur = np.mean(np.vstack(g.curve), 0)
        L.append(f"| {fold.replace('__', '+')} | " + " | ".join(f"{x:.3f}" for x in cur) + " |")

    # ---- Stage 2 best-alpha diagnostic ----------------------------------------------------------------------
    dg = pd.DataFrame(diag)
    dg = dg[~dg.fold.isin(LABEL)]
    per = dg.groupby(["inner", "seed"])[["equal", "fold_best", "rehearsed", "oracle"]].mean()
    L.append("\n## Stage 2 diagnostic: fused TPR@1% minus best tree on inner families (not a gate)\n")
    L.append("equal = alpha 0.60 (1/5 each); fold-best = alpha with the best macro on that fold's inner families; "
             "rehearsed = the Stage 4 pick; oracle = best alpha per (fold, family), an upper bound.\n")
    L.append("| inner family | equal share | fold-best alpha | rehearsed alpha | oracle alpha |\n|---|---|---|---|---|")
    for j, g in per.groupby(level=0):
        L.append(f"| {j} | " + " | ".join(fmt(g[c]) for c in per.columns) + " |")
    mac = per.groupby(level=1).mean()
    L.append("| **macro** | " + " | ".join(fmt(mac[c]) for c in per.columns) + " |")

    # ---- per-family tables -----------------------------------------------------------------------------------
    L.append("\n## Outer held-out families: TPR at each FPR (mean [95% t-CI] over seeds)\n")
    for f in br.FPRS:
        col = f"tpr@{f:g}"
        if ("fusion", col) not in res:
            continue
        L.append(f"### FPR {f:g}\n")
        L.append("| fold | n | " + " | ".join(METHODS) + " | fusion test FPR |\n|---|---|" + "---|" * (len(METHODS) + 1))
        for fold, g in res.groupby("fold"):
            tag = f" ({LABEL[fold]})" if fold in LABEL else ""
            L.append(f"| {fold.replace('__', '+')}{tag} | {int(g.n.min())} | " +
                     " | ".join(fmt(g[(m, col)]) for m in METHODS) + f" | {fmt(g[('fusion', f'test_fpr@{f:g}')])} |")
        L.append("")

    # ---- H5 decision ------------------------------------------------------------------------------------------
    m1 = res[res.fold.isin(main_folds)]
    macro = m1.groupby("seed")[[(m, "tpr@0.01") for m in METHODS]].mean()
    bs = boot(store, seeds, main_folds, a.boot)
    fam = m1.groupby("fold")[[(m, "tpr@0.01") for m in METHODS]].mean()
    L.append("## H5 decision (macro TPR@1% over eligible main families)\n")
    L.append("| fusion minus | mean | 5-seed t-CI | two-level bootstrap CI | CI used (wider) | CI > 0 | worst family diff |")
    L.append("|---|---|---|---|---|---|---|")
    dec = {}
    for v in VERSUS:
        d = macro[("fusion", "tpr@0.01")] - macro[(v, "tpr@0.01")]
        m, lo, hi = br.ci(d)
        bd = bs[:, 0] - bs[:, METHODS.index(v)]
        blo, bhi = np.percentile(bd, [2.5, 97.5])
        ulo, uhi = (lo, hi) if hi - lo >= bhi - blo else (blo, bhi)
        worst = (fam[("fusion", "tpr@0.01")] - fam[(v, "tpr@0.01")])
        dec[v] = {"mean": m, "t": [lo, hi], "boot": [blo, bhi], "used": [ulo, uhi], "pass": bool(ulo > 0),
                  "worst": float(worst.min()), "worst_fold": worst.idxmin()}
        L.append(f"| {v} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{blo:+.3f}, {bhi:+.3f}] | [{ulo:+.3f}, {uhi:+.3f}] | "
                 f"{'yes' if ulo > 0 else 'no'} | {worst.min():+.3f} ({worst.idxmin().replace('__', '+')}) |")
    h5 = dec["xgb"]["pass"] and dec["mcde"]["pass"] and min(dec["xgb"]["worst"], dec["mcde"]["worst"]) >= -0.03
    adds = dec["mlp_ce"]["pass"] and dec["single"]["pass"]
    L.append(f"\nmacro TPR@1%: " + ", ".join(f"{m} {fmt(macro[(m, 'tpr@0.01')])}" for m in METHODS))
    L.append(f"\n**H5 (beat XGB and MCDE, CI > 0, no family worse by > 0.03): {'PASS' if h5 else 'FAIL'}**  ")
    L.append(f"**Fusion adds something (beats MLP-CE alone and the rehearsal-selected member): "
             f"{'YES' if adds else 'NO'}**")
    text = "\n".join(L)
    print(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".md").write_text(text + "\n")
    res.columns = ["|".join(c) if isinstance(c, tuple) else c for c in res.columns]
    a.out.with_suffix(".json").write_text(json.dumps(
        {"picks": picks, "outer": res.to_dict("records"), "diag": pd.DataFrame(diag).to_dict("records"),
         "decision": dec, "h5": h5, "adds": adds, "main_folds": main_folds}, indent=1, default=float))


if __name__ == "__main__":
    main()
