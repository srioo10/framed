"""Stage 2 (ceiling gate) and Stage 3 (member tails, abort gate) tables from score-bank runs (PREREG §9 H5).

Decisions use rehearsal (pseudo-unknown) families only; outer tables are diagnostics.
Usage: .venv/bin/python scripts/bank_report.py RUN_GLOB [--json out.json]
       e.g. "runs/bank/unsw_raw_ct_causal_cap40000_s*"
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import fusion_checks as fc  # noqa: E402

FPRS = (0.001, 0.01, 0.05)
FUSION = ["rf", "xgb", "mlp_ce", "iforest", "clust"]
OUTER = ["rf", "xgb", "xgb_tail", "mlp_ce", "mlp_kd", "iforest", "clust", "mcde"]
MIN_ROWS = 300
EXTRA_FOLDS = {"Flood+Mirai", "DoS", "DDoS"}  # CICIoT joint / attack-type hold-outs, reported apart
W_MCDE = 0.9


def mcde_score(rank, x, alpha_min):
    """MCDE fused score S = -min(u_sup / w, u_mode / (1 - w)) with mode-conditioned distance references."""
    n_min = math.ceil(5 / alpha_min)
    u_sup = fc.tail_pvalues(rank[["mcde_sup"]].to_numpy(np.float64), x[["mcde_sup"]].to_numpy(np.float64))[:, 0]
    # one-hot MCDE modes when the bank has them; without categorical features they equal clust / mode
    dc, mc = ("mcde_dist", "mcde_mode") if "mcde_dist" in rank else ("clust", "mode")
    u_mode = fc.tail_pvalues_by_mode(rank[dc].to_numpy(np.float64), rank[mc].to_numpy(),
                                     x[dc].to_numpy(np.float64), x[mc].to_numpy(), n_min)
    return -np.minimum(u_sup / W_MCDE, u_mode / (1 - W_MCDE))


def member_rows(rank, cal, test_b, unk, members, alpha_min):
    """TPR at each FPR (cal threshold), realised benign-test FPR, pAUC<=1%, AUROC per member."""
    out = {}
    for m in members:
        if m == "mcde":
            s_cal, s_b, s_u = (mcde_score(rank, d, alpha_min) for d in (cal, test_b, unk))
        else:
            s_cal, s_b, s_u = (d[m].to_numpy(np.float64) for d in (cal, test_b, unk))
        r = {}
        for f in FPRS:
            if len(s_cal) * f < 10:
                continue
            tau = fc.threshold(s_cal, f)
            assert (s_cal > tau).mean() <= f  # hard gate
            r[f"tpr@{f:g}"] = float((s_u > tau).mean())
            r[f"test_fpr@{f:g}"] = float((s_b > tau).mean())
        if s_b is not None:
            y = np.r_[np.zeros(len(s_b)), np.ones(len(s_u))]
            s = np.r_[s_b, s_u]
            r["auroc"] = float(roc_auc_score(y, s))
            r["pauc@0.01"] = float(roc_auc_score(y, s, max_fpr=0.01))
        out[m] = r
    return out


def pseudo_rows(rank, cal, ps):
    """Stage 2/3 numbers on one pseudo-unknown family: ceiling headroom + member TPR@1% minus best tree."""
    names = FUSION
    br, bc, a = (d[names].to_numpy(np.float64) for d in (rank, cal, ps))
    ceil = fc.union_ceiling_check(br, bc, a, names)
    tau = fc.threshold(bc, 0.01)
    tpr = dict(zip(names, (a > tau).mean(0)))
    best_tree = max(tpr["rf"], tpr["xgb"])
    return {"headroom_vs_trees": ceil["headroom_vs_trees"], "headroom_vs_best_member": ceil["headroom_vs_best_member"],
            "union_tpr": ceil["bonferroni_union_tpr"], "fused_tpr": ceil["recalibrated_equal_fusion_tpr"],
            **{f"gain_{m}": float(tpr[m] - best_tree) for m in names if m not in ("rf", "xgb")},
            **{f"tpr_{m}": float(tpr[m]) for m in names}}


def one_seed(run):
    """Per-seed outer member table and pseudo (inner) table."""
    alpha_min = 0.01 if "unsw_official" in run.name else 0.001
    outer, inner = [], []
    for d in sorted(run.glob("config=*")):
        kind, name = d.name[len("config="):].split("_", 1)
        df = pd.read_parquet(d / "part.parquet")
        rank, cal = df[df.role == "rank"], df[df.role == "cal"]
        if kind == "outer":
            unk = df[df.role == "test_unknown"]
            tb = df[df.role == "test_benign"]
            for m, r in member_rows(rank, cal, tb, unk, [c for c in OUTER if c in df or c == "mcde"],
                                    alpha_min).items():
                outer.append({"heldout": name.replace("__", "+"), "member": m, "n": len(unk), **r})
        else:
            for fold, ps in df[df.role == "pseudo"].groupby("served"):
                assert ps.family.nunique() == 1, (fold, ps.family.unique())
                j = ps.family.iloc[0]
                row = {"fold": fold.replace("__", "+"), "inner": j, "n": len(ps)}
                if len(ps) >= MIN_ROWS:
                    row.update(pseudo_rows(rank, cal, ps))
                inner.append(row)
    return pd.DataFrame(outer), pd.DataFrame(inner)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def fmt(m, lo, hi):
    return f"{m:.3f}" if not np.isfinite(lo) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = sorted(Path(p) for p in glob.glob(a.runs))
    seeds = {}
    for r in runs:
        seeds[r.name.rsplit("_s", 1)[1]] = one_seed(r)
    print(f"runs: {[r.name for r in runs]}\n")

    # ---- Stage 2: inner-family ceiling gate --------------------------------------------------------------
    inner_all = pd.concat([v[1].assign(seed=s) for s, v in seeds.items()], ignore_index=True)
    if "fold" not in inner_all:  # outer-only runs (ct_* variants, unsw_official): no rehearsal rows
        inner_all = pd.DataFrame({"fold": [], "inner": [], "n": [], "seed": [], "headroom_vs_trees": [],
                                  "headroom_vs_best_member": [], "union_tpr": [], "fused_tpr": []})
        print("(outer-only run: no rehearsal rows, Stage 2/3 gates not applicable)")
    extra = inner_all[inner_all.fold.isin(EXTRA_FOLDS)]
    inner = inner_all[~inner_all.fold.isin(EXTRA_FOLDS)]
    elig = inner.dropna(subset=["headroom_vs_trees"]) if "headroom_vs_trees" in inner else inner.iloc[:0]
    print("## Stage 2 — ceiling on rehearsal families (gate)\n")
    print("| inner family | folds | headroom vs trees | seeds > 0 | headroom vs best member | union TPR | fused TPR |")
    print("|---|---|---|---|---|---|---|")
    per_seed_j = elig.groupby(["inner", "seed"]).mean(numeric_only=True)
    go_family = False
    for j, g in per_seed_j.groupby(level=0):
        h = g["headroom_vs_trees"]
        pos = int((h > 0).sum())
        go_family |= pos >= 4
        print(f"| {j} | {int(elig[elig.inner == j].fold.nunique())} | {fmt(*ci(h))} | {pos}/{len(h)} | "
              f"{fmt(*ci(g['headroom_vs_best_member']))} | {fmt(*ci(g['union_tpr']))} | {fmt(*ci(g['fused_tpr']))} |")
    macro = per_seed_j["headroom_vs_trees"].groupby(level=1).mean()
    m, lo, hi = ci(macro)
    go = (np.isfinite(lo) and lo > 0) or go_family
    print(f"\nmacro headroom vs trees: {fmt(m, lo, hi)} -> **{'GO' if go else 'FEATURE-CAPPED'}** "
          f"(rule: macro CI > 0, or one family > 0 on >= 4/5 seeds; {len(macro)} seed(s))")
    small = inner[inner.n < MIN_ROWS]
    if len(small):
        print(f"excluded (< {MIN_ROWS} rows): " + ", ".join(f"{r.inner} in {r.fold} ({r.n})" for r in small.itertuples()))

    # ---- Stage 3: member gain on rehearsal families (abort gate) -------------------------------------------
    print("\n## Stage 3 — member TPR@1% minus max(RF, XGB) on rehearsal families (gate)\n")
    gains = [c for c in elig.columns if c.startswith("gain_")]
    print("| inner family | " + " | ".join(c[5:] for c in gains) + " |")
    print("|---|" + "---|" * len(gains))
    any_pos = False
    for j, g in per_seed_j.groupby(level=0):
        cells = []
        for c in gains:
            mm, l, h = ci(g[c])
            any_pos |= np.isfinite(l) and l > 0
            cells.append(fmt(mm, l, h))
        print(f"| {j} | " + " | ".join(cells) + " |")
    print(f"\nany member CI > 0: {any_pos} -> **{'continue' if any_pos else 'ABORT fusion'}** "
          f"(needs >= 2 seeds for a CI)")

    # ---- extra hold-outs: rehearsal numbers per fold (alpha for these folds is tuned on these inner families)
    if len(extra) and "headroom_vs_trees" in extra:
        print("\n## Extra hold-outs — rehearsal families per fold\n")
        g = extra.dropna(subset=["headroom_vs_trees"]).groupby(["fold", "inner", "seed"]).mean(numeric_only=True)
        cols = ["headroom_vs_trees"] + [c for c in g.columns if c.startswith("gain_")]
        print("| fold | inner | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
        for (fold, j), gg in g.groupby(level=[0, 1]):
            print(f"| {fold} | {j} | " + " | ".join(fmt(*ci(gg[c])) for c in cols) + " |")

    # ---- outer diagnostics ---------------------------------------------------------------------------------
    outer = pd.concat([v[0].assign(seed=s) for s, v in seeds.items()], ignore_index=True)
    print("\n## Outer held-out families (diagnostic only)\n")
    cols = [c for c in outer.columns if c.startswith(("tpr@", "pauc", "auroc"))]
    for held, g in outer.groupby("heldout"):
        print(f"### {held} (n = {int(g.n.iloc[0])})\n")
        print("| member | " + " | ".join(cols) + " | test FPR@0.01 |")
        print("|---|" + "---|" * (len(cols) + 1))
        for mem, gm in g.groupby("member", sort=False):
            cells = [fmt(*ci(gm[c])) for c in cols]
            print(f"| {mem} | " + " | ".join(cells) + f" | {fmt(*ci(gm['test_fpr@0.01']))} |")
        print()
    # attack-present false-alarm cost: realised benign-test FPR minus the 1% target
    drift = outer.pivot_table(index="heldout", columns="member", values="test_fpr@0.01", aggfunc="mean") - 0.01
    print("## Benign-test FPR minus 1% target (attack-present false-alarm cost when ct_* are causal)\n")
    print(drift.round(4).to_string())
    kd = outer[outer.member.isin(["mlp_ce", "mlp_kd"])].pivot_table(index="heldout", columns="member",
                                                                    values="tpr@0.01", aggfunc="mean")
    print("\n## KD vs CE, TPR@1%\n")
    print(kd.round(3).to_string())
    if a.json:
        a.json.write_text(json.dumps({"inner": inner_all.to_dict("records"), "outer": outer.to_dict("records")},
                                     indent=1, default=float))


if __name__ == "__main__":
    main()
