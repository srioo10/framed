"""Host-context arm report (PREREG §9 host-context row): nf_unsw banks none / host / host_iso / host_perm.

Per variant, family and seed: TPR@1%FPR of every member (cal threshold) and realised benign-test FPR; then the
claim rule: paired (same seed) difference host - none for the best member and for each member.
Usage: .venv/bin/python scripts/host_report.py [--ds nf_unsw] --out results/host/nf_unsw
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc  # noqa: E402
import bank_report as br  # noqa: E402
import score_bank as sb  # noqa: E402

MEMBERS = ["rf", "xgb", "mlp_ce", "iforest", "clust", "mcde"]
VARIANTS = ["none", "host", "host_iso", "host_perm"]


def rows(ds):
    out = []
    for v in VARIANTS:
        for run in sorted(sb.OUT.glob(f"{ds}_{v}_cap40000_s*")):
            seed = int(run.name.rsplit("_s", 1)[1])
            for part in sorted(run.glob("config=outer_*/part.parquet")):
                k = part.parent.name.removeprefix("config=outer_")
                d = pd.read_parquet(part)
                r = {x: d[d.role == x] for x in ("rank", "cal", "test_benign", "test_unknown")}
                for m in MEMBERS:
                    s = {x: (br.mcde_score(r["rank"], g, 0.001) if m == "mcde" else g[m].to_numpy(np.float64))
                         for x, g in r.items()}
                    th = fc.threshold(s["cal"], 0.01)
                    out.append({"variant": v, "seed": seed, "fold": k, "member": m, "n": len(r["test_unknown"]),
                                "tpr": float((s["test_unknown"] > th).mean()),
                                "fpr_test": float((s["test_benign"] > th).mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nf_unsw")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = rows(a.ds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(f"{a.out}_rows.csv", index=False)
    best = res.loc[res.groupby(["variant", "seed", "fold"]).tpr.idxmax()]
    f = lambda x: br.fmt(*br.ci(x)) if len(x) > 1 else f"{x.iloc[0]:.3f}"  # noqa: E731
    seeds = sorted(res.seed.unique())
    have = [v for v in VARIANTS if v in set(res.variant)]
    L = [f"# Host-context arm — {a.ds} (seeds {seeds})\n", "TPR@1%FPR on the held-out family, best member per seed "
         "(member most often best in brackets); 5-seed mean [CI]\n",
         "| family | n | " + " | ".join(have) + " | host − none (paired) |", "|---|---|" + "---|" * (len(have) + 1)]
    for k, g in best.groupby("fold"):
        cells = []
        for v in have:
            gv = g[g.variant == v]
            cells.append(f"{f(gv.tpr)} ({gv.member.mode().iloc[0]})" if len(gv) else "—")
        p = g.pivot_table(index="seed", columns="variant", values="tpr")
        diff = f(p["host"] - p["none"]) if {"host", "none"} <= set(p.columns) and p[["host", "none"]].notna().all(1).any() else "—"
        L.append(f"| {k} | {int(g.n.min())} | " + " | ".join(cells) + f" | {diff} |")
    L += ["\nSame member, TPR@1%FPR mean over seeds (rows: member × variant)\n"]
    piv = res.pivot_table(index=["member", "variant"], columns="fold", values="tpr").round(3)
    L.append("```\n" + piv.to_string() + "\n```")
    L += ["\nRealised benign-test FPR at the cal threshold (target 0.01), mean over families and seeds\n"]
    L.append("```\n" + res.pivot_table(index="member", columns="variant", values="fpr_test").round(4).to_string() + "\n```")
    text = "\n".join(L)
    print(text)
    Path(f"{a.out}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
