"""Adaptive-attacker characterization (PREREG §9 ADV). Reads the perturbed score banks; no training.

Per (dataset, perturbation, detector, method): macro host TPR (attacker host with >= 1 alarmed host-day, macro over
held-out families, mean over seeds), macro host-day TPR, realised benign host-day FPR. Baseline = the same chain's
dilate:1 banks. Claim rule: robust at a perturbation only where host TPR >= 0.5 x baseline on >= 2 of 3 datasets; a
dataset whose baseline host TPR is 0 gives no support either way.
Usage: .venv/bin/python scripts/adversarial.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import discover  # noqa: E402
import score_bank as sb  # noqa: E402

TAGS = ["_dilate1", "_dilate10", "_dilate100", "_dilate1000", "_cover1", "_cover2", "_cover4", "_cover8"]
OUT = ROOT / "results" / "adversarial"


def measure(ds, tag):
    H, F = [], []
    for s in (1, 2, 3, 4):
        if (sb.OUT / f"{ds}_host_cap40000_s{s}{tag}" / "manifest_outer.json").exists():  # written when the bank is done
            h, f = discover.hits(ds, s, tag)
            H.append(h)
            F.append(f)
    if not H:
        return None
    H, F = pd.concat(H), pd.concat(F)
    cols = [c for c in H.columns if c.endswith(("_P", "_A"))]
    host = H.groupby(["seed", "fold", "host"])[cols].any().groupby(["seed", "fold"]).mean()
    day = H.groupby(["seed", "fold"])[cols].mean()
    out = []
    for c in cols:
        d, m = c.rsplit("_", 1)
        f = F[(F.detector == d) & (F.method == m)]
        out.append({"ds": ds, "tag": tag.lstrip("_"), "detector": d, "method": m, "seeds": H.seed.nunique(),
                    "host_tpr": host[c].groupby("seed").mean().mean(), "day_tpr": day[c].groupby("seed").mean().mean(),
                    "fpr": f.fpr.mean(), "fpr_max_fold": f.fpr.max()})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame([r for ds in discover.DATASETS for t in TAGS for r in (measure(ds, t) or [])])
    R.to_csv(OUT / "measures.csv", index=False)
    base = R[R.tag == "dilate1"].set_index(["ds", "detector", "method"]).host_tpr
    with np.errstate(invalid="ignore", divide="ignore"):
        R["ratio"] = R.host_tpr.to_numpy() / base.reindex(pd.MultiIndex.from_frame(R[["ds", "detector", "method"]])).to_numpy()
    L = ["# Adaptive-attacker characterization (PREREG §9 ADV)\n",
         "Host TPR = share of attacker hosts with >= 1 alarmed host-day (macro over held-out families, mean over seeds). "
         "Ratio = host TPR / the same chain's dilate:1 host TPR. Robust = ratio >= 0.5 on >= 2 of 3 datasets.\n"]
    verdict = []
    for (d, m), g in R.groupby(["detector", "method"]):
        L += [f"## {d} {m}\n", "| perturbation | " + " | ".join(f"{ds} host TPR (ratio) / day TPR / FPR"
                                                                    for ds in discover.DATASETS) + " | robust |",
              "|---|" + "---|" * (len(discover.DATASETS) + 1)]
        for t in [x.lstrip("_") for x in TAGS]:
            gt = g[g.tag == t].set_index("ds")
            if gt.empty:
                continue
            cells, support = [], 0
            for ds in discover.DATASETS:
                if ds not in gt.index:
                    cells.append("—")
                    continue
                r = gt.loc[ds]
                support += bool(np.isfinite(r.ratio) and base.get((ds, d, m), 0) > 0 and r.ratio >= 0.5)
                cells.append(f"{r.host_tpr:.2f} ({r.ratio:.2f}) / {r.day_tpr:.2f} / {100 * r.fpr:.1f}% [{r.seeds}s]")
            if t == "dilate1":
                L.append(f"| {t} (baseline) | " + " | ".join(cells) + " | — |")
                continue
            ok = support >= 2
            verdict.append({"detector": d, "method": m, "tag": t, "datasets": len(gt), "support": support, "robust": ok})
            L.append(f"| {t} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} ({support}/3) |")
        L.append("")
    pd.DataFrame(verdict).to_csv(OUT / "verdict.csv", index=False)
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
