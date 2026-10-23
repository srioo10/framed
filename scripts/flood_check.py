"""Extrapolation check for an outer held-out family (PREREG §9 extra hold-outs row; meant for CICIoT Flood).

1. On the 10 features where the held-out test rows differ most from the training rows (KS statistic), the share
   of held-out rows outside the training range [min, max].
2. Held-out rows binned by how far they sit outside that range (max over those features, in training-std units):
   mean score and TPR@1% (benign_cal threshold) per supervised member per bin. Trees saturating while the MLP keeps
   rising means the family is easy for the MLP only because of extrapolation.
Features are the cleaned columns (original ct_* on unsw_raw). Usage:
  .venv/bin/python scripts/flood_check.py RUN_DIR [--heldout Flood]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import clean, fusion_checks as fc, load  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import score_bank  # noqa: E402

MEMBERS = ["rf", "xgb", "xgb_tail", "mlp_ce", "mlp_kd"]
TOP = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--heldout", default="Flood")
    a = ap.parse_args()
    ds = a.run.name.split("_ct_")[0].split("_none_")[0]
    part = pd.read_parquet(a.run / f"config=outer_{a.heldout}" / "part.parquet")
    ids = {}
    for z in a.run.glob("ids_*.npz"):
        ids.update(dict(np.load(z)))
    held = set(a.heldout.split("__"))
    df, _ = clean.clean(load.LOADERS[ds](), ds)
    fam = df["family"].astype(str).to_numpy()
    fine = score_bank.fine_family(df, ds)
    feats = [c for c in clean.feature_columns(df) if pd.api.types.is_numeric_dtype(df[c])]
    train = np.concatenate([v for k, v in ids.items() if k.startswith("cap_")])
    train = train[~(np.isin(fam[train], list(held)) | np.isin(fine[train], list(held)))]
    unk = part[part.role == "test_unknown"].reset_index(drop=True)
    Xtr = df.iloc[train][feats].to_numpy(np.float64)
    Xu = df.iloc[unk.row_id.to_numpy()][feats].to_numpy(np.float64)
    del df

    ks = np.array([stats.ks_2samp(Xtr[:, i], Xu[:, i]).statistic for i in range(len(feats))])
    top = np.argsort(-ks)[:TOP]
    lo, hi, sd = Xtr.min(0), Xtr.max(0), Xtr.std(0) + 1e-9
    out = (Xu < lo) | (Xu > hi)
    print(f"{ds} held-out {a.heldout}: {len(unk)} test rows, {len(train)} training rows\n")
    print("| feature | KS | share outside train range |\n|---|---|---|")
    for i in top:
        print(f"| {feats[i]} | {ks[i]:.3f} | {out[:, i].mean():.3%} |")
    print(f"\nany of top {TOP} outside range: {out[:, top].any(1).mean():.3%}; any feature: {out.any(1).mean():.3%}\n")

    beyond = np.maximum(lo - Xu, Xu - hi)[:, top] / sd[top]
    dist = np.clip(beyond, 0, None).max(1)
    bins = [-1, 0, 0.5, 2, 10, np.inf]
    lab = pd.cut(dist, bins, labels=["inside", "0-0.5sd", "0.5-2sd", "2-10sd", ">10sd"])
    cal = part[part.role == "cal"]
    tau = {m: fc.threshold(cal[m].to_numpy(np.float64), 0.01) for m in MEMBERS if m in part}
    print("| bin | n | " + " | ".join(f"{m} mean / TPR@1%" for m in tau) + " |")
    print("|---|---|" + "---|" * len(tau))
    for b in lab.categories:
        q = np.asarray(lab == b)
        if not q.any():
            continue
        cells = [f"{unk[m][q].mean():.3f} / {(unk[m][q] > t).mean():.3f}" for m, t in tau.items()]
        print(f"| {b} | {q.sum()} | " + " | ".join(cells) + " |")
    print("\nscore at the member's maximum (saturation): " +
          ", ".join(f"{m} {(unk[m] >= unk[m].max()).mean():.1%}" for m in tau))


if __name__ == "__main__":
    main()
