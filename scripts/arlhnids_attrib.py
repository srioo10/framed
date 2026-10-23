"""H0 attribution (PREREG §4): start from the paper's order and switch leak steps to leak-free one at a time.

  a  preprocessing fit on train+test       -> on train only
  b  oversample, then split                -> split, then oversample train only (test = real rows)
  c  early stopping on test                -> on a validation split carved from train
  d  duplicates and label conflicts kept   -> removed before the split
  f  BoT-IoT label columns (attack, subcategory) as features -> dropped
  g  other leak columns as features (BoT-IoT ids, UNSW TTL fields, CICIoT Number) -> dropped

Real rows are always split 80/20 at random first. With b off, balancing runs on all real rows and each
synthetic row goes to train or test at random with the same 80/20 odds; this is the same distribution as the
paper's balance-then-split, and lets a be switched independently of b. Step e (DoS vs Flood held out) concerns
the anomaly detector and is measured there.

Usage: .venv/bin/python scripts/arlhnids_attrib.py DATASET --fixed abcdfg [--seed 0] [--balance gamo|smote]
       (--fixed "" = paper order; --fixed abcdfg = every step leak-free)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from arlhnids_sids import balance, nn_distance, report  # noqa: E402
from nids import clean, load  # noqa: E402
from nids.arlhnids import gwo, paper, sids  # noqa: E402
from nids.arlhnids.prep import Prep  # noqa: E402

STEPS = "abcdfg"
LABEL_COLS = {"bot_iot": ["attack", "subcategory"]}


def dedup(df, feats):
    """Step d: drop rows whose features carry two labels, then exact duplicates (as clean.clean, on `feats`)."""
    h = clean.row_hash(df, feats)
    y = df["y"].astype(str).to_numpy()
    pairs = pd.DataFrame({"h": h, "y": y}).drop_duplicates()
    conflict = np.isin(h, pairs["h"][pairs["h"].duplicated()].to_numpy())
    dup = np.zeros(len(df), bool)
    dup[~conflict] = pd.Series(h[~conflict]).duplicated().to_numpy()
    return df.loc[~conflict & ~dup].reset_index(drop=True), {"conflict_rows": int(conflict.sum()),
                                                             "duplicates": int(dup.sum())}


def run(ds, fixed, seed, method):
    t0 = time.time()
    raw = load.LOADERS[ds]()
    orig = raw["y"].astype(str).value_counts()
    classes = sorted(orig.index)
    ratio_by_name = {c: paper.BALANCED[ds][c] / orig[c] for c in orig.index}
    info = {"dataset": ds, "fixed": "".join(sorted(fixed)), "seed": seed, "balance": method}

    drop = set(clean.LABELS) | set(paper.PAPER_DROP[ds])
    if "f" in fixed:
        drop |= set(LABEL_COLS.get(ds, []))
    if "g" in fixed:
        drop |= set(clean.LEAK_COLS[ds]) - set(LABEL_COLS.get(ds, []))
    feats = [c for c in raw.columns if c not in drop]
    if "d" in fixed:
        raw, info["dedup"] = dedup(raw, feats)
    y = pd.Categorical(raw["y"].astype(str), categories=classes).codes.astype(np.int64)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    va = None
    if "c" in fixed:
        tr, va = train_test_split(tr, test_size=0.25, random_state=seed)
    prep = Prep().fit(raw.iloc[np.sort(tr)][feats] if "a" in fixed else raw[feats])
    X = prep.transform(raw[feats])
    del raw
    ratio = {i: ratio_by_name[c] for i, c in enumerate(classes)}

    if "b" in fixed:  # balance train only; val and test stay real
        Xtr, ytr, _, info["balance_info"], _ = balance(X[tr], y[tr], ratio, paper.FCM[ds], method, seed)
        Xte, yte, synth_te = X[te], y[te], np.zeros(len(te), bool)
    else:  # balance every non-val row, then send rows to train/test with the 80/20 split's odds
        pool = np.sort(np.concatenate([tr, te]))
        Xb, yb, synth, info["balance_info"], kept = balance(X[pool], y[pool], ratio, paper.FCM[ds], method, seed)
        to_test = np.empty(len(yb), bool)
        to_test[:len(kept)] = np.isin(pool[kept], te)
        rng = np.random.default_rng(seed)
        to_test[len(kept):] = rng.random(len(yb) - len(kept)) < len(te) / len(pool)
        Xtr, ytr, Xte, yte, synth_te = Xb[~to_test], yb[~to_test], Xb[to_test], yb[to_test], synth[to_test]
    info["test_synthetic_share"] = float(synth_te.mean())
    info["rows"] = {"train": int(len(ytr)), "test": int(len(yte))}
    t1 = time.time()

    g = paper.GWO[ds]
    mask = gwo.select(Xtr, ytr, g["pop"], g["iters"], seed=paper.GWO_SEED + seed, alpha=paper.GWO_ALPHA,
                      max_corr=paper.GWO_MAX_CORR)
    info["gwo_features"] = [f for f, m in zip(feats, mask) if m]
    Xtr, Xte = Xtr[:, mask], Xte[:, mask]
    Xes, yes = (X[va][:, mask], y[va]) if va is not None else (Xte, yte)
    t2 = time.time()
    vote, preds, _ = sids.fit_predict(sids.build(paper.SIDS[ds], seed), Xtr, ytr, Xte, Xes, yes, len(classes))
    info["sids"] = report(yte, vote, classes)
    info["per_model_accuracy"] = {m: float((p == yte).mean()) for m, p in preds.items()}
    info["accuracy_real_test_rows"] = float((vote == yte)[~synth_te].mean())
    info["test_nn_distance"] = nn_distance(Xtr, Xte, synth_te, seed)
    info["seconds"] = {"data": t1 - t0, "gwo": t2 - t1, "sids": time.time() - t2}
    info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip())
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=paper.DATASETS)
    ap.add_argument("--fixed", default="", help=f"subset of {STEPS} switched to leak-free")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", choices=["gamo", "smote"], default="gamo")
    a = ap.parse_args()
    assert set(a.fixed) <= set(STEPS), a.fixed
    res = run(a.dataset, set(a.fixed), a.seed, a.balance)
    tag = res["fixed"] or "none"
    out = ROOT / "results" / "attrib" / f"{a.dataset}_{a.balance}_fixed-{tag}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    s = res["sids"]
    print(f"{a.dataset} fixed={tag}: acc {s['accuracy']:.4f} real-row acc {res['accuracy_real_test_rows']:.4f} "
          f"macroF1 {s['macro']['f']:.3f} -> {out}")
