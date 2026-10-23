"""ARLHNIDS SIDS reproduction (PLAN §4, PREREG H0), paper order vs leak-free order.

  paper    Algorithm 1: merge official train+test -> preprocess -> FCM undersample + oversample -> random 80/20
           -> GWO on train -> 7-tree vote (Table 13 settings, XGBoost early-stopped on test). Paper's columns.
  leakfree clean() (leak columns, conflicts, duplicates) -> 60/20/20 split -> preprocess, balance and GWO
           on train only -> same 7-tree vote, early-stopped on val -> scored on real test rows.

Oversampling: modified GAMO (paper) or SMOTE (control); class targets keep the paper's Table 10 ratios, so in
paper order they equal Table 10 exactly.

Usage: .venv/bin/python scripts/arlhnids_sids.py DATASET {paper,leakfree} [--seed 0] [--balance smote|gamo|none]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import clean, load, split  # noqa: E402
from nids.arlhnids import fcm, gamo, gwo, paper, sids  # noqa: E402
from nids.arlhnids.prep import Prep  # noqa: E402

BENIGN = "Normal"  # benign label in all three paper datasets


def balance(X, y, ratio, fcm_params, method, seed):
    """Per class: target = round(ratio * count). Below count -> FCM undersample; above -> oversample.
    GAMO sees only the classes that are not undersampled (Table 5 minority).
    Returns (X, y, synthetic mask, info, kept): real rows come first, as input rows `kept` (sorted)."""
    rng = np.random.default_rng(seed)
    keep, over = [], {}
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        target = int(round(ratio[c] * len(idx)))
        if target < len(idx):
            idx = idx[fcm.undersample(X[idx], target, fcm_params, rng)]
        elif target > len(idx):
            over[c] = target
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    X, y = X[keep], y[keep]
    if method == "none" or not over:
        return X, y, np.zeros(len(y), bool), {}, keep
    if method == "gamo":
        minor = np.isin(y, [c for c in np.unique(y) if ratio[c] >= 1])
        Xg, yg, info = gamo.oversample(X[minor], y[minor], over, seed,
                                       log=lambda t, d, m, g: print(f"  gamo step {t} D {d:.3f} M {m:.3f} G {g:.3f}",
                                                                    flush=True))
        n = int(minor.sum())
        X, y = np.concatenate([X, Xg[n:]]), np.concatenate([y, yg[n:]])
        return X, y, np.arange(len(y)) >= len(keep), info, keep
    k = min(5, min(int((y == c).sum()) for c in over) - 1)
    Xr, yr = SMOTE(sampling_strategy=over, k_neighbors=k, random_state=seed).fit_resample(X, y)
    return Xr.astype(np.float32), yr, np.arange(len(yr)) >= len(y), {}, keep  # imblearn appends synthetic rows


def report(y, pred, classes):
    labels = np.arange(len(classes))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    b = classes.index(BENIGN)
    benign = y == b
    out = {"accuracy": float((y == pred).mean()),
           "macro": dict(zip("prf", map(float, (p.mean(), r.mean(), f.mean())))),
           "weighted": dict(zip("prf", map(float, np.average([p, r, f], axis=1, weights=n)))),
           "far_benign": float((pred[benign] != b).mean()),       # benign flagged as any attack
           "far_macro": gwo.far(y, pred, len(classes)),           # paper Eq. 23, one-vs-rest
           "attack_recall": float((pred[~benign] != b).mean()),
           "per_class": {c: {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
                         for i, c in enumerate(classes)}}
    # paper check: F1 must equal 2PR/(P+R) per class (deck Issue 3)
    out["f1_consistent"] = bool(np.allclose(f, np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)))
    return out


def row_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def nn_distance(Xtr, Xte, synth_te, seed, n=5000):
    """Median distance from test rows to their nearest training row (selected features), real vs synthetic test
    rows. Synthetic rows built from rows that land in train sit close to them; real held-out rows should not."""
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xtr)
    out = {}
    for name, idx in (("real", np.flatnonzero(~synth_te)), ("synthetic", np.flatnonzero(synth_te))):
        if len(idx):
            idx = rng.choice(idx, min(n, len(idx)), replace=False)
            d = nn.kneighbors(Xte[idx])[0][:, 0]
            out[name] = {"median": float(np.median(d)), "share_below_0.01": float((d < 0.01).mean()), "n": len(idx)}
    return out


def run(ds, mode, seed, method):
    t0 = time.time()
    raw = load.LOADERS[ds]()
    orig = raw["y"].astype(str).value_counts()
    ratio_by_name = {c: paper.BALANCED[ds][c] / orig[c] for c in orig.index}
    classes = sorted(orig.index)
    info = {"dataset": ds, "mode": mode, "seed": seed, "balance": method}

    if mode == "paper":
        feats = [c for c in raw.columns if c not in clean.LABELS + paper.PAPER_DROP[ds]]
        prep = Prep().fit(raw[feats])
        X, y = prep.transform(raw[feats]), pd.Categorical(raw["y"].astype(str), categories=classes).codes
        del raw
        ratio = {i: ratio_by_name[c] for i, c in enumerate(classes)}
        X, y, synth, info["balance_info"], _ = balance(X, y, ratio, paper.FCM[ds], method, seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)  # Algorithm 1 line 11
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        info["test_synthetic_share"] = float(synth[te].mean())
        synth_te = synth[te]
    else:
        df, rep = clean.clean(raw, ds)
        del raw
        tr, va, te = split.grouped_split(df, seed)
        feats = clean.feature_columns(df)
        prep = Prep().fit(df.iloc[tr][feats])
        y_all = pd.Categorical(df["y"].astype(str), categories=classes).codes
        Xtr, ytr = prep.transform(df.iloc[tr][feats]), y_all[tr]
        Xva, yva = prep.transform(df.iloc[va][feats]), y_all[va]
        Xte, yte = prep.transform(df.iloc[te][feats]), y_all[te]
        del df
        ratio = {i: ratio_by_name[c] for i, c in enumerate(classes)}
        Xtr, ytr, synth_tr, info["balance_info"], _ = balance(Xtr, ytr, ratio, paper.FCM[ds], method, seed)
        info["cleaning"] = {k: v for k, v in rep.items() if k not in ("per_class", "kept_rows")}
        info["train_synthetic_share"] = float(synth_tr.mean())
        synth_te = np.zeros(len(yte), bool)
    info["features"] = feats
    info["rows"] = {"train": int(len(ytr)), "test": int(len(yte))}
    info["train_class_counts"] = {classes[i]: int(n) for i, n in zip(*np.unique(ytr, return_counts=True))}
    t1 = time.time()

    g = paper.GWO[ds]
    mask = gwo.select(Xtr, ytr, g["pop"], g["iters"], seed=paper.GWO_SEED + seed, alpha=paper.GWO_ALPHA,
                      max_corr=paper.GWO_MAX_CORR, log=lambda t, f, k: print(f"  gwo it {t} best {f:.4f} k={k}",
                                                                              flush=True))
    info["gwo_features"] = [f for f, m in zip(feats, mask) if m]
    info["gwo_kept"] = int(mask.sum())
    info["paper_gwo_kept"] = paper.GWO_KEPT[ds]
    t2 = time.time()

    Xtr, Xte = Xtr[:, mask], Xte[:, mask]
    if mode == "paper":
        Xes, yes = Xte, yte  # early stopping on test, as in the paper
        info["test_rows_in_train"] = float(np.isin(row_hash(Xte), row_hash(Xtr)).mean())
    else:
        Xes, yes = Xva[:, mask], yva
    info["test_nn_distance"] = nn_distance(Xtr, Xte, synth_te, seed)
    vote, preds, secs = sids.fit_predict(sids.build(paper.SIDS[ds], seed), Xtr, ytr, Xte, Xes, yes, len(classes))
    info["sids"] = report(yte, vote, classes)
    info["per_model_accuracy"] = {m: float((p == yte).mean()) for m, p in preds.items()}
    if synth_te.any():  # paper order: how much of the score comes from synthetic test rows
        info["accuracy_real_test_rows"] = float((vote == yte)[~synth_te].mean())
        info["accuracy_synthetic_test_rows"] = float((vote == yte)[synth_te].mean())
    info["fit_seconds"] = secs
    info["paper_sids_accuracy"] = paper.SIDS_ACCURACY[ds]
    info["seconds"] = {"data": t1 - t0, "gwo": t2 - t1, "sids": time.time() - t2}
    info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip()
    info["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip())
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=paper.DATASETS)
    ap.add_argument("mode", choices=["paper", "leakfree"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", choices=["smote", "gamo", "none"], default="smote")
    a = ap.parse_args()
    res = run(a.dataset, a.mode, a.seed, a.balance)
    out = ROOT / "results" / "arlhnids" / f"sids_{a.dataset}_{a.mode}_{a.balance}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    s = res["sids"]
    print(f"{a.dataset} {a.mode}: acc {s['accuracy']:.4f} (paper {res['paper_sids_accuracy']}) "
          f"macroF1 {s['macro']['f']:.4f} FAR {s['far_benign']:.4f} -> {out}")
