"""Single-feature shortcut audit (PREREG §9 fingerprint row): for each feature alone, a depth-2 tree separating
all attacks (pooled) from benign; TPR at the 1%-FPR benign quantile of its score. Run on the cleaned frame with the
dataset's current leak columns removed. In-sample by design: it measures how separable one feature is, not a model.
Usage: .venv/bin/python scripts/feature_audit.py nf_unsw [--top 12]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import clean, load  # noqa: E402


def audit(df, n_benign=200_000, seed=0):
    y = (df["family"].astype(str) != "Benign").to_numpy()
    rng = np.random.default_rng(seed)
    b = np.flatnonzero(~y)
    i = np.r_[rng.choice(b, min(n_benign, len(b)), replace=False), np.flatnonzero(y)]
    out = []
    for c in clean.feature_columns(df):
        x = df[c].to_numpy("float64", na_value=0.0)[i].reshape(-1, 1)
        p = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(x, y[i]).predict_proba(x)[:, 1]
        th = np.quantile(p[~y[i]], 0.99)
        out.append((c, float((p[y[i]] > th).mean())))
    return sorted(out, key=lambda r: -r[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    df, _ = clean.clean(load.LOADERS[a.ds](), a.ds)
    for c, t in audit(df)[:a.top]:
        print(f"{c:32s} {t:.3f}")
