"""Splits (PREREG §4, §6). All functions return positional index arrays into the cleaned frame."""
from itertools import islice

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def grouped_split(df, seed):
    """60/20/20 train/val/test. StratifiedGroupKFold on `group` when present, else stratified random."""
    y = df["y"].astype(str).to_numpy()
    idx = np.arange(len(df))
    if "group" not in df.columns:
        rest, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        tr, va = train_test_split(rest, test_size=0.25, random_state=seed, stratify=y[rest])
        return np.sort(tr), np.sort(va), np.sort(te)
    # sklearn's SGKF shuffle only breaks ties, so the seed picks the fold instead:
    # seeds 0-4 put every group in test exactly once (5-fold CV over groups).
    g = df["group"].to_numpy()
    rest, te = next(islice(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(idx, y, g), seed % 5, None))
    tr, va = next(islice(StratifiedGroupKFold(4, shuffle=True, random_state=0).split(rest, y[rest], g[rest]),
                         seed % 4, None))
    return rest[tr], rest[va], te


def lofo(df, splits, held):
    """Leave-one-family-out: drop `held` from train and val; test is unchanged."""
    if held == "Benign":
        raise ValueError("Benign is never held out")
    tr, va, te = splits
    keep = df["family"].to_numpy() != held
    return tr[keep[tr]], va[keep[va]], te


def paper_split(df, seed):
    """ARLHNIDS Algorithm 1 line 11: plain random 80/20 (no stratify, no groups)."""
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed)
    return np.sort(tr), np.sort(te)
