"""Splits (PREREG §4, §6). All functions return positional index arrays into the cleaned frame."""
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def grouped_split(df, seed):
    """60/20/20 train/val/test. StratifiedGroupKFold on `group` when present, else stratified random."""
    y = pd.factorize(df["y"], sort=True)[0]  # sorted int codes: same classes as the strings, no 46M-string sort
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


def forward_split(df, seed, train=("june_week3", "june_week4"), attack=("july_week5",),
                  test=("august_week1", "august_week2")):
    """PREREG §9 FWD: benign of `train` weeks and attacks of `attack` weeks form train + val (grouped 75/25 as in
    grouped_split); test = every row of `test` weeks. Other rows (July benign) are unused."""
    wk, ben = df["src"].astype(str).to_numpy(), df["family"].astype(str).to_numpy() == "Benign"
    rest = np.flatnonzero((np.isin(wk, train) & ben) | (np.isin(wk, attack) & ~ben))
    te = np.flatnonzero(np.isin(wk, test))
    y = pd.factorize(df["y"], sort=True)[0]
    g = df["group"].to_numpy()
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
