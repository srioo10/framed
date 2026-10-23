"""Leak guards on the real cleaned data (PLAN §7). Skipped when a dataset is not downloaded.
ciciot_csv needs ~20 GB RAM, so it runs only with NIDS_SLOW=1."""
import os

import numpy as np
import pytest

from nids import clean, load, split

DATASETS = ["unsw_official", "ciciot_paper", "bot_iot", "unsw_raw",
            pytest.param("ciciot_csv", marks=pytest.mark.skipif(not os.environ.get("NIDS_SLOW"),
                                                                 reason="set NIDS_SLOW=1 (~20 GB RAM)"))]


@pytest.fixture(scope="module", params=DATASETS)
def cleaned(request):
    try:
        df = load.LOADERS[request.param]()
    except FileNotFoundError:
        pytest.skip(f"{request.param} not downloaded")
    out, _ = clean.clean(df, request.param)
    return request.param, out


def test_no_feature_row_shared_between_splits(cleaned):
    name, df = cleaned
    h = clean.row_hash(df, clean.feature_columns(df))
    for seed in range(5):
        tr, va, te = split.grouped_split(df, seed)
        assert not np.isin(h[te], h[tr]).any() and not np.isin(h[va], h[tr]).any(), (name, seed)
        assert not np.isin(h[te], h[va]).any(), (name, seed)


def test_groups_never_cross_splits(cleaned):
    name, df = cleaned
    if "group" not in df:
        pytest.skip(f"{name} has no group id")
    g = df["group"].to_numpy()
    for seed in range(5):
        tr, va, te = (set(np.unique(g[i])) for i in split.grouped_split(df, seed))
        assert not (tr & va or tr & te or va & te), (name, seed)


def test_every_family_reaches_test(cleaned):
    name, df = cleaned
    fam = df["family"].astype(str).to_numpy()
    for seed in range(5):
        te = split.grouped_split(df, seed)[2]
        missing = set(fam) - set(np.unique(fam[te]))
        assert not missing - {"Theft"}, (name, seed, missing)  # BoT-IoT Theft has 75 rows (PREREG §6 low-n)


def test_no_leak_column_survives(cleaned):
    name, df = cleaned
    assert not set(clean.LEAK_COLS[name]) & set(df.columns)


def test_lofo_never_trains_on_held_out_family(cleaned):
    name, df = cleaned
    fam = df["family"].astype(str).to_numpy()
    splits = split.grouped_split(df, 0)
    for held in sorted(set(fam) - {"Benign"}):
        tr, va, te = split.lofo(df, splits, held)
        assert (fam[tr] != held).all() and (fam[va] != held).all() and (fam[te] == held).any(), (name, held)


def test_every_split_has_benign(cleaned):
    _, df = cleaned
    fam = df["family"].astype(str).to_numpy()
    for seed in range(5):
        assert all((fam[i] == "Benign").any() for i in split.grouped_split(df, seed))
