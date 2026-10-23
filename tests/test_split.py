import numpy as np
import pandas as pd
import pytest

from nids import split


def _frame(n=3000, n_groups=60, seed=0):
    rng = np.random.default_rng(seed)
    fam = rng.choice(["Benign", "Flood", "Recon", "Web"], size=n, p=[0.5, 0.3, 0.15, 0.05])
    return pd.DataFrame({"a": rng.normal(size=n), "y": fam, "family": fam,
                         "group": rng.integers(0, n_groups, size=n)})


def test_grouped_split_is_a_disjoint_partition_without_shared_groups():
    df = _frame()
    tr, va, te = split.grouped_split(df, seed=0)
    assert len(set(tr) | set(va) | set(te)) == len(df)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    g = df["group"].to_numpy()
    assert not (set(g[tr]) & set(g[te]) or set(g[tr]) & set(g[va]) or set(g[va]) & set(g[te]))
    assert 0.5 < len(tr) / len(df) < 0.7 and 0.1 < len(te) / len(df) < 0.3


def test_grouped_split_depends_on_seed():
    df = _frame()
    assert not np.array_equal(split.grouped_split(df, seed=0)[2], split.grouped_split(df, seed=1)[2])


def test_seeds_0_to_4_are_5_fold_cv_over_groups():
    df = _frame()
    tests = [split.grouped_split(df, seed)[2] for seed in range(5)]
    assert sorted(np.concatenate(tests)) == list(range(len(df)))  # disjoint and covering
    assert np.array_equal(split.grouped_split(df, 5)[2], tests[0])


def test_random_split_used_when_no_group_column():
    df = _frame().drop(columns="group")
    tr, va, te = split.grouped_split(df, seed=0)
    assert len(tr) + len(va) + len(te) == len(df)
    # stratified: Web share preserved in test within 1 point
    web = (df["family"] == "Web").to_numpy()
    assert abs(web[te].mean() - web.mean()) < 0.01


@pytest.mark.parametrize("held", ["Flood", "Recon", "Web"])
def test_lofo_removes_family_from_train_and_val_only(held):
    df = _frame()
    tr, va, te = split.grouped_split(df, seed=0)
    ltr, lva, lte = split.lofo(df, (tr, va, te), held)
    fam = df["family"].to_numpy()
    assert (fam[ltr] != held).all() and (fam[lva] != held).all()
    assert (fam[lte] == held).any()
    assert set(lte) == set(te)


def test_lofo_rejects_benign():
    df = _frame()
    with pytest.raises(ValueError):
        split.lofo(df, split.grouped_split(df, seed=0), "Benign")


def test_paper_split_is_80_20_random():
    df = _frame()
    tr, te = split.paper_split(df, seed=0)
    assert len(tr) + len(te) == len(df) and abs(len(te) / len(df) - 0.2) < 0.01
