import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leaf_diagnostic", ROOT / "scripts" / "leaf_diagnostic.py")
ld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ld)


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    return X, (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1)


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_leaf_bounds_contain_their_rows(kind):
    X, y = _data()
    if kind == "rf":
        m = RandomForestClassifier(5, random_state=0, min_samples_leaf=2).fit(X, y)
        trees = ld.rf_trees(m, {"x": X}, [0, 3])
    else:
        m = XGBClassifier(n_estimators=4, max_depth=3, tree_method="hist").fit(X, y)
        trees = ld.xgb_trees(m, {"x": X}, [0, 5], X.shape[1])
    for leaves, lo, hi in trees:
        leaf = leaves["x"]
        right = (X <= hi[leaf]) if kind == "rf" else (X < hi[leaf])
        left = (X > lo[leaf]) if kind == "rf" else (X >= lo[leaf] - 1e-6)  # XGB dumps splits rounded (~1e-8)
        assert right.all() and left.all()
        assert np.isfinite(lo[leaf]).any() or np.isfinite(hi[leaf]).any()  # every leaf has a path


def test_seg_quantiles_and_aggregate():
    leaf = np.array([0, 0, 0, 1, 1])
    X = np.array([[1.0], [3.0], [2.0], [10.0], [20.0]])
    q = ld.seg_quantiles(leaf, X, (0.0, 0.5, 1.0))
    assert q[:, 0, 0].tolist() == [1, 2, 3] and q[:, 1, 0].tolist() == [10, 10, 20]
    z = np.array([[1.0, 5.0, 2.0, 4.0]])
    a = ld.aggregate(z, np.array([[True, True, False, True]]))
    assert a["max"][0] == 5 and np.isclose(a["top3"][0], 10 / 3) and np.isclose(a["rss"][0], np.sqrt(42))
    assert np.isnan(ld.aggregate(z, np.zeros((1, 4), bool))["max"][0])


LEAF = os.environ.get("NIDS_LEAF_CHECK")


@pytest.mark.skipif(not LEAF, reason="set NIDS_LEAF_CHECK to a bank run dir")
def test_leafstat_rows_never_rank_or_cal():
    bank = Path(LEAF)
    ids = {}
    for z in bank.glob("ids_*.npz"):
        ids.update(dict(np.load(z)))
    seed = bank.name.rsplit("_s", 1)[1]
    rows = np.load(ld.LEAFDIR / f"ciciot_s{seed}_leafstat_rows.npz")["bsample"]
    assert not np.intersect1d(rows, np.union1d(ids["rank"], ids["cal"])).size
