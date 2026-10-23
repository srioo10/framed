import numpy as np
import pandas as pd
import pytest

from nids import load
from nids.arlhnids import fcm, gamo, gwo, paper, sids
from nids.arlhnids.prep import Prep, majority_classes


@pytest.mark.parametrize("ds", paper.DATASETS)
def test_eq2_reproduces_table5_and_table10_covers_every_class(ds):
    try:
        counts = load.LOADERS[ds]()["y"].astype(str).value_counts().to_dict()
    except FileNotFoundError:
        pytest.skip(f"{ds} not downloaded")
    assert majority_classes(counts) == paper.MAJORITY[ds]
    assert set(paper.BALANCED[ds]) == set(counts)
    for c in paper.MAJORITY[ds]:  # Table 9: majority classes only shrink
        assert paper.BALANCED[ds][c] <= counts[c]


def test_prep_clamps_only_skewed_columns_and_uses_fit_statistics():
    df = pd.DataFrame({"skew": [1.0] * 99 + [1000.0], "flat": np.arange(100.0), "cat": ["a", "b"] * 50})
    p = Prep().fit(df)
    assert np.isfinite(p.cap[0]) and np.isinf(p.cap[1])
    X = p.transform(pd.DataFrame({"skew": [5000.0], "flat": [np.nan], "cat": ["zzz"]}))
    assert X[0, 0] == pytest.approx((p.cap[0] - p.mu[0]) / p.sd[0])  # clamped to train q95
    assert X[0, 1] == pytest.approx(0.0, abs=1e-6) and X[0, 2] == pytest.approx(0.0, abs=1e-6)  # -> train mean


def test_fcm_undersample_hits_target_and_covers_clusters():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(c * 10, 1, size=(n, 2)) for c, n in [(0, 900), (1, 90), (2, 10)]])
    idx = fcm.undersample(X, 100, dict(c=3, m=1.5, error=1e-4, maxiter=200), rng)
    assert len(idx) == 100 == len(set(idx))
    blob = np.rint(X[idx, 0] / 10).astype(int)  # ~90 / 9 / 1 from the three blobs
    assert set(np.unique(blob)) == {0, 1, 2} and abs((blob == 0).sum() - 90) <= 1


def test_gwo_prefers_the_informative_feature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    X = np.c_[y + rng.normal(0, 0.1, 600), rng.normal(size=(600, 5))].astype(np.float32)
    mask = gwo.select(X, y, pop=8, iters=5, seed=0)
    assert mask[0] and mask.sum() <= 3


def test_hard_vote_takes_majority_and_breaks_ties_low():
    class Fixed:
        def __init__(self, p): self.p = np.array(p)
        def fit(self, *a, **k): return self
        def predict(self, X): return self.p
    models = {"a": Fixed([1, 2]), "b": Fixed([1, 0]), "c": Fixed([0, 1])}
    vote, _, _ = sids.fit_predict(models, None, None, np.zeros((2, 1)), None, None, k=3)
    assert list(vote) == [1, 0]


def test_gamo_hits_targets_with_convex_rows_of_the_right_class(monkeypatch):
    monkeypatch.setattr(gamo, "STEPS", 30)
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(c * 5, 1, size=(n, 4)) for c, n in [(0, 400), (1, 60), (2, 20)]]).astype(np.float32)
    y = np.repeat([0, 1, 2], [400, 60, 20])
    Xo, yo, info = gamo.oversample(X, y, {1: 400, 2: 300}, seed=0)
    assert np.array_equal(Xo[:len(y)], X) and np.array_equal(yo[:len(y)], y)  # real rows first, untouched
    assert np.bincount(yo).tolist() == [400, 400, 300]
    for c in (1, 2):  # convex combinations of the class's own rows stay inside its bounding box
        real, fake = X[y == c], Xo[len(y):][yo[len(y):] == c]
        assert (fake >= real.min(0) - 1e-4).all() and (fake <= real.max(0) + 1e-4).all()
    assert info["gamo_basis"] == {"1": 45, "2": 15}  # 75% of each grown class
