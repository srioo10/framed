import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.kitnet import KitNET  # noqa: E402


def test_kitnet_scores_outliers_higher():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5000, 3))
    X = np.column_stack([z, z @ rng.normal(size=(3, 12)) + 0.05 * rng.normal(size=(5000, 12))])
    k = KitNET(0, max_ae=5).fit(X)
    assert len(k.groups) >= 3 and sorted(sum(k.groups, [])) == list(range(15))
    ben = rng.normal(size=(1000, 3))
    ben = np.column_stack([ben, ben @ rng.normal(size=(3, 12))])
    bad = rng.normal(size=(1000, 15)) * 3
    assert np.median(k.score(bad)) > np.quantile(k.score(X), 0.95)
