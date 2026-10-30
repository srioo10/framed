"""PREREG §9 CERT K3: the tree box bound covers the exact score everywhere in the box and is exact on a point."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cert_trees  # noqa: E402


@pytest.mark.parametrize("k,bi", [(3, 0), (3, 2), (2, 0), (2, 1)])
def test_tree_bounds_sound(k, bi):
    rng = np.random.default_rng(k + bi)
    X = rng.normal(size=(800, 5)).astype(np.float32)
    y = np.clip((X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5), 0, k - 1)
    lo, hi = X[:60] - rng.uniform(0, 0.4, (60, 5)), X[:60] + rng.uniform(0, 0.4, (60, 5))
    hi[:, 4] = np.inf
    for m, up in ((RandomForestClassifier(30, min_samples_leaf=2, random_state=0).fit(X, y),
                   lambda m: cert_trees.rf_upper(m, lo, hi, bi)),
                  (XGBClassifier(n_estimators=20, max_depth=4).fit(X, y),
                   lambda m: cert_trees.xgb_upper(m, lo, hi, bi, k))):
        u = up(m)
        exact = lambda P: -np.log(np.maximum(m.predict_proba(P.astype(np.float32))[:, bi], 1e-45))  # noqa: E731
        for _ in range(30):
            P = lo + rng.uniform(size=lo.shape) * (np.minimum(hi, 5) - lo)
            assert (exact(P) <= u + 1e-4).all()
        pt = (cert_trees.rf_upper(m, X[:60], X[:60], bi) if isinstance(m, RandomForestClassifier)
              else cert_trees.xgb_upper(m, X[:60], X[:60], bi, k))
        assert np.allclose(pt, exact(X[:60]), atol=1e-3)
