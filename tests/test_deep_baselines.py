import numpy as np

from nids import deep_baselines as db


def test_deep_sad_and_cade_rank_an_unseen_cluster_higher():
    rng = np.random.default_rng(0)
    db.EPOCHS = 5
    ben = rng.normal(0, 1, (3000, 6)).astype(np.float32)
    att = rng.normal(0, 1, (1000, 6)).astype(np.float32) + np.float32([3, 0, 0, 0, 0, 0])
    new = rng.normal(0, 1, (500, 6)).astype(np.float32) + np.float32([0, 0, 0, 0, 0, 6])
    X = np.vstack([ben, att])
    y = np.r_[np.zeros(3000, np.int64), np.ones(1000, np.int64)]
    tr, es = np.arange(4000) % 10 != 0, np.arange(4000) % 10 == 0
    sad = db.deep_sad(X[tr], y[tr] == 1, X[es], y[es] == 1, seed=0)
    cad = db.cade(X[tr], y[tr], X[es], y[es], 2, seed=0)
    for score in (sad, cad):
        s_new, s_ben = score(new), score(ben[:500])
        assert np.isfinite(s_new).all() and np.median(s_new) > np.median(s_ben)
