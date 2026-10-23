import numpy as np
import pytest

from nids import fusion_checks as fc


def test_tail_pvalues_match_hand_counts():
    ref = np.array([[1.0], [2.0], [2.0], [3.0]])
    x = np.array([[0.0], [2.0], [2.5], [9.0]])
    # #{ref >= x} = 4, 3, 1, 0  ->  (1 + ge) / 5
    assert np.allclose(fc.tail_pvalues(ref, x)[:, 0], [1.0, 0.8, 0.4, 0.2])


def test_ties_keep_cal_fpr_at_or_below_target():
    rng = np.random.default_rng(0)
    for n in (1000, 5000, 20000):
        for s in (np.round(rng.random(n), 1), rng.integers(0, 3, n).astype(float), rng.random(n)):
            tau = fc.threshold(s, 0.01)
            assert (s > tau).mean() <= 0.01


def test_threshold_needs_support():
    with pytest.raises(ValueError):
        fc.threshold(np.zeros(999), 0.01)
    fc.threshold(np.zeros(1000), 0.01)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_scores_raise(bad):
    ref = np.zeros((10, 1))
    with pytest.raises(ValueError):
        fc.tail_pvalues(ref, np.array([[bad]]))
    with pytest.raises(ValueError):
        fc.tail_pvalues(np.vstack([ref, [[bad]]]), np.zeros((1, 1)))
    with pytest.raises(ValueError):
        fc.threshold(np.r_[np.zeros(2000), bad], 0.01)


def test_zero_share_removes_member():
    p = np.array([[0.001, 0.5], [0.5, 0.5]])
    assert np.allclose(fc.budget_fusion_score(p, [0.0, 1.0]), [-0.5, -0.5])
    assert fc.budget_fusion_score(p, [0.5, 0.5])[0] > fc.budget_fusion_score(p, [0.0, 1.0])[0]  # member 0 counts again
    names = ["rf", "xgb", "mlp_ce", "iforest", "clust_global"]
    assert np.allclose(fc.shares_from_alpha(1.0, names), [1 / 3, 1 / 3, 1 / 3, 0, 0])


def test_mode_pvalues_fall_back_to_pooled_for_sparse_modes():
    ref = np.r_[np.arange(10.0), np.arange(100.0, 103.0)]
    ref_mode = np.r_[np.zeros(10, int), np.ones(3, int)]
    x, x_mode = np.array([5.0, 101.0]), np.array([0, 1])
    p = fc.tail_pvalues_by_mode(ref, ref_mode, x, x_mode, n_min=5)
    assert np.isclose(p[0], (1 + 5) / 11)        # mode 0: own 10-row reference
    assert np.isclose(p[1], (1 + 2) / 14)        # mode 1 has 3 < 5 rows: pooled 13-row reference


def test_tie_break_tolerance_and_flat_flag():
    grid = np.array([0.3, 0.6, 0.9, 1.0])
    # a curve inside 0.005 everywhere is flat; harmonic mean decides first ...
    flat = np.array([[0.50, 0.40], [0.501, 0.401], [0.50, 0.40], [0.499, 0.399]])
    a, info = fc.pick_alpha(grid, flat)
    assert info["flat"] and a == 0.6
    # ... and the prior (0.9) breaks an exact harmonic-mean tie
    a, info = fc.pick_alpha(grid, np.array([[0.5, 0.4], [0.5, 0.4], [0.5, 0.4], [0.499, 0.399]]))
    assert info["flat"] and a == 0.9
    # a clear winner beyond the paired SE is picked and the curve is not flat
    clear = np.array([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1]])
    a, info = fc.pick_alpha(grid, clear)
    assert a == 0.6 and not info["flat"] and info["tied"].sum() == 1
    # equal macro, different spread: harmonic mean prefers the even alpha
    spread = np.array([[0.9, 0.1], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
    a, _ = fc.pick_alpha(grid, spread)
    assert a == 0.6


def test_paired_se_not_family_spread_sets_tolerance():
    grid = np.array([0.3, 0.9])
    # families differ hugely (1.0 vs 0.1) but alpha 0.9 beats 0.3 by 0.1 on every family: not tied
    pf = np.array([[0.9, 0.0, 0.4], [1.0, 0.1, 0.5]])
    a, info = fc.pick_alpha(grid, pf)
    assert a == 0.9 and not info["flat"]
