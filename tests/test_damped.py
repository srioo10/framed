"""Brute-force soundness of the damped deletion box: random clean streams plus adversarial in-budget injections
(placed at window edges, where they weigh most); the clean statistics must lie in the box of the observed ones."""
import numpy as np
from nids.damped import box, stats


def inject_times(rng, m, win, horizon):
    """ages of an in-budget injection pattern: groups of <= m flows spanning <= 1 ms, each group starting at least
    win + 0.01 s after the previous one (so no `win` s interval holds two groups); mostly packed tight (heaviest)."""
    ages, base = [], 1e-6
    while base < horizon:
        n = rng.integers(0, m + 1)
        ages += list(base + np.sort(rng.random(n)) * 1e-3)
        base += win + 0.01 + (0.0 if rng.random() < 0.7 else rng.random() * win)
    return np.array(ages)


def in_budget(ages, m, win):
    a = np.sort(ages)
    return all(np.searchsorted(a, x + win, side="left") - i <= m for i, x in enumerate(a))


def test_box_contains_clean():
    rng = np.random.default_rng(0)
    for trial in range(4000):
        lam = float(rng.choice([0.1, 0.01, 0.001, 0.0005]))
        m, win, G = int(rng.integers(1, 12)), 240.0, float(rng.choice([1.0, 1500.0]))
        horizon = 3000.0
        ca = rng.random(int(rng.integers(0, 60))) * horizon
        cg = rng.random(len(ca)) * G * (rng.random() < 0.5 or 0.1)
        ia = inject_times(rng, m, win, horizon)
        assert in_budget(ia, m, win)
        ig = np.where(rng.random(len(ia)) < 0.5, G, 0.0) if len(ia) else np.array([])
        obs = stats(np.r_[ca, ia], np.r_[cg, ig], lam)
        cln = stats(ca, cg, lam)
        if obs[0] == 0:
            continue
        (wl, wh), (ml, mh), (sl, sh) = box(*obs, lam, m, G, win)
        tol = 1e-9 * max(1.0, G * G)
        assert wl - 1e-9 <= cln[0] <= wh + 1e-9, (trial, cln, obs)
        if cln[0] > 0:
            assert ml - tol <= cln[1] <= mh + tol, (trial, cln, obs, (ml, mh))
            assert sl - 1e-6 * max(1, G) <= cln[2] <= sh + 1e-6 * max(1, G), (trial, cln, obs, (sl, sh))


def test_weight_bound_nearly_tight():
    """m flows just after every window edge reach the weight bound up to the epsilon offsets."""
    lam, m, win = 0.005, 10, 240.0
    ia = np.concatenate([k * (win + 1e-4) + np.arange(m) * 1e-9 for k in range(200)])
    w = np.exp(-lam * ia).sum()
    S = 1 / (1 - np.exp(-lam * win))
    assert in_budget(ia, m, win) and w <= m * S and (m * S - w) / (m * S) < 1e-4
