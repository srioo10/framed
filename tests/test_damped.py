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


def novelty_flags(times, ports, look=120.0):
    """flag = 1 if the port was not seen in the `look` s before the flow (all flows in `times` count)."""
    order = np.argsort(times, kind="stable"); last, out = {}, np.zeros(len(times))
    for i in order:
        p, t = ports[i], times[i]
        out[i] = 1.0 if p not in last or t - last[p] > look else 0.0
        last[p] = t
    return out


def test_novelty_box_contains_clean():
    """injections pre-touch the ports of later real flows (the worst case for novelty)."""
    rng = np.random.default_rng(1)
    T = 3000.0
    for trial in range(2000):
        lam, m = float(rng.choice([0.01, 0.002, 0.0005])), int(rng.integers(1, 11))
        ct = np.sort(rng.random(int(rng.integers(1, 80))) * T)
        cp = rng.integers(0, 30, len(ct))
        ia = inject_times(rng, m, 240.0, T)
        it = T - ia  # injection times
        # each injection targets the port of a real flow 0-120 s after it when one exists, else a random port
        ip = np.array([cp[(ct > x) & (ct <= x + 120)][0] if ((ct > x) & (ct <= x + 120)).any()
                       else rng.integers(0, 30) for x in it], dtype=int) if len(it) else np.array([], int)
        allt, allp = np.r_[ct, it], np.r_[cp, ip]
        fo = novelty_flags(allt, allp)                     # observed flags (injections included)
        fc = novelty_flags(ct, cp)                         # clean flags
        obs = stats(T - allt, fo, lam)
        cln = stats(T - ct, fc, lam)
        (wl, wh), (ml, mh), _ = box(*obs, lam, m, 1.0, novelty=True)
        assert wl - 1e-9 <= cln[0] <= wh + 1e-9
        assert ml - 1e-9 <= cln[1] <= mh + 1e-9, (trial, cln, obs, (ml, mh))


def _bridge_stream(rng, lam, m):
    """worst case for novelty: real flows to port p at t0 and t0 + 200 s (clean: the second is new); one injection to p
    at t0 + 100 s looks old itself and makes the second real flow look old. m bridges per 240 s window."""
    ct, cp, it, ip, port = [], [], [], [], 0
    for k in range(12):
        for j in range(m):
            t0 = k * 240.01 + j * 1e-3
            ct += [t0, t0 + 200.0]; cp += [port, port]; it.append(t0 + 100.0); ip.append(port); port += 1
    T = max(ct) + 1.0
    return np.array(ct), np.array(cp), np.array(it), np.array(ip), T


def test_novelty_flip_term_needed_and_sufficient():
    rng = np.random.default_rng(2)
    for lam in (0.01, 0.002):
        m = 10
        ct, cp, it, ip, T = _bridge_stream(rng, lam, m)
        assert in_budget(T - it, m, 240.0)
        allt, allp = np.r_[ct, it], np.r_[cp, ip]
        obs = stats(T - allt, novelty_flags(allt, allp), lam)
        cln = stats(T - ct, novelty_flags(ct, cp), lam)
        _, (ml, mh), _ = box(*obs, lam, m, 1.0, novelty=True)
        assert ml - 1e-9 <= cln[1] <= mh + 1e-9                     # sound with the flip term
        _, (ml0, mh0), _ = box(*obs, lam, m, 1.0, novelty=False)
        assert cln[1] > mh0 + 1e-6, (cln[1], mh0)                   # and the flip term is needed


def test_column_boxes_match_damped_box():
    """scripts/sensitivity.py's hd_* column boxes contain the clean / observed rows of brute-force streams."""
    import sys; sys.path.insert(0, "scripts")
    import sensitivity
    rng = np.random.default_rng(3)
    cols = [f"hd_src_{s}_0.01" for s in ("w", "fail", "newport", "newpeer")]
    mu, sd = np.zeros(4), np.ones(4)
    prep = lambda v: np.log1p(np.asarray(v, float))  # noqa: E731
    T = 3000.0
    for _ in range(500):
        m = int(rng.integers(1, 11))
        ct = np.sort(rng.random(int(rng.integers(1, 60))) * T); cp = rng.integers(0, 20, len(ct))
        cf = rng.integers(0, 2, len(ct)).astype(float)
        it = T - inject_times(rng, m, 240.0, T)
        ip = rng.integers(0, 20, len(it)); iff = rng.integers(0, 2, len(it)).astype(float)
        allt, allp, allf = np.r_[ct, it], np.r_[cp, ip], np.r_[cf, iff]
        def row(times, ports, fails):
            w, mf, _ = stats(T - times, fails, 0.01)
            _, mn, _ = stats(T - times, novelty_flags(times, ports), 0.01)
            return np.array([w, mf, mn, mn])
        clean, obs = row(ct, cp, cf), row(allt, allp, allf)
        lo, hi = sensitivity.deletion_box(prep(obs)[None], cols, mu, sd, m)
        assert np.all(lo[0] <= prep(clean) + 1e-9) and np.all(prep(clean) <= hi[0] + 1e-9), (clean, obs)
        lo, hi = sensitivity.box(prep(clean)[None], cols, mu, sd, m)
        assert np.all(lo[0] <= prep(obs) + 1e-9) and np.all(prep(obs) <= hi[0] + 1e-9), (clean, obs)
