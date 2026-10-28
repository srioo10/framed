import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calib as c  # noqa: E402


def test_iid_recovers_ville():
    rng = np.random.default_rng(0)
    X = c.logg(np.maximum(rng.random(400_000), 1e-6))
    st = np.zeros(len(X), bool)
    st[::2000] = True
    th, K = c.fit_markov(X, st)
    assert abs(th - 1) < 0.1 and K < 1.2


def test_markov_chain_alarm_rate():
    # correctly specified: X takes two values with sticky Markov switching, so the bin chain is exact
    rng = np.random.default_rng(1)

    def days(n, L=400):
        out = []
        for _ in range(n):
            s, xs = 0, np.empty(L)
            for i, r in enumerate(rng.random(L)):
                s = s if r > (0.05 if s == 0 else 0.2) else 1 - s
                xs[i] = 1.0 if s else -0.6
            out.append(xs)
        return out
    cal = days(200)
    X = np.concatenate(cal)
    st = np.zeros(len(X), bool)
    st[::400] = True
    th, K = c.fit_markov(X, st)
    assert th is not None and np.isfinite(th)
    test = days(3000)
    rate = np.mean([(lambda C, t: (C > c.m_threshold(t, th, K)).any())(*c.cusum(x, np.r_[True, np.zeros(399, bool)]))
                    for x in test])
    assert rate <= c.ALPHA + 3 * np.sqrt(c.ALPHA / 3000)
