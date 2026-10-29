import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flowtok import Coder, contexts  # noqa: E402

V = 12


def stream(n=400, hosts=7, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, V, n), rng.integers(0, hosts, n), np.sort(rng.integers(0, 10**6, n))


def test_probabilities_sum_to_one():
    tok, src, t = stream()
    C, _ = contexts(tok, src, t, np.ones(len(tok), bool), V)
    m = Coder(V).fit(C, tok)
    for i in (0, 5, 77):  # every token in one row's context
        Ci = np.repeat(C[i:i + 1], V, 0)
        assert abs(np.exp2(m.logp(Ci, np.arange(V))).sum() - 1) < 1e-9


def test_time_warp_invariance():  # T1: pacing one host (monotone warp of its times) leaves contexts unchanged
    tok, src, t = stream()
    pop = np.ones(len(tok), bool)
    C0, _ = contexts(tok, src, t, pop, V)
    t2 = t.copy()
    h = src == 3
    t2[h] = t[h].min() + 1000 * (t[h] - t[h].min())
    C1, _ = contexts(tok, src, t2, pop, V)
    assert np.array_equal(C0, C1)


def test_spoof_invariance():  # T2: flows from other source keys never change a host's contexts
    tok, src, t = stream()
    rng = np.random.default_rng(1)
    m = 1000
    tok2, src2, t2 = np.r_[tok, rng.integers(0, V, m)], np.r_[src, 100 + np.arange(m)], np.r_[t, rng.integers(0, 10**6, m)]
    C0, _ = contexts(tok, src, t, np.ones(len(tok), bool), V)
    C1, _ = contexts(tok2, src2, t2, np.ones(len(tok2), bool), V)
    assert np.array_equal(C0, C1[:len(tok)])


def test_ple_bounded_and_saturating():  # PREREG §9 PLE: encoding stays in [0, 1] and saturates past the fit range
    from ple import Encoder
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=500), rng.integers(0, 3, 500)].astype(np.float32)
    e = Encoder().fit(X)
    big = np.array([[1e9, 1e9], [X[:, 0].max(), 2]], np.float32)
    E = e.transform(np.r_[X, big])
    assert E.min() >= 0 and E.max() <= 1
    assert np.array_equal(E[-2], E[-1])  # a spoofed huge value encodes like the largest training value
