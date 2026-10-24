import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np

import host_stream as hs


def test_stats_match_loops():
    rng = np.random.default_rng(0)
    n = 400
    key, t, s, ref = rng.integers(0, 5, n), rng.permutation(n), rng.random(n), rng.random(300)
    st = hs.stats(key, t, s, ref)
    rs = np.sort(ref)
    for k in range(5):
        idx = np.flatnonzero(key == k)
        idx = idx[np.argsort(t[idx])]
        C, e, qs = 0.0, 1.0, []
        for j, i in enumerate(idx):
            u = (1 + (rs >= s[i]).sum()) / (len(rs) + 1)
            q = -np.log(u)
            qs.append(q)
            C = max(0.0, C + np.log(np.mean(hs.KAPPAS * u ** (hs.KAPPAS - 1))))
            e = hs.LAM * q + (1 - hs.LAM) * e
            assert np.isclose(st["A"][i], C) and np.isclose(st["E"][i], e)
            assert np.isclose(st["W"][i], np.mean(qs[-hs.WIN:]))
