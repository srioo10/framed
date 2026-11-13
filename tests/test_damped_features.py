"""HostFlows.damped against a direct O(n^2) reference."""
import numpy as np
import pandas as pd
from nids.hostctx import HostFlows


def test_damped_matches_reference():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({"FLOW_START_MILLISECONDS": np.sort(rng.integers(0, 2_000_000, n)) // 1000 * 1000,
                        "IPV4_SRC_ADDR": rng.choice(["1.1.1.1", "2.2.2.2", "3.3.3.3"], n),
                        "IPV4_DST_ADDR": rng.choice(["9.9.9.9", "8.8.8.8"], n),
                        "L4_DST_PORT": rng.integers(0, 6, n), "OUT_PKTS": rng.integers(0, 2, n),
                        "IN_BYTES": rng.integers(0, 100, n), "family": "Benign"})
    hf = HostFlows(raw)
    pop = rng.random(n) < 0.8
    X = hf.damped(pop, lams=(0.01,))
    t, k = hf.t, hf.key["src"]
    flags = [hf.fail, hf._novel(hf.pair_port["src"], pop, 120_000), hf._novel(hf.pair_peer["src"], pop, 120_000)]
    for i in range(n):
        m = pop & (k == k[i]) & (t < t[i])
        e = np.exp(-0.01 * (t[i] - t[m]) / 1000.0)
        w = e.sum()
        assert abs(X[i, 0] - w) < 1e-4 * max(1, w)
        for c, f in enumerate(flags):
            ref = (e * f[m]).sum() / max(w, 1e-12)
            assert abs(X[i, 1 + c] - ref) < 1e-4, (i, c)
