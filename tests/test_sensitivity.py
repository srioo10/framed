"""Brute-force soundness of Theorem 1 (PREREG §9 CERT): inject random flows through HostFlows, every real flow's
host context must stay inside sensitivity.box."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import sensitivity  # noqa: E402
from nids import hostctx  # noqa: E402


def stream(rng, n, hosts, span_ms):
    return pd.DataFrame({"FLOW_START_MILLISECONDS": np.sort(rng.integers(0, span_ms, n)),
                         "IPV4_SRC_ADDR": rng.choice(hosts, n), "IPV4_DST_ADDR": rng.choice(hosts, n),
                         "L4_DST_PORT": rng.choice([22, 53, 80, 443, 8080], n),
                         "OUT_PKTS": rng.choice([0, 1, 5], n, p=[0.3, 0.3, 0.4]),
                         "IN_BYTES": rng.integers(40, 5000, n), "family": "Benign"})


def prep(H, mu, sd):
    return (np.log1p(H) - mu) / sd


@pytest.mark.parametrize("m", [1, 3, 10])
def test_box_contains_every_injection(m):
    rng = np.random.default_rng(m)
    hosts = [f"10.0.0.{i}" for i in range(6)]
    cols = hostctx.COLS
    for trial in range(40):
        real = stream(rng, 300, hosts, 400_000)
        mu, sd = rng.normal(0, 1, len(cols)), rng.uniform(0.5, 2, len(cols))
        X = prep(hostctx.HostFlows(real).features(np.ones(len(real), bool)), mu, sd)
        lo, hi = sensitivity.box(X.astype(np.float32), cols, mu, sd, m)
        j = rng.integers(1, m + 1)  # <= m injected flows in total, so <= m per key in any interval
        inj = stream(rng, j, hosts + ["6.6.6.6"], 400_000)
        if trial % 2:  # adversarial: all j aimed at one victim flow's key, just before it (inside both windows)
            v = real.iloc[rng.integers(len(real))]
            t = int(v.FLOW_START_MILLISECONDS)
            inj["FLOW_START_MILLISECONDS"] = rng.integers(max(t - 9_000, 0), max(t, 1), j)
            side = "IPV4_DST_ADDR" if rng.random() < 0.5 else "IPV4_SRC_ADDR"
            inj[side] = v[side]
        inj["IN_BYTES"] = rng.choice([0, 10**9], j)
        inj["L4_DST_PORT"] = rng.choice([22, 80, 31337], j)
        both = pd.concat([real.assign(r=True), inj.assign(r=False)]).sort_values("FLOW_START_MILLISECONDS",
                                                                                kind="stable")
        H = hostctx.HostFlows(both.drop(columns="r")).features(np.ones(len(both), bool))[both.r.to_numpy()]
        X1 = prep(H, mu, sd)  # the stable sort keeps real rows in their original order
        assert (X1 >= lo).all() and (X1 <= hi).all(), f"trial {trial}: injection left the box"


def test_box_is_tight_on_counts():
    cols = hostctx.COLS
    X = np.zeros((1, len(cols)))
    lo, hi = sensitivity.box(X, cols, np.zeros(len(cols)), np.ones(len(cols)), 10)
    j = cols.index("hc_dst_n_10s")
    assert abs(hi[0, j] - np.log1p(10)) < 1e-3 and np.isinf(hi[0, cols.index("hc_dst_bytes_10s")])
