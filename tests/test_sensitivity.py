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
        dlo, dhi = sensitivity.deletion_box(X1, cols, mu, sd, m)  # Theorem 1': the clean row is reachable back
        assert (X >= dlo).all() and (X <= dhi).all(), f"trial {trial}: clean row outside the deletion box"


def test_box_is_tight_on_counts():
    cols = hostctx.COLS
    X = np.zeros((1, len(cols)))
    lo, hi = sensitivity.box(X, cols, np.zeros(len(cols)), np.ones(len(cols)), 10)
    j = cols.index("hc_dst_n_10s")
    assert abs(hi[0, j] - np.log1p(10)) < 1e-3 and np.isinf(hi[0, cols.index("hc_dst_bytes_10s")])


def test_bounds_are_attained():
    """Tightness of Theorem 1 per group (src key, 10 s window, m = 2): for a stream consistent with the target flow's own
    (n, c, S), explicit injections reach every finite bound of the box. Flows: (t, src, dst, port, OUT_PKTS, IN_BYTES)."""
    cols, m, T = hostctx.COLS, 2, (100_000, "A", "Z", 80, 1, 500)
    real = [(91_000, "A", "D1", 1001, 1, 300), (93_000, "A", "D2", 1002, 1, 700), (95_000, "A", "D3", 1003, 0, 100),
            (97_000, "A", "D4", 1004, 1, 900),
            # older same-pair flows just outside the window, so a killer injection can itself be non-novel
            (80_500, "A", "X1", 1001, 1, 50), (82_500, "A", "X2", 1002, 1, 50),
            (80_700, "A", "D1", 3001, 1, 50), (82_700, "A", "D2", 3002, 1, 50)]
    attacks = {  # name: (injections, {stat: bound attained})
        "count": ([(98_000, "A", "F1", 5001, 0, 0), (99_000, "A", "F2", 5002, 0, 0)],
                  {"n": "hi", "fail": "hi", "newport": "hi", "newpeer": "hi", "bytes": "lo"}),
        "fail_lo": ([(98_000, "A", "F1", 5001, 1, 0), (99_000, "A", "F2", 5002, 1, 0)], {"fail": "lo"}),
        # same port as r1 / r2: cancels their novelty, non-novel itself thanks to the 80.5 k / 82.5 k flows
        "newport_lo": ([(90_200, "A", "Y1", 1001, 1, 0), (90_400, "A", "Y2", 1002, 1, 0)], {"newport": "lo"}),
        "newpeer_lo": ([(90_300, "A", "D1", 6001, 1, 0), (90_500, "A", "D2", 6002, 1, 0)], {"newpeer": "lo"}),
    }
    names = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "OUT_PKTS", "IN_BYTES"]
    frame = lambda rows: pd.DataFrame(sorted(rows), columns=names).assign(family="Benign")  # noqa: E731
    zero, one = np.zeros(len(cols)), np.ones(len(cols))
    clean = frame(real + [T])
    i = int(np.flatnonzero(clean.FLOW_START_MILLISECONDS.to_numpy() == T[0])[0])
    X = prep(hostctx.HostFlows(clean).features(np.ones(len(clean), bool)), zero, one)
    lo, hi = sensitivity.box(X[[i]], cols, zero, one, m)
    for name, (inj, hits) in attacks.items():
        S = frame(real + [T] + inj)
        j = int(np.flatnonzero(S.FLOW_START_MILLISECONDS.to_numpy() == T[0])[0])
        X1 = prep(hostctx.HostFlows(S).features(np.ones(len(S), bool)), zero, one)[j]
        for stat, side in hits.items():
            c = cols.index(f"hc_src_{stat}_10s")
            bound = (lo if side == "lo" else hi)[0, c]
            assert abs(X1[c] - bound) <= 2 * sensitivity.EPS, f"{name}: {stat} {side} {X1[c]} vs {bound}"


@pytest.mark.parametrize("m", [1, 3])
def test_long_windows_budgeted_injection(m):
    """hl_* columns (PREREG K7): m injected flows in EVERY 240 s slot over the 2w before a victim flow (the most the
    budget allows) must stay inside the long-window box, and the clean row inside the observed row's deletion box."""
    rng = np.random.default_rng(10 + m)
    hosts = [f"10.0.0.{i}" for i in range(4)]
    cols, W = hostctx.LONG_COLS, hostctx.LONG_MS
    for trial in range(6):
        real = stream(rng, 400, hosts, 3 * W[0])
        mu, sd = rng.normal(0, 1, len(cols)), rng.uniform(0.5, 2, len(cols))
        feats = lambda df: hostctx.HostFlows(df).features(np.ones(len(df), bool), W)  # noqa: E731
        X = prep(feats(real), mu, sd)
        lo, hi = sensitivity.box(X.astype(np.float32), cols, mu, sd, m)
        v = real.iloc[rng.integers(len(real) // 2, len(real))]
        t = int(v.FLOW_START_MILLISECONDS)
        side = "IPV4_DST_ADDR" if trial % 2 else "IPV4_SRC_ADDR"
        d = -(-240_000 // m)  # lattice: any 240 s interval holds <= m injected flows
        times = np.arange(t - 1 - rng.integers(d), max(t - 2 * W[0], 0), -d)[::-1]
        inj = stream(rng, len(times), ["6.6.6.6"], 1)  # the other end is the attacker's own address
        inj["FLOW_START_MILLISECONDS"] = times
        inj[side] = v[side]
        inj["L4_DST_PORT"] = v.L4_DST_PORT if trial % 3 else rng.choice([22, 80, 31337], len(times))
        inj["IN_BYTES"] = rng.choice([0, 10**9], len(times))
        both = pd.concat([real.assign(r=True), inj.assign(r=False)]).sort_values("FLOW_START_MILLISECONDS",
                                                                                kind="stable")
        X1 = prep(feats(both.drop(columns="r"))[both.r.to_numpy()], mu, sd)
        assert (X1 >= lo).all() and (X1 <= hi).all(), f"trial {trial}: long-window injection left the box"
        dlo, dhi = sensitivity.deletion_box(X1, cols, mu, sd, m)
        assert (X >= dlo).all() and (X <= dhi).all(), f"trial {trial}: clean row outside the long deletion box"
