import numpy as np
import pandas as pd

from nids import hostctx


def _raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "FLOW_START_MILLISECONDS": np.sort(rng.integers(0, 300_000, n)),
        "IPV4_SRC_ADDR": rng.choice(["a", "b", "c"], n), "IPV4_DST_ADDR": rng.choice(["x", "y"], n),
        "L4_DST_PORT": rng.choice([22, 80, 443, 8080], n), "OUT_PKTS": rng.integers(0, 3, n),
        "IN_BYTES": rng.integers(40, 2000, n), "family": rng.choice(["Benign", "Benign", "Fuzz", "DoS"], n)})


def test_matches_brute_force_and_is_causal():
    raw = _raw()
    h = hostctx.HostFlows(raw)
    pop = raw.family.to_numpy() != "DoS"
    F = h.features(pop)
    t = raw.FLOW_START_MILLISECONDS.to_numpy() - raw.FLOW_START_MILLISECONDS.min()
    src, port, dst = raw.IPV4_SRC_ADDR.to_numpy(), raw.L4_DST_PORT.to_numpy(), raw.IPV4_DST_ADDR.to_numpy()
    w = hostctx.WINDOWS_MS[0]
    for i in range(0, len(raw), 37):
        m = pop & (src == src[i]) & (t >= t[i] - w) & (t < t[i])  # strictly earlier, DoS removed
        j = np.flatnonzero(m)
        assert F[i, 0] == len(j)
        if len(j):
            assert np.isclose(F[i, 1], (raw.OUT_PKTS.to_numpy()[j] == 0).mean())
            assert np.isclose(F[i, 4], raw.IN_BYTES.to_numpy()[j].mean())
            novel = [not np.any(pop & (src == src[k]) & (port == port[k]) & (t >= t[k] - w) & (t < t[k])) for k in j]
            # same-ms pairs are not strictly earlier: brute force treats them as unseen, the fast path as seen
            tie = [np.any(pop & (src == src[k]) & (port == port[k]) & (t == t[k]) & (np.arange(len(t)) != k)) for k in j]
            if not any(tie):
                assert np.isclose(F[i, 2], np.mean(novel))
    # a flow's own family-mates after it never change its features
    later = t > t[200]
    pop2 = pop & ~later
    assert np.allclose(h.features(pop2)[:201][t[:201] < t[200]], F[:201][t[:201] < t[200]])


def test_isolated_ignores_other_attack_families():
    raw = _raw(seed=1)
    h = hostctx.HostFlows(raw)
    fam = raw.family.to_numpy()
    iso = h.isolated()
    m = fam == "Fuzz"
    assert np.allclose(iso[m], h.features(np.isin(fam, ["Benign", "Fuzz"]))[m])


def test_permutation_keeps_blocks():
    raw = _raw(seed=2)
    before = raw.IPV4_SRC_ADDR.copy()
    h0, h1 = hostctx.HostFlows(raw), hostctx.HostFlows(raw, permute_seed=0)
    assert raw.IPV4_SRC_ADDR.equals(before)  # input frame is not modified
    assert not np.array_equal(h0.key["src"], h1.key["src"])
    # per block, each permuted code occurs as often as some original code: same IP multiset inside the block
    blk = (raw.FLOW_START_MILLISECONDS.to_numpy() - raw.FLOW_START_MILLISECONDS.min()) // 600_000
    for b in np.unique(blk):
        m = blk == b
        c0 = sorted(np.unique(h0.key["src"][m], return_counts=True)[1])
        c1 = sorted(np.unique(h1.key["src"][m], return_counts=True)[1])
        assert c0 == c1
