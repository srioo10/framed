"""Causal per-host context for NetFlow-v3 (PREREG §9, host-context row).

For a flow starting at t (ms) and each key (source host, destination host) and window W, over the population flows
of the same key that started in [t - W, t) (strictly earlier; IPs are keys, never features):
  n        number of flows
  fail     share with no reply packet (OUT_PKTS == 0)
  newport  share that opened a (key, dst port) pair not seen in the W before them
  newpeer  share that opened a (key, other host) pair not seen in the W before them
  bytes    mean IN_BYTES
Population = raw flows minus a removal set of families (the score bank's role rule), or, for the `host_iso` control,
Benign plus the flow's own family only (no other attack family can light up a shared attacker host).
"""
import numpy as np
import pandas as pd

WINDOWS_MS = (10_000, 120_000)
STATS = ("n", "fail", "newport", "newpeer", "bytes")
COLS = [f"hc_{k}_{s}_{w // 1000}s" for w in WINDOWS_MS for k in ("src", "dst") for s in STATS]


def _codes(*cols):
    return pd.MultiIndex.from_arrays(cols).factorize()[0].astype(np.int64) if len(cols) > 1 else \
        pd.factorize(cols[0])[0].astype(np.int64)


class HostFlows:
    def __init__(self, raw, permute_seed=None):
        """permute_seed: control that shuffles source and destination IPs across flows within each 10-minute block
        (host identity destroyed, traffic volume per block kept)."""
        self.t = raw["FLOW_START_MILLISECONDS"].to_numpy(np.int64)
        self.t = self.t - self.t.min()
        self.span = int(self.t.max()) + max(WINDOWS_MS) + 1
        self.family = raw["family"].astype(str).to_numpy()
        src, dst = raw["IPV4_SRC_ADDR"].astype(str).to_numpy(), raw["IPV4_DST_ADDR"].astype(str).to_numpy()
        port = raw["L4_DST_PORT"].to_numpy()
        if permute_seed is not None:
            rng, block = np.random.default_rng(permute_seed), self.t // 600_000
            for a in (src, dst):
                o = np.lexsort((rng.random(len(a)), block))
                a[np.sort(o)] = a[o]  # rows sorted by block, then a random permutation inside each block
        self.key = {"src": _codes(src), "dst": _codes(dst)}
        self.pair_port = {"src": _codes(src, port), "dst": _codes(dst, port)}
        self.pair_peer = {"src": _codes(src, dst), "dst": _codes(dst, src)}
        self.fail = (raw["OUT_PKTS"].to_numpy() == 0).astype(np.float64)
        self.bytes = raw["IN_BYTES"].to_numpy(np.float64)
        self._cache = {}
        self.keep = slice(None)  # rows kept by clean; the bank sets it so caches hold only those rows

    def _novel(self, pair, pop, w):
        """Per flow: no earlier population flow of the same pair in [t - w, t)."""
        idx = np.flatnonzero(pop)
        o = idx[np.lexsort((self.t[idx], pair[idx]))]
        same = np.r_[False, pair[o][1:] == pair[o][:-1]]
        gap = np.r_[np.inf, np.diff(self.t[o]).astype(np.float64)]
        out = np.zeros(len(self.t), bool)
        out[o] = ~same | (gap > w)
        return out

    def _window_sums(self, key, pop, w, values):
        """Sum of each value array over population flows of the same key with start in [t - w, t), for every flow."""
        idx = np.flatnonzero(pop)
        s = key[idx] * self.span + self.t[idx]
        o = np.argsort(s, kind="stable")
        s = s[o]
        q = key * self.span + self.t
        hi, lo = np.searchsorted(s, q, side="left"), np.searchsorted(s, q - w, side="left")
        out = []
        for v in values:
            cs = np.r_[0.0, np.cumsum(v[idx][o])]
            out.append(cs[hi] - cs[lo])
        return out

    def features(self, pop):
        """float32 (n_flows, len(COLS)) for every raw flow against population mask `pop`."""
        cols = []
        ones = np.ones(len(self.t))
        for w in WINDOWS_MS:
            for k in ("src", "dst"):
                newport = self._novel(self.pair_port[k], pop, w).astype(np.float64)
                newpeer = self._novel(self.pair_peer[k], pop, w).astype(np.float64)
                n, f, p, r, b = self._window_sums(self.key[k], pop, w, (ones, self.fail, newport, newpeer, self.bytes))
                d = np.maximum(n, 1)
                cols += [n, f / d, p / d, r / d, b / d]
        return np.column_stack(cols).astype(np.float32)

    def removal(self, R):
        """Features of the kept rows with families R removed from the population (cached per removal set)."""
        R = tuple(sorted(R))
        if R not in self._cache:
            self._cache[R] = self.features(~np.isin(self.family, list(R)))[self.keep]
        return self._cache[R]

    def isolated(self):
        """Control: each flow's population is Benign plus its own family."""
        if "iso" not in self._cache:
            out = np.empty((len(self.t), len(COLS)), np.float32)
            ben = self.family == "Benign"
            for f in np.unique(self.family):
                m = self.family == f
                out[m] = self.features(ben | m)[m]
            self._cache["iso"] = out[self.keep]
        return self._cache["iso"]
