"""Adaptive-attacker perturbations of the test-time host-context population (PREREG §9 ADV).

dilate:k  per (source host, attack family) group, start times t -> t0 + k (t - t0), t0 = the group's first flow
cover:r   per group, r x min(n, COVER_CAP) copies of benign flows from benign hosts, source set to the attacker,
          start times uniform over [first - 60 s, last + 60 s] of the group
Training roles never see these flows: the bank builds a second HostFlows on the perturbed population and uses it only
for test roles.
"""
import numpy as np
import pandas as pd

from nids import hostctx

COLS = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "OUT_PKTS", "IN_BYTES", "family"]
COVER_CAP, PAD_MS = 20_000, 60_000  # shortcut: cover per group capped (RAM); floods get a lower cover ratio than r


class Perturbed:
    def __init__(self, raw, spec, seed, kept):
        mode, v = spec.split(":")
        v = float(v)
        assert mode in ("dilate", "cover"), spec
        light = pd.DataFrame({c: raw[c].to_numpy() for c in COLS})
        light["family"] = light["family"].astype(str)
        light["IPV4_SRC_ADDR"] = light["IPV4_SRC_ADDR"].astype(str)
        fam, src = light["family"].to_numpy(), light["IPV4_SRC_ADDR"].to_numpy()
        att = fam != "Benign"
        g = pd.MultiIndex.from_arrays([src[att], fam[att]]).factorize()[0]
        t = light["FLOW_START_MILLISECONDS"].to_numpy(np.int64).copy()
        ta = t[att]
        lo = pd.Series(ta).groupby(g).transform("min").to_numpy()
        hi = pd.Series(ta).groupby(g).transform("max").to_numpy()
        if mode == "dilate":
            t[att] = lo + np.round(v * (ta - lo)).astype(np.int64)
            light["FLOW_START_MILLISECONDS"] = t
        self.n_kept = len(kept)
        self.src_kept = src[kept]
        self.t_kept = t[kept]
        self.cover_row = self.cover_fam = self.cover_src = self.cover_t = np.empty(0)
        if mode == "cover":
            rng = np.random.default_rng([seed, 7])
            attacker = np.unique(src[att])
            pool = np.flatnonzero(~np.isin(src[kept], attacker) & (fam[kept] == "Benign"))  # df positions
            gi = np.flatnonzero(att)
            order = np.argsort(g, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]
            rows, fams, srcs, ts = [], [], [], []
            for s, e in zip(starts, np.r_[starts[1:], len(order)]):
                j = gi[order[s]]
                m = int(v * min(e - s, COVER_CAP))
                a, b = lo[order[s]] - PAD_MS, hi[order[s]] + PAD_MS
                rows.append(rng.choice(pool, m))
                fams.append(np.full(m, fam[j]))
                srcs.append(np.full(m, src[j]))
                ts.append(rng.integers(a, b + 1, m))
            self.cover_row, self.cover_fam = np.concatenate(rows), np.concatenate(fams)
            self.cover_src, self.cover_t = np.concatenate(srcs), np.concatenate(ts).astype(np.int64)
            c = light.iloc[kept[self.cover_row]].reset_index(drop=True)
            c["IPV4_SRC_ADDR"], c["FLOW_START_MILLISECONDS"], c["family"] = self.cover_src, self.cover_t, self.cover_fam
            light = pd.concat([light, c], ignore_index=True)
        self.host = hostctx.HostFlows(light)
        self.host.keep = np.r_[kept, len(raw) + np.arange(len(self.cover_row))]

    def meta(self):
        return {"t_kept": self.t_kept, "cover_row": self.cover_row, "cover_fam": self.cover_fam.astype("U"),
                "cover_src": self.cover_src.astype("U"), "cover_t": self.cover_t}
