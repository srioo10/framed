"""Causal connection-window features for unsw_raw (PREREG §9, ct_* row).

Window of a flow at second t: the smallest run of whole seconds [t0, t] holding >= 100 population flows,
clipped to the flow's own stime//600 block; the current second is included in full (1-s decision latency).
The population is every raw flow (duplicates included: they are real traffic) minus a removal set of families.
"""
import numpy as np
import pandas as pd

WINDOW = 100
BLOCK = 600
KEYS = {  # recomputed feature -> flow attributes that must match
    "ct_src_ltm": ("srcip",),
    "ct_dst_ltm": ("dstip",),
    "ct_dst_src_ltm": ("srcip", "dstip"),
    "ct_src_dport_ltm": ("srcip", "dsport"),
    "ct_dst_sport_ltm": ("dstip", "sport"),
    "ct_srv_src": ("service", "srcip"),
    "ct_srv_dst": ("service", "dstip"),
}


class Flows:
    """Raw unsw_raw flows as integer arrays. `raw` is the loader frame (before clean)."""

    def __init__(self, raw):
        st = raw["stime"].to_numpy(np.int64)
        self.t = st - st.min()
        self.family = raw["family"].astype(str).to_numpy()
        cols = {c: raw[c].astype(str).str.replace("﻿", "").str.replace("ï»¿", "") if c in ("srcip", "dstip")
                else raw[c].astype(str) for c in ("srcip", "dstip", "sport", "dsport", "service")}
        self.codes = {}
        for name, attrs in KEYS.items():
            key = cols[attrs[0]] if len(attrs) == 1 else cols[attrs[0]] + "|" + cols[attrs[1]]
            self.codes[name] = pd.factorize(key)[0].astype(np.int64)
        self.block_start = self.t - st % BLOCK  # stime//600 blocks in absolute time

    def window_start(self, pop):
        """t0 per flow for population mask `pop` (unclipped and block-clipped)."""
        tp = np.sort(self.t[pop])
        # flows in [s, t] = #(tp <= t) - #(tp < s); t0 = largest s with count >= WINDOW, else the earliest second
        upto = np.searchsorted(tp, self.t, side="right")
        need = upto - WINDOW  # index in tp of the WINDOW-th flow counting back; its second is t0
        t0 = np.where(need >= 0, tp[np.clip(need, 0, None)], 0)
        return t0, np.maximum(t0, self.block_start)

    def features(self, pop, query):
        """Recomputed ct_* (int32, shape (len(query), 7)) for flows `query` against population `pop`."""
        _, t0 = self.window_start(pop)
        t0, t = t0[query], self.t[query]
        out = np.empty((len(query), len(KEYS)), np.int32)
        for i, name in enumerate(KEYS):
            c = self.codes[name]
            span = int(self.t.max()) + 1
            s = np.sort(c[pop] * span + self.t[pop])
            q = c[query] * span
            out[:, i] = np.searchsorted(s, q + t, side="right") - np.searchsorted(s, q + t0, side="left")
        return out

    def exposure(self, family, query):
        """Per query flow: (any `family` flow in its all-flow window, a `family` flow sharing a ct_* key)."""
        pop = np.ones(len(self.t), bool)
        _, t0 = self.window_start(pop)
        t0, t = t0[query], self.t[query]
        f = np.sort(self.t[self.family == family])
        anyf = np.searchsorted(f, t, side="right") - np.searchsorted(f, t0, side="left") > 0
        keyed = np.zeros(len(query), bool)
        span = int(self.t.max()) + 1
        for name in KEYS:
            c = self.codes[name]
            s = np.sort(c[self.family == family] * span + self.t[self.family == family])
            q = c[query] * span
            keyed |= np.searchsorted(s, q + t, side="right") - np.searchsorted(s, q + t0, side="left") > 0
        return anyf, keyed
