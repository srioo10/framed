"""Theorem 1 of PREREG §9 CERT: exact per-flow box of the host-context columns under spoofed-flow injection.

For a flow whose window (key k, width w) holds n real flows, c of them failed / novel, byte sum S, any j <= m injected
flows of the same key in the 240 s before the flow keep
  n -> [n, n + m]
  fail share     in [c / (n+m), (c+m) / (n+m)]           (injections only add flows, failed or not)
  newport/peer   in [(c-m)+ / (n+m), (c+m) / (n+m)]       (each injection adds <= 1 novelty, removes <= 1 real one)
  mean bytes     in [S / (n+m), inf)                      (bounded above only by the cert-mode input clip)
Each bound is monotone in j, so its value at j = m covers every j <= m. Tight per group: for each (n, c, S) some consistent
stream and m injections reach every finite bound (the novelty lower bound needs injections that cancel a real flow's
novelty while being non-novel themselves; tests/test_sensitivity.py::test_bounds_are_attained). The product box over
groups is not jointly attained: that slack, plus IBP's, is the certificate's looseness. Non-host-context columns are untouched (the
victim's own flow fields are not attacker-controlled). Works in Prep space (signed log1p, then standardize).
Usage: lo, hi = box(X, cols, mu, sd, m)
"""
import numpy as np

EPS = 1e-4  # widening in standardized units, covers float32 rounding of the recovered counts


def box(X, cols, mu, sd, m):
    """(n, d) prepped rows -> (lo, hi) of the same shape; mu / sd are Prep's per-column log-space mean / scale."""
    assert not any(c.startswith("hl_") for c in cols), "Theorem 1 not yet extended to long windows (PREREG K7)"
    X = np.asarray(X, np.float64)
    lo, hi = X.copy(), X.copy()
    raw = lambda j: np.expm1(np.maximum(X[:, j] * sd[j] + mu[j], 0))  # noqa: E731  all host-context values are >= 0
    back = lambda j, v: (np.log1p(v) - mu[j]) / sd[j]  # noqa: E731
    groups = {c.rsplit("_", 2)[0] + "_" + c.rsplit("_", 1)[1] for c in cols if c.startswith("hc_")}  # hc_src_10s, ...
    for g in groups:
        key, w = g.rsplit("_", 1)
        idx = {s: cols.index(f"{key}_{s}_{w}") for s in ("n", "fail", "newport", "newpeer", "bytes")}
        n = np.rint(raw(idx["n"]))
        d, dm = np.maximum(n, 1), np.maximum(n + m, 1)
        lo[:, idx["n"]], hi[:, idx["n"]] = X[:, idx["n"]], back(idx["n"], n + m)
        for s in ("fail", "newport", "newpeer"):
            c = raw(idx[s]) * d
            c_lo = c if s == "fail" else np.maximum(c - m, 0)
            lo[:, idx[s]], hi[:, idx[s]] = back(idx[s], c_lo / dm), back(idx[s], np.minimum((c + m) / dm, 1))
        lo[:, idx["bytes"]], hi[:, idx["bytes"]] = back(idx["bytes"], raw(idx["bytes"]) * d / dm), np.inf
    hc = np.array([c.startswith("hc_") for c in cols])
    lo[:, hc] = np.minimum(lo[:, hc], X[:, hc]) - EPS
    hi[:, hc] = np.maximum(hi[:, hc], X[:, hc]) + EPS
    return lo, hi


def deletion_box(X, cols, mu, sd, m):
    """Theorem 1' (robust-alarm gating): box of every host context reachable from an OBSERVED row by deleting the
    j <= m injected flows. Observed window: n', c' failed / novel. Deleting j_in <= min(m, n') window flows gives
      n              in [(n'-m)+, n']
      fail share     in [(c'-m)+ / (n'-m), c' / (n'-m)]               (deletions only remove failures)
      newport/peer   in [(c'-m)+ / (n'-m), (c'+m) / (n'-m)]           (a deleted flow ends <= 1 novelty and makes
                                                                       <= 1 later real flow of its pair novel)
      mean bytes     in [0, inf)                                      (per-flow bytes are not in the row)
    shares capped to [0, 1]; when n' <= m every share may be anything in [0, 1]. The clean row of an attacked flow lies
    in the deletion box of its observed row, so an alarm whose lower bound over this box clears tau would have fired
    without the injections."""
    assert not any(c.startswith("hl_") for c in cols), "Theorem 1 not yet extended to long windows (PREREG K7)"
    X = np.asarray(X, np.float64)
    lo, hi = X.copy(), X.copy()
    raw = lambda j: np.expm1(np.maximum(X[:, j] * sd[j] + mu[j], 0))  # noqa: E731
    back = lambda j, v: (np.log1p(v) - mu[j]) / sd[j]  # noqa: E731
    groups = {c.rsplit("_", 2)[0] + "_" + c.rsplit("_", 1)[1] for c in cols if c.startswith("hc_")}
    for g in groups:
        key, w = g.rsplit("_", 1)
        idx = {s: cols.index(f"{key}_{s}_{w}") for s in ("n", "fail", "newport", "newpeer", "bytes")}
        n = np.rint(raw(idx["n"]))
        r = np.maximum(n - m, 0)
        dr, empty = np.maximum(r, 1), r < 1
        lo[:, idx["n"]] = back(idx["n"], r)
        for s in ("fail", "newport", "newpeer"):
            c = raw(idx[s]) * np.maximum(n, 1)
            s_lo = np.where(empty, 0, np.maximum(c - m, 0) / dr)
            s_hi = np.where(empty, 1, np.minimum((c if s == "fail" else c + m) / dr, 1))
            lo[:, idx[s]], hi[:, idx[s]] = back(idx[s], s_lo), back(idx[s], s_hi)
        lo[:, idx["bytes"]], hi[:, idx["bytes"]] = back(idx["bytes"], 0), np.inf
    hc = np.array([c.startswith("hc_") for c in cols])
    lo[:, hc] = np.minimum(lo[:, hc], X[:, hc]) - EPS
    hi[:, hc] = np.maximum(hi[:, hc], X[:, hc]) + EPS
    return lo, hi
