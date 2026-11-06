"""Framing audit: can m spoofed flows per host key per 240 s make a detector alarm on innocent hosts?

    python scripts/frame_audit.py flows.csv --detector mymod:score [--m 10] [--tau T] [--out audit.json]

flows.csv  benign NetFlow records with columns FLOW_START_MILLISECONDS, IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_DST_PORT,
           OUT_PKTS, IN_BYTES (extra columns are passed through to the detector untouched).
--detector module:function, called as fn(flows: DataFrame, ctx: ndarray, ctx_cols: list) -> per-flow scores (higher =
           more anomalous). ctx is the causal per-host context of every flow (hostctx.COLS: counts and shares over
           10 s / 120 s windows for the source and destination key), recomputed after injection.
           `frame_audit:rule_f2b` (failed flows per source in 120 s) is a built-in example.
--tau      alarm threshold; default: the 99th percentile of the per-host-day maximum score on the clean flows, the same
           1% benign host-day rule the paper uses.

For each of the 8 injection patterns (random or fixed peer, random or victim port, failed or completed), it injects
flows under each innocent source key, greedily just before that key's own flows, at most m per key in any 240 s
(spoof_attack.inject), and reports the share of innocent host-days with an alarm on one of their own flows, clean and
attacked. A detector is framable at budget m if the worst attacked share is well above the clean share.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import spoof_attack as sa  # noqa: E402
from nids import hostctx  # noqa: E402

DAY_MS = 86_400_000


def rule_f2b(flows, ctx, cols):
    """fail2ban-style example: failed flows from the source key in the last 120 s."""
    c = {k: i for i, k in enumerate(cols)}
    return ctx[:, c["hc_src_n_120s"]] * ctx[:, c["hc_src_fail_120s"]]


def day_share(scores, keys, tau):
    """Share of host-days whose maximum score on their own flows exceeds tau."""
    s = pd.Series(scores).groupby(keys).max()
    return float((s > tau).mean())


def audit(flows, fn, m=10, tau=None):
    flows = flows.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    keys = flows.IPV4_SRC_ADDR.astype(str) + "/" + (flows.FLOW_START_MILLISECONDS // DAY_MS).astype(str)
    v = np.ones(len(flows), bool)
    clean = np.asarray(fn(flows, sa.host_context(flows[sa.COLS], flows[sa.COLS].iloc[:0], v), hostctx.COLS), float)
    if tau is None:
        tau = float(np.quantile(pd.Series(clean).groupby(keys).max(), 0.99))
    res = {"m": m, "tau": tau, "host_days": int(keys.nunique()), "clean": day_share(clean, keys, tau), "patterns": {}}
    for peer in (0, 1):
        for port in (0, 1):
            for fail in (1, 0):
                inj = sa.inject(flows[sa.COLS], "src", m, peer, port, fail, 0)
                ctx = sa.host_context(flows[sa.COLS], inj, v)
                name = f"peer{'rand' if peer else 'fixed'}_port{'rand' if port else 'victim'}_{'failed' if fail else 'done'}"
                res["patterns"][name] = day_share(np.asarray(fn(flows, ctx, hostctx.COLS), float), keys, tau)
    res["worst"] = max(res["patterns"].values())
    return res


def _selfcheck():
    """Synthetic: 40 hosts, completed flows only. The fail rule must be framed; a constant detector must not."""
    rng = np.random.default_rng(1)
    n = 4000
    f = pd.DataFrame({"FLOW_START_MILLISECONDS": np.sort(rng.integers(0, DAY_MS, n)),
                      "IPV4_SRC_ADDR": [f"10.0.0.{i}" for i in rng.integers(40, size=n)],
                      "IPV4_DST_ADDR": [f"10.1.0.{i}" for i in rng.integers(5, size=n)],
                      "L4_DST_PORT": rng.choice([80, 443, 53], n), "OUT_PKTS": 1, "IN_BYTES": 500})
    r = audit(f, rule_f2b, m=10, tau=5.0)
    assert r["clean"] == 0.0 and r["worst"] == 1.0, r
    assert audit(f, lambda fl, c, cols: np.zeros(len(fl)), m=10, tau=0.5)["worst"] == 0.0
    print("selfcheck ok", json.dumps({k: r[k] for k in ("clean", "worst")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("flows", nargs="?")
    ap.add_argument("--detector", default="frame_audit:rule_f2b")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck or not a.flows:
        return _selfcheck()
    mod, fn = a.detector.split(":")
    r = audit(pd.read_csv(a.flows), getattr(importlib.import_module(mod), fn), a.m, a.tau)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
