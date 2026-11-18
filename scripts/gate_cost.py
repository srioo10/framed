"""PREREG RETRY-GATE follow-up: model-free cap on the gate's detection cost. A flow is 'gateable' iff it got no reply
packet (OUT_PKTS == 0), no no-reply flow of the same source to a different (dst, dport) falls in the trailing 240 s,
and the source had a replied flow (OUT_BYTES > 0) to the same (dst, dport) in the prior 600 s. The gate can only
suppress blocks on gateable flows, so a family's share of gateable flows caps its TPR loss.
Usage: python scripts/gate_cost.py nf_ctu13 [nf_ugr16 ...]"""
import sys

import numpy as np
import pandas as pd

V2 = "--v2" in sys.argv
COLS = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "OUT_BYTES", "OUT_PKTS", "family"]


def gateable(D):
    D = D.sort_values(["IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"], kind="stable").reset_index(drop=True)
    t = D.FLOW_START_MILLISECONDS.to_numpy(np.int64)
    svc = D.groupby(["IPV4_DST_ADDR", "L4_DST_PORT"], sort=False).ngroup().to_numpy()
    src = pd.factorize(D.IPV4_SRC_ADDR)[0]
    nr = np.ones(len(D), bool) if V2 else D.OUT_PKTS.to_numpy() == 0  # v2: every flow counts
    g = np.zeros(len(D), bool)
    # A: among no-reply flows of a source, the latest flow with a different svc is the last one of the previous run
    idx = np.flatnonzero(nr)
    s, v, tt = src[idx], svc[idx], t[idx]
    new_run = np.r_[True, (s[1:] != s[:-1]) | (v[1:] != v[:-1])]
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    prev_end_t = np.full(len(run_start), -10**15, dtype=np.int64)
    same_src_prev = np.r_[False, s[run_start[1:]] == s[run_start[1:] - 1]]
    prev_end_t[same_src_prev] = tt[run_start[same_src_prev] - 1]
    okA = tt - prev_end_t[run_id] > 240_000
    # B: a replied flow from the same source to the same svc in the prior 600 s
    rep = D.OUT_BYTES.to_numpy() > 0
    key = src.astype(np.int64) * (svc.max() + 1) + svc
    R = pd.DataFrame({"k": key[rep], "t": t[rep]}).sort_values("t")
    Q = pd.DataFrame({"k": key[idx], "t": tt, "i": idx}).sort_values("t")
    M = pd.merge_asof(Q, R.rename(columns={"t": "tr"}), left_on="t", right_on="tr", by="k", allow_exact_matches=False)
    okB = pd.Series(((M.t - M.tr) <= 600_000).to_numpy(), index=M.i).reindex(idx).fillna(False).to_numpy()
    g[idx] = okA & okB
    return D.family.to_numpy(), (D.family != "Benign").to_numpy().astype(int), g


def main(dss):
    for ds in dss:
        D = pd.read_parquet(f"data/processed/{ds}.parquet", columns=COLS)
        fam, y, g = gateable(D)
        T = pd.DataFrame({"family": fam, "y": y, "g": g})
        per = T[T.y == 1].groupby("family").g.mean().sort_values(ascending=False)
        print(f"{ds}: benign gateable {T[T.y == 0].g.mean():.4%}; attack gateable {T[T.y == 1].g.mean():.4%}; "
              f"worst family {per.index[0]} {per.iloc[0]:.4%}")
        print("  " + ", ".join(f"{k} {v:.3%}" for k, v in per.items()))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a != "--v2"])
