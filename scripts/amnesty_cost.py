"""PREREG AMNESTY-COST: model-free cap on what veteran amnesty can let through. A flow is 'veteran' iff its 5-tuple
(src, sport, dst, dport, proto) was active continuously (gap between one flow's end and the next flow's start <= 30 s)
since >= 240 s before this flow starts. Amnesty can only pass veteran flows of a blocked key, so a family's veteran share
caps the share of its flows amnesty could let through. Usage: python scripts/amnesty_cost.py nf_ctu13 [nf_ugr16 ...]"""
import sys

import numpy as np
import pandas as pd

COLS = ["FLOW_START_MILLISECONDS", "FLOW_DURATION_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR",
        "L4_DST_PORT", "PROTOCOL", "family"]
TUP = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL"]


def veteran(D, gap=30_000, w=240_000):
    D = D.sort_values(TUP + ["FLOW_START_MILLISECONDS"], kind="stable").reset_index(drop=True)
    s = D.FLOW_START_MILLISECONDS.to_numpy(np.int64); e = s + D.FLOW_DURATION_MILLISECONDS.fillna(0).to_numpy().astype(np.int64)  # end = start + duration (CTU-13 has no end column)
    tid = D.groupby(TUP, sort=False).ngroup().to_numpy()
    same = np.r_[False, tid[1:] == tid[:-1]]
    prev_end = np.r_[np.int64(-10**15), e[:-1]]
    chain_break = ~same | (s - prev_end > gap)
    chain_id = np.cumsum(chain_break) - 1
    chain_start = s[np.flatnonzero(chain_break)][chain_id]
    return D.family.to_numpy(), s - chain_start >= w


def main(dss):
    for ds in dss:
        D = pd.read_parquet(f"data/processed/{ds}.parquet", columns=COLS)
        fam, v = veteran(D)
        T = pd.DataFrame({"family": fam, "v": v})
        att = T[T.family != "Benign"]
        per = att.groupby("family").v.mean().sort_values(ascending=False)
        print(f"{ds}: benign veteran {T[T.family == 'Benign'].v.mean():.3%}; attack veteran {att.v.mean():.3%}; "
              f"family-mean {per.mean():.3%}; worst {per.index[0]} {per.iloc[0]:.3%}")
        print("  " + ", ".join(f"{k} {x:.2%}" for k, x in per.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
