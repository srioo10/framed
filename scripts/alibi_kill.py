"""PREREG ALIBI-KILL (2026-11-11): can blocking be scoped to new pairs? No model, no training.
pair = (source key, peer address, service port = min(src port, dst port)); persistence of a pair at a flow = number of
distinct earlier 240 s windows in which the pair had a flow; established iff persistence >= P.
rho   = share of an innocent source's benign flows on established pairs (flows >= 1 h after the source's first flow)
sigma_own = share of attack flows on pairs not established for their own source key at that time
sigma_any = share of attack flows whose (peer, service port) is never established for any innocent source
Writes results/cert/alibi_kill.csv. Usage: python scripts/alibi_kill.py"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "family"]
W, WARM = 240_000, 3_600_000


def persistence(src, pair, win):
    """Per flow: number of distinct earlier windows in which (src, pair) had a flow."""
    g = pd.DataFrame({"s": src, "q": pair, "w": win})
    u = g.drop_duplicates().sort_values("w")
    u["c"] = u.groupby(["s", "q"]).cumcount()
    return g.merge(u, on=["s", "q", "w"], how="left")["c"].to_numpy()


def one(ds):
    d = pd.read_parquet(ROOT / "data" / "processed" / f"{ds}.parquet", columns=C)
    t = d.FLOW_START_MILLISECONDS.to_numpy()
    src = pd.factorize(d.IPV4_SRC_ADDR)[0]
    svc = np.minimum(d.L4_SRC_PORT.to_numpy(), d.L4_DST_PORT.to_numpy()).astype(np.int64)
    gp = pd.factorize(d.IPV4_DST_ADDR)[0].astype(np.int64) * 65536 + svc  # (peer, service port)
    att = (d.family != "Benign").to_numpy()
    del d
    c = persistence(src, gp, t // W)
    bad_src = np.zeros(src.max() + 1, bool)
    bad_src[src[att]] = True
    first = pd.Series(t).groupby(src).transform("min").to_numpy()
    ev = ~att & ~bad_src[src] & (t >= first + WARM)  # innocent sources' flows after warm-up
    rows = []
    for P in (1, 3, 10):
        est = c >= P
        host = pd.Series(est[ev]).groupby(src[ev]).agg(["mean", "size"])
        any_est = np.unique(gp[est & ~att & ~bad_src[src]])
        rows.append({"dataset": ds, "P": P, "n_benign": int(ev.sum()), "n_hosts": int((host["size"] >= 20).sum()),
                     "rho_flow": est[ev].mean(), "rho_host_median": host["mean"][host["size"] >= 20].median(),
                     "n_attack": int(att.sum()), "sigma_own": (~est[att]).mean(),
                     "sigma_any": (~np.isin(gp[att], any_est)).mean()})
    return rows


if __name__ == "__main__":
    # self-check: one pair seen in windows 0, 0, 1, 3 -> persistence 0, 0, 1, 2
    assert persistence(np.zeros(4, int), np.zeros(4, int), np.array([0, 0, 1, 3])).tolist() == [0, 0, 1, 2]
    D = pd.DataFrame([r for ds in ("nf_ctu13", "nf_ugr16", "nf_cic", "nf_ton") for r in one(ds)])
    D.to_csv(ROOT / "results" / "cert" / "alibi_kill.csv", index=False)
    print(D.round(3).to_string())
