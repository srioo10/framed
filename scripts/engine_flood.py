"""Denial-of-state test of the C engine (PREREG §9 ENG). Needs engine/build/{onepass,model.txt,stream.csv} from
scripts/engine_export.py (nf_iot23, captures c7,c34,ch4).

Per (dilation k, spoofed flows per real flow N): Mirai start times dilated per source host (t0 + k (t - t0)); before
every real flow, N one-flow spoofed sources (fresh key, random benign destination, per-flow fields of a random benign
flow, same time). The engine runs at each table size; only real flows are scored.
Usage: .venv/bin/python scripts/engine_flood.py [--n 0,1,10,100] [--k 1,10,100]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import load  # noqa: E402

B = ROOT / "engine" / "build"
OUT = ROOT / "results" / "engine"
TABLES = (512, 4096, 65536)
DAY = 86_400_000


def fnv(s):
    h = 1469598103934665603
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & (2**64 - 1)
    return h or 1


def families(n):
    raw = load.LOADERS["nf_iot23"]()
    cap = raw["IPV4_SRC_ADDR"].astype(str).str.split(":").str[0]
    rows = np.flatnonzero(cap.isin(["c7", "c34", "ch4"]).to_numpy())
    rows = rows[np.argsort(raw["FLOW_START_MILLISECONDS"].to_numpy()[rows], kind="stable")]
    assert len(rows) == n, "stream.csv does not match the export captures"
    return raw["family"].astype(str).to_numpy()[rows]


def build(S, fam, k, n, rng):
    S = S.copy()
    att = fam != "Benign"
    t = S[0].to_numpy(np.int64).copy()
    t0 = pd.Series(t[att]).groupby(S[1][att].to_numpy()).transform("min").to_numpy()
    t[att] = t0 + k * (t[att] - t0)
    S[0] = t
    o = np.argsort(t, kind="stable")
    S, real_fam = S.iloc[o].reset_index(drop=True), fam[o]
    if n == 0:
        return S, real_fam, np.ones(len(S), bool)
    ben = np.flatnonzero(real_fam == "Benign")
    m = len(S) * n
    F = S.iloc[rng.choice(ben, m)].reset_index(drop=True)
    F[0] = np.repeat(S[0].to_numpy(), n)
    F[1] = "spoof:" + pd.Series(np.arange(m)).astype(str)
    F[2] = S[2].to_numpy()[rng.choice(ben, m)]
    pos = np.r_[np.repeat(np.arange(len(S)) * (n + 1), n) + np.tile(np.arange(n), len(S)), np.arange(len(S)) * (n + 1) + n]
    A = pd.concat([F, S], ignore_index=True)
    A.index = pos
    A = A.sort_index()
    return A, real_fam, ~A[1].astype(str).str.startswith("spoof:").to_numpy()


def measure(S, fam, out, ev, akeys):
    d = pd.DataFrame({"src": S[1].to_numpy(), "day": S[0].to_numpy() // DAY, "att": fam != "Benign",
                      "A": out[4].to_numpy(), "P": out[3].to_numpy()})
    d["i"] = d.groupby(["src", "day"]).cumcount()
    g = d.groupby(["src", "day"])
    hd = pd.DataFrame({"att": g.att.any(), "A": g.A.max(), "P": g.P.max(),
                       "delay": d[d.A == 1].groupby(["src", "day"]).i.min()}).fillna({"delay": -1})
    a, b = hd[hd.att], hd[~hd.att]
    return {"att_days": len(a), "att_A": int(a.A.sum()), "att_P": int(a.P.sum()),
            "delay": a.delay.astype(int).tolist(), "ben_days": len(b), "ben_A": float(b.A.mean()),
            "ben_P": float(b.P.mean()), "att_evictions": int(np.isin(ev, akeys).sum()), "evictions": len(ev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="0,1,10,100")
    ap.add_argument("--k", default="1,10,100")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(B / "stream.csv", header=None, dtype={1: str, 2: str})
    fam = families(len(S))
    akeys = np.array([fnv(s) for s in np.unique(S[1][fam != "Benign"])], dtype=np.uint64)
    res = []
    for k in map(int, a.k.split(",")):
        for n in map(int, a.n.split(",")):
            A, real_fam, real = build(S, fam, k, n, np.random.default_rng([k, n]))
            f = B / f"flood_k{k}_n{n}.csv"
            A.to_csv(f, header=False, index=False)
            for T in TABLES:
                ev_f = B / "evict.log"
                with open(f) as fin, open(B / "flood_out.csv", "w") as fout:
                    p = subprocess.run([str(B / "onepass"), "--model", str(B / "model.txt"), "--hosts", str(T),
                                        "--evict-log", str(ev_f)], stdin=fin, stdout=fout, stderr=subprocess.PIPE, text=True)
                assert p.returncode == 0, p.stderr
                out = pd.read_csv(B / "flood_out.csv", header=None)[real].reset_index(drop=True)
                ev = np.loadtxt(ev_f, dtype=np.uint64, usecols=0, ndmin=1) if ev_f.stat().st_size else np.empty(0, np.uint64)
                r = {"k": k, "n": n, "table": T, **measure(A[real].reset_index(drop=True), real_fam, out, ev, akeys),
                     "engine": p.stderr.strip().splitlines()[-1]}
                print(json.dumps(r), flush=True)
                res.append(r)
            f.unlink()
    pd.DataFrame(res).to_csv(OUT / "flood.csv", index=False)


if __name__ == "__main__":
    main()
