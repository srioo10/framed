"""K1 on UNSW-NB15 (PREREG §9 AUDIT F4, threat breadth): spoofed-flow injection against its connection-count features
`ct_*` (windows.Flows: per flow, flows sharing a key among the last >= 100 population flows, causal, 1-s resolution).

Victims: up to 300 benign test host-days (seed 0) of the first outer config, seed 1, as spoof_attack.py. Injected
flows sit in the victim flow's own second (the current second counts in full), at most m per targeted host key in any
240 s interval (greedy). Pattern grid per m: side {dst: spoofed source to the victim's destination, src: the victim's
source address to a random 6.6.x.x, both: victim source to victim destination (budget on both keys)} x same service
{0, 1} x same port {0, 1} (dst: victim sport, src / both: victim dsport, both also sport). ct_* is recomputed exactly
on all raw flows plus the injected ones. Detectors: RF, XGB (arena configs) on the ct_causal feature set. Benign
host-day alarms (methods P and A, thresholds from calibration hosts) as spoof_attack.py.
Pass (F4): best pattern at m <= 50 gives RF or XGB >= 10% and >= 5 x clean (method P). Writes results/cert/k1_unsw_raw.csv.
Usage: .venv/bin/python scripts/k1_unsw.py [--m 10,50]
"""
import argparse
import itertools
import time
from collections import deque

import numpy as np
import pandas as pd

import host_stream as hs
import score_bank as sb
import spoof_attack as sa
from nids import load, models, windows

DS, SEED, BUDGET_S = "unsw_raw", 1, 240
COLS = ["stime", "srcip", "dstip", "sport", "dsport", "service", "family"]


def detectors(bank, held):
    P = (held,)
    fit, _ = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    bi = known.index("Benign")
    dfit = bank.frame(fit, sb.role_removal(P, "fit"))
    prep = models.Prep().fit(dfit)
    Xf, yf = prep.transform(dfit), np.array([known.index(f) for f in bank.fam[fit]])
    rf, xg = models.rf(SEED).fit(Xf, yf), models.xgb(SEED).fit(Xf, yf)
    nl = lambda p: -np.log(np.maximum(p, 1e-45))  # noqa: E731
    return {"rf": lambda X: nl(rf.predict_proba(X)[:, bi]), "xgb": lambda X: nl(xg.predict_proba(X)[:, bi])}, prep


def inject(V, side, m, srv, port):
    """Injected raw flows (DataFrame, COLS) for victim flows V (time-sorted), greedy within m per key per 240 s."""
    keys = {"dst": ("dstip",), "src": ("srcip",), "both": ("srcip", "dstip")}[side]
    q, out, rng = {}, [], np.random.default_rng(0)
    for r in V.itertuples(index=False):
        t = r.stime
        dq = [q.setdefault((k, getattr(r, k)), deque()) for k in keys]
        for d in dq:
            while d and d[0] <= t - BUDGET_S:
                d.popleft()
        n = m - max(len(d) for d in dq)
        for _ in range(max(n, 0)):
            other = f"6.6.{rng.integers(256)}.{rng.integers(256)}"
            rp = str(int(rng.integers(40000, 65000)))
            s, d_ = {"dst": (other, r.dstip), "src": (r.srcip, other), "both": (r.srcip, r.dstip)}[side]
            sp = r.sport if port and side in ("dst", "both") else rp
            dp = r.dsport if port and side in ("src", "both") else rp
            out.append((t, s, d_, sp, dp, r.service if srv else "-", "Benign"))
            for d in dq:
                d.append(t)
    return pd.DataFrame(out, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="10,50")
    a = ap.parse_args()
    t0 = time.time()
    raw = load.LOADERS[DS]()
    light = pd.DataFrame({c: raw[c].astype(str).to_numpy() if c != "stime" else raw[c].to_numpy(np.int64) for c in COLS})
    light["srcip"] = light.srcip.str.replace("﻿", "").str.replace("ï»¿", "")
    del raw
    bank = sb.Bank(DS, SEED, "ct_causal", 40000, 200000)
    held = bank.families[0]
    dets, prep = detectors(bank, held)
    cal, z = sa.calibrate(bank, held, dets, prep)
    assert len(z["src"]) == len(bank.df), "hostmeta rows != bank rows"
    rows, key = sa.victims(bank, z)
    raw_rows = bank.kept[rows]
    base = bank.frame(rows, ())  # all flows as population: the deployed view
    ct = list(windows.KEYS)
    pop0 = np.ones(len(light), bool)
    assert (windows.Flows(light).features(pop0, raw_rows) == base[ct].to_numpy()).all(), "ct_* rebuild != bank"
    V = light.iloc[raw_rows].sort_values("stime", kind="stable")
    t = z["t"][rows]
    print(f"victims {len(rows)} flows, {len(np.unique(key))} host-days, held {held}, {time.time() - t0:.0f}s", flush=True)
    res = []
    grid = list(itertools.product(("dst", "src", "both"), (0, 1), (0, 1)))
    for m in [0] + list(map(int, a.m.split(","))):
        for side, srv, port in grid if m else [("none", 0, 0)]:
            inj = inject(V, side, m, srv, port) if m else light.iloc[:0]
            X = base.copy()
            if m:
                both = pd.concat([light, inj], ignore_index=True)
                X[ct] = windows.Flows(both).features(np.ones(len(both), bool), raw_rows)
            Xp = prep.transform(X)
            r0 = {"ds": DS, "held": held, "m": m, "side": side, "srv": srv, "port": port, "n_inj": len(inj),
                  "days": int(len(np.unique(key)))}
            for name, f in dets.items():
                ref, th = cal[name]
                st = hs.stats(key, t, f(Xp), ref)
                r = dict(r0, detector=name)
                for mm in ("P", "A"):
                    r[f"ben_{mm}"] = float(pd.Series(st[mm] > th[mm]).groupby(key).any().mean())
                res.append(r)
            print(m, side, srv, port, len(inj), {r["detector"]: round(r["ben_P"], 3) for r in res[-len(dets):]},
                  f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(res).to_csv(sa.OUT / f"k1_{DS}.csv", index=False)
    R = pd.DataFrame(res)
    clean = R[R.m == 0].set_index("detector").ben_P
    best = R[R.m > 0].groupby("detector").ben_P.max()
    k1 = {d: bool(best[d] >= 0.10 and best[d] >= 5 * max(clean[d], 1e-9)) for d in best.index}
    print("clean P", clean.to_dict(), "\nbest attacked P", best.to_dict(), "\nF4 rule per detector", k1)


if __name__ == "__main__":
    main()
