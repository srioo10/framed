"""Adaptive spoofed-flow injection against benign victims (PREREG §9 CERT, kill test K1; later K4 and §9 CONF).

Victims: 300 benign test host-days (seed 0) of one outer config (held = first family), seed 1. The attacker knows the
victims' flow times and injects realisable flows (spoofed source, any destination / port / OUT_PKTS / IN_BYTES) right
before each victim flow, at most m per targeted host key in any 240 s interval (greedy, so the budget is never
exceeded). Host context is recomputed exactly by hostctx.HostFlows on every flow that shares a key with a victim plus
the injected flows. Pattern grid per m: side {dst, src, both} x new peer per flow {0, 1} x new port {0, 1} x
failed {0, 1} x IN_BYTES {0, 1e3, 1e9}. Detectors: RF, XGB (arena configs) and the latest bowl champion. Benign
host-day alarm rate with the P and A host-day methods (thresholds from calibration hosts, as flood_detectors.py).
K1 decision (method P): the best pattern at m <= 50 reaches >= 10% and >= 5 x clean for some detector.
--blind (PREREG §9 AUDIT F4): the attacker does not know victim flow times, only which 10-min bins a targeted key is
active in; it injects on a fixed lattice (spacing ceil(240 s / m), random phase) through every active bin, so the
budget still holds. Same grid; champion = round 3 (A). Writes k1_blind_{ds}.csv.
Usage: .venv/bin/python scripts/spoof_attack.py nf_ctu13 [--m 10,50] [--blind]
"""
import argparse
import itertools
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import hostctx, load, models  # noqa: E402

SEED, N_DAYS, BUDGET_MS = 1, 300, 240_000
COLS = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "OUT_PKTS", "IN_BYTES"]
OUT = ROOT / "results" / "cert"
PUB = ["kitnet", "iforest", "horuseye", "deep_sad", "cade", "kitnet_online", "horuseye_online"]  # PREREG §9 BSAN


def champion(r=None):
    r = r or max(int(p.parent.name[5:]) for p in (ROOT / "results" / "bowl").glob("round*/death.json"))
    return r, json.loads((ROOT / "results" / "bowl" / f"round{r}" / "death.json").read_text())["champion"]


def setup(ds):
    cache = load.CACHE / f"{ds}.parquet"  # read only the 6 columns (the full 57-column NF-ToN frame is ~10 GB)
    raw = pd.read_parquet(cache, columns=COLS) if cache.exists() else load.LOADERS[ds]()
    light = pd.DataFrame({c: raw[c].to_numpy() for c in COLS})
    light["IPV4_SRC_ADDR"], light["IPV4_DST_ADDR"] = (light[c].astype(str) for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"))
    del raw
    bank = sb.Bank(ds, SEED, "host", 40000, 200000)
    return light, bank


def detectors(bank, held, rnd=None, baselines=False):
    """{name: score(prepped X)} + prep, trained on the outer config's fit rows like score_bank / arena."""
    import bowl
    P = (held,)
    fit, _ = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    bi = code["Benign"]
    dfit = bank.frame(fit, sb.role_removal(P, "fit"))
    prep = models.Prep().fit(dfit)
    Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit]])
    rf, xg = models.rf(SEED).fit(Xf, yf), models.xgb(SEED).fit(Xf, yf)
    r, g = champion(rnd)
    z = {"X_fit": Xf, "y_fit": yf, "k": len(known), "bi": bi, "nflow": len(bank.feats),
         "teacher": models.oof_proba(Xf, yf, SEED).astype(np.float32),
         "cols": list(dfit.columns), "mu": prep.mu, "sd": prep.sd}  # clip / cert genes need the Prep columns
    for k in ("cal", "pseudo", "es"):
        z[f"X_{k}"] = Xf[:10]
    m, h, _ = bowl.run_pair(g, z, "cpu", SEED, 30)
    nl = lambda p: -np.log(np.maximum(p, 1e-45))  # noqa: E731
    dets = {"rf": lambda X: nl(rf.predict_proba(X)[:, bi]), "xgb": lambda X: nl(xg.predict_proba(X)[:, bi]),
            f"champion_r{r}": lambda X: h.score(m, X.astype(np.float32), "cpu")}
    if baselines:  # PREREG §9 K14: the published-IDS baselines, fitted as in CONF amendment 4 (+ BSAN online variants)
        import conf_baselines
        for b in (PUB if baselines is True else baselines):
            dets[b] = conf_baselines.fit(b, z, SEED)
    return dets, prep, (m, h, bi)


def calibrate(bank, held, dets, prep):
    z = np.load(ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(z["ips"], SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(z["src"][z["fam"] != "Benign"]))
    ch = lambda r: cal_host[z["src"][r]] & benign_host[z["src"][r]]  # noqa: E731
    rank, cal = bank.rank[ch(bank.rank)], bank.cal[ch(bank.cal)]
    Xr, Xc = (prep.transform(bank.frame(r, (held,))) for r in (rank, cal))
    key = z["src"][cal].astype(np.int64) * hs.MUL + z["t"][cal] // hs.DAY_MS
    out = {}
    for name, f in dets.items():
        ref = np.sort(f(Xr))
        st = hs.stats(key, z["t"][cal], f(Xc), ref)
        out[name] = (ref, {mm: float(hs.threshold(hs.day_max(key, st[mm]))) for mm in ("P", "A")})
    return out, z


def victims(bank, z):
    rows = bank.test_benign
    key = z["src"][rows].astype(np.int64) * hs.MUL + z["t"][rows] // hs.DAY_MS
    days = np.random.default_rng(0).permutation(np.unique(key))[:N_DAYS]
    keep = np.isin(key, days)
    return rows[keep], key[keep]


def inject(V, side, m, peer, port, fail, nbytes):
    """Injected flows (DataFrame, COLS) for victim flows V (time-sorted DataFrame), greedy within budget m per key."""
    out, rng = [], np.random.default_rng(0)
    for col in {"dst": ["IPV4_DST_ADDR"], "src": ["IPV4_SRC_ADDR"], "both": ["IPV4_DST_ADDR", "IPV4_SRC_ADDR"]}[side]:
        for kv, grp in V.groupby(col, sort=False):
            q = deque()
            for t, vport in zip(grp.FLOW_START_MILLISECONDS.to_numpy(), grp.L4_DST_PORT.to_numpy()):
                while q and q[0] <= t - m - BUDGET_MS:
                    q.popleft()
                if len(q) >= m:  # budget spent for this key: skip without trying each slot
                    continue
                for j in range(m, 0, -1):  # inject at t - j ms, oldest first
                    tau = t - j
                    while q and q[0] <= tau - BUDGET_MS:
                        q.popleft()
                    if len(q) >= m or (q and tau <= q[-1]):
                        continue
                    q.append(tau)
                    other = f"6.6.{rng.integers(256)}.{rng.integers(256)}" if peer else "6.6.6.6"
                    s, d = (other, kv) if col == "IPV4_DST_ADDR" else (kv, other)
                    out.append((tau, s, d, int(rng.integers(40000, 65000)) if port else vport, 0 if fail else 1,
                                nbytes))
    return pd.DataFrame(out, columns=COLS)


def inject_blind(V, side, m, peer, port, fail, nbytes, bin_ms=600_000):
    """Lattice injection without victim flow times: per targeted key, every ceil(BUDGET_MS / m) ms (random phase)
    inside each bin_ms bin that holds a victim flow of that key. Any 240 s interval holds <= m injected flows."""
    out, rng, d = [], np.random.default_rng(0), -(-BUDGET_MS // m)
    for col in {"dst": ["IPV4_DST_ADDR"], "src": ["IPV4_SRC_ADDR"], "both": ["IPV4_DST_ADDR", "IPV4_SRC_ADDR"]}[side]:
        for kv, grp in V.groupby(col, sort=False):
            phase, vport = int(rng.integers(d)), int(grp.L4_DST_PORT.iloc[0])
            for b in np.unique(grp.FLOW_START_MILLISECONDS.to_numpy() // bin_ms):
                lo = b * bin_ms
                for tau in range(lo + (phase - lo) % d, lo + bin_ms, d):
                    other = f"6.6.{rng.integers(256)}.{rng.integers(256)}" if peer else "6.6.6.6"
                    s, dd = (other, kv) if col == "IPV4_DST_ADDR" else (kv, other)
                    out.append((tau, s, dd, int(rng.integers(40000, 65000)) if port else vport, 0 if fail else 1,
                                nbytes))
    return pd.DataFrame(out, columns=COLS)


def host_context(S, inj, vmask):
    both = pd.concat([S.assign(v=vmask), inj.assign(v=False)], ignore_index=True)
    both = both.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    both["family"] = "Benign"
    H = hostctx.HostFlows(both.drop(columns="v")).features(np.ones(len(both), bool))
    pos = np.flatnonzero(both.v.to_numpy())
    order = np.argsort(both.index.to_numpy()[pos], kind="stable")  # back to the victims' original order
    return H[pos][order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--m", default="10,50")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--baselines", nargs="*")  # no names: all of PUB
    a = ap.parse_args()
    t0 = time.time()
    light, bank = setup(a.dataset)
    held = bank.families[0]
    pub = None if a.baselines is None else (a.baselines or True)
    dets, prep, _ = detectors(bank, held, 3 if a.blind else None, pub)
    cal, z = calibrate(bank, held, dets, prep)
    rows, key = victims(bank, z)
    raw_rows = bank.kept[rows]
    V = light.iloc[raw_rows]
    S_mask = light.IPV4_SRC_ADDR.isin(set(V.IPV4_SRC_ADDR)) | light.IPV4_DST_ADDR.isin(set(V.IPV4_DST_ADDR))
    S_idx = np.flatnonzero(S_mask.to_numpy())
    S = light.iloc[S_idx].reset_index(drop=True)
    vmask = np.isin(S_idx, raw_rows)
    assert vmask.sum() == len(rows)
    base = bank.frame(rows, ())  # flow features + exact host context of the clean stream (rows sorted, as in S)
    H0 = host_context(S, light.iloc[:0], vmask)
    assert np.allclose(H0, base[hostctx.COLS].to_numpy(), rtol=1e-4, atol=1e-3), "subset host context != bank"
    t = z["t"][rows]
    Vs = V.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    res = []
    grid = list(itertools.product(("dst", "src", "both"), (0, 1), (0, 1), (0, 1), (0, 1_000, 10**9)))
    for m in [0] + list(map(int, a.m.split(","))):
        for side, peer, port, fail, nb in grid if m else [("none", 0, 0, 0, 0)]:
            inj = (inject_blind if a.blind else inject)(Vs, side, m, peer, port, fail, nb) if m else light.iloc[:0]
            X = base.copy()
            X[hostctx.COLS] = host_context(S, inj, vmask) if m else H0
            Xp = prep.transform(X)
            for name, f in dets.items():
                ref, th = cal[name]
                st = hs.stats(key, t, f(Xp), ref)
                r = {"ds": a.dataset, "held": held, "m": m, "side": side, "peer": peer, "port": port, "fail": fail,
                     "bytes": nb, "detector": name, "n_inj": len(inj), "days": int(len(np.unique(key)))}
                for mm in ("P", "A"):
                    r[f"ben_{mm}"] = float(pd.Series(st[mm] > th[mm]).groupby(key).any().mean())
                res.append(r)
            print(m, side, peer, port, fail, nb, {r["detector"]: round(r["ben_P"], 3) for r in res[-len(dets):]},
                  f"{time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(res)
    R.to_csv(OUT / f"k1_{'blind_' if a.blind else ''}{'' if pub is None else 'pub_' if pub is True else 'pub_' + '+'.join(pub) + '_'}{a.dataset}.csv", index=False)
    clean = R[R.m == 0].set_index("detector").ben_P
    best = R[R.m > 0].groupby("detector").ben_P.max()
    k1 = {d: bool(best[d] >= 0.10 and best[d] >= 5 * max(clean[d], 1e-9)) for d in best.index}
    print("clean P", clean.to_dict(), "\nbest attacked P", best.to_dict(), "\nK1 rule per detector", k1)


if __name__ == "__main__":
    main()
