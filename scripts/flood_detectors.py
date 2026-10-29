"""Spoof-induced false alarms across detectors (PREREG §9 ENG-X). Python twin of scripts/engine_flood.py.

The §9 ENG flood streams (IoT-23 parity stream from engine_export.py) get host context from hostctx.HostFlows over all
flows (unbounded state); real flows are scored by every bank detector of one outer config, thresholded (P / A, 1%
budget) on the cal role of calibration hosts as engine_export.py. Reports benign host-day alarm rates per (k, N).
Usage: .venv/bin/python scripts/flood_detectors.py [--n 0,1,10] [--k 1,10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import engine_flood as ef  # noqa: E402
import host_stream as hs  # noqa: E402
import ple  # noqa: E402
import tabm  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import hostctx, kitnet, mcde, models  # noqa: E402

DS, HELD, SEED = "nf_iot23", "Okiru", 1
OUT = ROOT / "results" / "engine"


def detectors(bank):
    """{name: score(DataFrame of per-flow + host-context features)} trained as score_bank.run_config (outer, LEAN)."""
    P = (HELD,)
    fit, es = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    bi, k = code["Benign"], len(known)
    dfit = bank.frame(fit, P)
    prep = models.Prep().fit(dfit)
    Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit]])
    es = es[np.isin(bank.fam[es], known)]
    Xe, ye = prep.transform(bank.frame(es, P)), np.array([code[f] for f in bank.fam[es]])
    rf = models.rf(SEED).fit(Xf, yf)
    xg = models.xgb(SEED).fit(Xf, yf)
    mlp = models.train_mlp(Xf, yf, Xe, ye, k, seed=SEED)
    snm = models.train_mlp(Xf, yf, Xe, ye, k, seed=SEED, arch=models.SNMLP).eval()
    gda = models.FeatureGDA().fit(snm, Xf, yf, k)
    dben = bank.frame(bank.bsample, P)
    bprep = mcde.RobustPrep().fit(dben)
    Xb = bprep.transform(dben)
    iso = models.iforest(Xb, SEED)
    kit = kitnet.KitNET(SEED).fit(Xb)
    pl = ple.fit_models(Xf, yf, Xe, ye, k, bi, SEED)  # PREREG §9 PLE: ple_ddu, ple_ce, clip_ddu
    pl.update({a: f for a, f in tabm.fit_models(Xf, yf, Xe, ye, k, bi, SEED).items() if a in ("tabm", "tabm_raw")})
    return {**{a: (lambda d, f=f: f(prep.transform(d))) for a, f in pl.items()},"mlp_ce": lambda d: 1 - models.mlp_proba(mlp, prep.transform(d))[:, bi],
            "mlp_sn_gda": lambda d: gda.score(snm, prep.transform(d)),
            "rf": lambda d: 1 - rf.predict_proba(prep.transform(d))[:, bi],
            "xgb": lambda d: 1 - xg.predict_proba(prep.transform(d))[:, bi],
            "iforest": lambda d: -iso.score_samples(bprep.transform(d)),
            "kitnet": lambda d: kit.score(bprep.transform(d))}


def calibrate(bank, dets):
    """{name: (sorted reference, {P: h, A: h})} exactly as engine_export.py."""
    z = np.load(ROOT / "data" / "processed" / f"{DS}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(z["ips"], SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(z["src"][z["fam"] != "Benign"]))
    ch = lambda r: cal_host[z["src"][r]] & benign_host[z["src"][r]]  # noqa: E731
    rank, cal = bank.rank[ch(bank.rank)], bank.cal[ch(bank.cal)]
    drank, dcal = bank.frame(rank, (HELD,)), bank.frame(cal, (HELD,))
    key = z["src"][cal].astype(np.int64) * hs.MUL + z["t"][cal] // hs.DAY_MS
    out = {}
    for name, f in dets.items():
        ref = np.sort(f(drank))
        st = hs.stats(key, z["t"][cal], f(dcal), ref)
        out[name] = (ref, {m: float(hs.threshold(hs.day_max(key, st[m]))) for m in ("P", "A")})
    return out, bank.feats


def engine_model(path=ROOT / "engine" / "build" / "model.txt"):
    """The exact engine network (int8 weights, Prep parameters, reference, P / A thresholds) as a numpy scorer."""
    L = [line.split() for line in open(path)]
    head = L[0]
    bi = int(head[head.index("benign") + 1])
    mean, mu, sd = (np.array(L[i][1:], np.float64) for i in (1, 2, 3))
    layers, i = [], 4
    for _ in range(3):
        r, c, s = int(L[i][1]), int(L[i][2]), float(L[i][3])
        layers.append((np.array(L[i + 1], np.float64).reshape(r, c) * s, np.array(L[i + 2], np.float64)))
        i += 3
    ref = np.sort(np.array(L[i + 1], np.float64))
    th = dict(zip(("P", "A"), map(float, next(x for x in L if x[0] == "thresholds")[1:])))

    def score(d):
        X = d.to_numpy(np.float64, na_value=np.nan)
        X[~np.isfinite(X)] = np.nan
        X = np.where(np.isnan(X), mean, X)
        h = (np.sign(X) * np.log1p(np.abs(X)) - mu) / sd
        for j, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if j < 2:
                h = np.maximum(h, 0)
        p = np.exp(h - h.max(1, keepdims=True))
        return 1 - p[:, bi] / p.sum(1)
    return score, ref, th


def flood_frame(A, real, feats):
    """Per-flow fields + host context of the real flows of a flood stream (stream.csv column layout)."""
    raw = pd.DataFrame({"FLOW_START_MILLISECONDS": A[0].to_numpy(np.int64), "IPV4_SRC_ADDR": A[1].astype(str),
                        "IPV4_DST_ADDR": A[2].astype(str), "L4_DST_PORT": A[3].to_numpy(), "OUT_PKTS": A[4].to_numpy(),
                        "IN_BYTES": A[5].to_numpy(), "family": "Benign"})
    H = hostctx.HostFlows(raw).features(np.ones(len(raw), bool))[real]
    X = A.loc[real, list(range(6, 6 + len(feats)))].reset_index(drop=True)
    X.columns = feats
    X[hostctx.COLS] = H
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="0,1,10")
    ap.add_argument("--k", default="1,10")
    a = ap.parse_args()
    bank = sb.Bank(DS, SEED, "host", 40000, 200000)
    dets = detectors(bank)
    cal, feats = calibrate(bank, dets)
    f, ref, th = engine_model()  # PREREG §9 ENG-X amendment: harness check on the exact engine network
    dets["engine_w"], cal["engine_w"] = f, (ref, th)
    S = pd.read_csv(ef.B / "stream.csv", header=None, dtype={1: str, 2: str})
    ref_s = pd.read_csv(ef.B / "reference.csv").score.to_numpy()  # parity: Python reference score of the clean stream
    X0 = flood_frame(S, np.ones(len(S), bool), feats)
    assert np.median(np.abs(f(X0) - ref_s)) < 1e-3, "engine_w does not reproduce the exported model"
    fam = ef.families(len(S))
    rows = []
    for k in map(int, a.k.split(",")):
        for n in map(int, a.n.split(",")):
            A, real_fam, real = ef.build(S, fam, k, n, np.random.default_rng([k, n]))  # same streams as engine_flood
            X = flood_frame(A, real, feats)
            t = A.loc[real, 0].to_numpy(np.int64)
            key = pd.factorize(A.loc[real, 1].astype(str))[0].astype(np.int64) * hs.MUL + t // hs.DAY_MS
            ben = ~pd.Series(real_fam != "Benign").groupby(key).any()
            for name, f in dets.items():
                ref, th = cal[name]
                st = hs.stats(key, t, f(X), ref)
                r = {"k": k, "n": n, "detector": name, "ben_days": int(ben.sum())}
                for m in ("P", "A"):
                    day = pd.Series(st[m] > th[m]).groupby(key).any()
                    r[f"ben_{m}"] = float(day[ben].mean())
                    r[f"att_{m}"] = float(day[~ben].mean())
                print(r, flush=True)
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "flood_detectors.csv", index=False)
    print(R.pivot_table(index=["detector"], columns=["k", "n"], values=["ben_A", "ben_P"]).round(3).to_string())


if __name__ == "__main__":
    main()
