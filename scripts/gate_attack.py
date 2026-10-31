"""K4 for robust-alarm gating (PREREG §9 CERT gate-K4 row): the K1 spoof attack (same victims, budget, pattern grid) on
a gated net. Per victim flow, a block under attack (IBP lower bound over the Theorem 1' deletion box of the attacked
row clears tau) must imply an alarm on the clean row; any violation is a bug. Reports flow and host-day rates of
alarms and blocks, clean vs the worst pattern. tau = clean 1%-FPR threshold on benign flows of calibration hosts.
--clean-only (PREREG §9 item 3): clean victim alarms with the global tau vs per-host-group taus (quartiles of
hc_src_n_120s on calibration-host benign flows, each group its own 1% quantile); writes gate_calib_{ds}.csv.
Usage: .venv/bin/python scripts/gate_attack.py nf_ctu13 [--m 10] [--genome '{"cert": 4, "rdet": 4}'] [--clean-only]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bowl  # noqa: E402
import cert_trees as ct  # noqa: E402
import fighters as fx  # noqa: E402
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
import spoof_attack as sa  # noqa: E402
from nids import hostctx, models  # noqa: E402

G = {"family": "mlp", "enc": "raw", "e": 2, "w": 64, "depth": 2, "sn": 0, "act": "relu", "members": 1, "head": "nlog",
     "kd": 0.5, "syn": 0.5, "ls": 0.0, "focal": 2.0, "lr": 0.003, "clip": 1, "gate": 1, "cert": 4, "rdet": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--genome", default="{}")
    ap.add_argument("--clean-only", action="store_true")
    a = ap.parse_args()
    g = G | json.loads(a.genome)
    t0 = time.time()
    light, bank = sa.setup(a.dataset)
    held = bank.families[0]
    P = (held,)
    fit, _ = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    dfit = bank.frame(fit, sb.role_removal(P, "fit"))
    prep = models.Prep().fit(dfit)
    cols = list(dfit.columns)
    nb = [i for i, c in enumerate(cols) if c.startswith("hc_") and "_bytes_" in c]

    def tx(df):  # prepped, host-context bytes zeroed like the gated bowl
        X = prep.transform(df)
        X[:, nb] = 0
        return X
    Xf, yf = tx(dfit), np.array([code[f] for f in bank.fam[fit]])
    z = {"X_fit": Xf, "y_fit": yf, "k": len(known), "bi": code["Benign"], "nflow": len(bank.feats), "cols": cols,
         "mu": prep.mu, "sd": prep.sd, "teacher": models.oof_proba(Xf, yf, sa.SEED).astype(np.float32)}
    for k in ("cal", "pseudo", "es"):
        z[f"X_{k}"] = Xf[:10]
    m, h, _ = bowl.run_pair(g, z, "cpu", sa.SEED, 30)
    bi, nl = z["bi"], (lambda p: -np.log(np.maximum(p, 1e-45)))
    rf, xg = models.rf(sa.SEED).fit(Xf, yf), models.xgb(sa.SEED).fit(Xf, yf)
    dets = {"gated_net": (lambda X: h.score(m, X, "cpu"),
                          lambda X: fx.score_lower(m, *bowl.dbox(z, X, a.m), bi, g["head"])),
            "rf": (lambda X: nl(rf.predict_proba(X)[:, bi]), lambda X: ct.rf_lower(rf, *bowl.dbox(z, X, a.m), bi)),
            "xgb": (lambda X: nl(xg.predict_proba(X)[:, bi]),
                    lambda X: ct.xgb_upper(xg, *bowl.dbox(z, X, a.m), bi, z["k"], lower=True))}
    meta = np.load(ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(meta["ips"], sa.SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(meta["src"][meta["fam"] != "Benign"]))
    cal = bank.cal[cal_host[meta["src"][bank.cal]] & benign_host[meta["src"][bank.cal]]]
    Xcal = tx(bank.frame(cal, P))
    tau = {n: np.quantile(f(Xcal), 0.99) for n, (f, _) in dets.items()}
    rows, key = sa.victims(bank, meta)
    raw_rows = bank.kept[rows]
    V = light.iloc[raw_rows]
    S_idx = np.flatnonzero((light.IPV4_SRC_ADDR.isin(set(V.IPV4_SRC_ADDR))
                            | light.IPV4_DST_ADDR.isin(set(V.IPV4_DST_ADDR))).to_numpy())
    S = light.iloc[S_idx].reset_index(drop=True)
    vmask = np.isin(S_idx, raw_rows)
    base = bank.frame(rows, ())
    H0 = sa.host_context(S, light.iloc[:0], vmask)
    assert np.allclose(H0, base[hostctx.COLS].to_numpy(), rtol=1e-4, atol=1e-3), "subset host context != bank"
    X0 = tx(base)
    day = lambda w: float(pd.Series(w).groupby(key).any().mean())  # noqa: E731
    if a.clean_only:
        c = cols.index("hc_src_n_120s")
        edges = np.quantile(Xcal[:, c], [0.25, 0.5, 0.75])
        gc, g0 = np.searchsorted(edges, Xcal[:, c], "right"), np.searchsorted(edges, X0[:, c], "right")
        res = []
        for n, (f, _) in dets.items():
            sc, s0 = f(Xcal), f(X0)
            tg = np.array([np.quantile(sc[gc == q], 0.99) if (gc == q).any() else tau[n] for q in range(4)])  # tied edges can empty a group
            for mode, al in (("global", s0 > tau[n]), ("group", s0 > tg[g0])):
                res.append({"detector": n, "tau": mode, "alarm": al.mean(), "alarm_day": day(al),
                            **{f"alarm_g{q}": al[g0 == q].mean() if (g0 == q).any() else np.nan for q in range(4)},
                            **{f"share_g{q}": (g0 == q).mean() for q in range(4)}})
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[-1].items()}), flush=True)
        pd.DataFrame(res).to_csv(sa.OUT / f"gate_calib_{a.dataset}.csv", index=False)
        return
    alarm0, res = {}, []
    for n, (f, b) in dets.items():
        alarm0[n], b0 = f(X0) > tau[n], b(X0) > tau[n]
        res.append({"detector": n, "pattern": "clean", "alarm": alarm0[n].mean(), "block": b0.mean(),
                    "alarm_day": day(alarm0[n]), "block_day": day(b0), "violations": 0})
    Vs = V.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    for side, peer, port, fail, nbytes in itertools.product(("dst", "src", "both"), (0, 1), (0, 1), (0, 1),
                                                           (0, 1_000, 10**9)):
        inj = sa.inject(Vs, side, a.m, peer, port, fail, nbytes)
        X = base.copy()
        X[hostctx.COLS] = sa.host_context(S, inj, vmask)
        X1 = tx(X)
        for n, (f, b) in dets.items():
            al, bl = f(X1) > tau[n], b(X1) > tau[n]
            res.append({"detector": n, "pattern": f"{side}/peer{peer}/port{port}/fail{fail}/b{nbytes}",
                        "alarm": al.mean(), "block": bl.mean(), "alarm_day": day(al), "block_day": day(bl),
                        "violations": int((bl & ~alarm0[n]).sum())})
        print(json.dumps({r["detector"]: [round(r["alarm_day"], 3), round(r["block_day"], 3)] for r in res[-3:]}),
              f"{side}/{peer}/{port}/{fail}/{nbytes}", f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(sa.OUT / f"gate_k4_{a.dataset}.csv", index=False)
    print(f"{a.dataset} m={a.m} victims={len(rows)} flows / {len(np.unique(key))} host-days")
    for n, D in R.groupby("detector", sort=False):
        c, w = D.iloc[0], D.iloc[1:]
        print(f"{n}: clean alarm {c.alarm:.4f} block {c.block:.4f} | host-day {c.alarm_day:.3f} / {c.block_day:.3f}; "
              f"worst attack alarm {w.alarm.max():.4f} block {w.block.max():.4f} | host-day {w.alarm_day.max():.3f} / "
              f"{w.block_day.max():.3f}; K4-gate violations {int(w.violations.sum())}")


if __name__ == "__main__":
    main()
