"""K4 for robust-alarm gating (PREREG §9 CERT gate-K4 row): the K1 spoof attack (same victims, budget, pattern grid) on
a gated net. Per victim flow, a block under attack (IBP lower bound over the Theorem 1' deletion box of the attacked
row clears tau) must imply an alarm on the clean row; any violation is a bug. Reports flow and host-day rates of
alarms and blocks, clean vs the worst pattern. tau = clean 1%-FPR threshold on benign flows of calibration hosts.
Usage: .venv/bin/python scripts/gate_attack.py nf_ctu13 [--m 10] [--genome '{"cert": 4, "rdet": 4}']
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
    score = lambda X: h.score(m, X, "cpu")  # noqa: E731
    block = lambda X: fx.score_lower(m, *bowl.dbox(z, X, a.m), z["bi"], g["head"])  # noqa: E731
    meta = np.load(ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(meta["ips"], sa.SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(meta["src"][meta["fam"] != "Benign"]))
    cal = bank.cal[cal_host[meta["src"][bank.cal]] & benign_host[meta["src"][bank.cal]]]
    tau = np.quantile(score(tx(bank.frame(cal, P))), 0.99)
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
    alarm0, block0 = score(X0) > tau, block(X0) > tau
    day = lambda w: float(pd.Series(w).groupby(key).any().mean())  # noqa: E731
    res = [{"pattern": "clean", "alarm": alarm0.mean(), "block": block0.mean(), "alarm_day": day(alarm0),
            "block_day": day(block0), "violations": 0}]
    Vs = V.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    for side, peer, port, fail, nbytes in itertools.product(("dst", "src", "both"), (0, 1), (0, 1), (0, 1),
                                                           (0, 1_000, 10**9)):
        inj = sa.inject(Vs, side, a.m, peer, port, fail, nbytes)
        X = base.copy()
        X[hostctx.COLS] = sa.host_context(S, inj, vmask)
        X1 = tx(X)
        al, bl = score(X1) > tau, block(X1) > tau
        res.append({"pattern": f"{side}/peer{peer}/port{port}/fail{fail}/b{nbytes}", "alarm": al.mean(),
                    "block": bl.mean(), "alarm_day": day(al), "block_day": day(bl),
                    "violations": int((bl & ~alarm0).sum())})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()}),
              f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(sa.OUT / f"gate_k4_{a.dataset}.csv", index=False)
    w = R.iloc[1:]
    print(f"{a.dataset} m={a.m} victims={len(rows)} flows / {len(np.unique(key))} host-days\n"
          f"clean: alarm {alarm0.mean():.4f} block {block0.mean():.4f} | host-day alarm {day(alarm0):.3f} block "
          f"{day(block0):.3f}\nworst attack: alarm {w.alarm.max():.4f} block {w.block.max():.4f} | host-day alarm "
          f"{w.alarm_day.max():.3f} block {w.block_day.max():.3f}\nK4-gate violations (block under attack, no clean "
          f"alarm): {int(w.violations.sum())}")


if __name__ == "__main__":
    main()
