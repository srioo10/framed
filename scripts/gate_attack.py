"""K4 for robust-alarm gating (PREREG §9 CERT gate-K4 row): the K1 spoof attack (same victims, budget, pattern grid) on
a gated net. Per victim flow, a block under attack (IBP lower bound over the Theorem 1' deletion box of the attacked
row clears tau) must imply an alarm on the clean row; any violation is a bug. Reports flow and host-day rates of
alarms and blocks, clean vs the worst pattern. tau = clean 1%-FPR threshold on benign flows of calibration hosts.
--clean-only (PREREG §9 item 3): clean victim alarms with the global tau vs per-host-group taus (quartiles of
hc_src_n_120s on calibration-host benign flows, each group its own 1% quantile); writes gate_calib_{ds}.csv.
--clean-only --tau-day (PREREG §9 HOSTDAY): flow tau vs tau_day (<= 1% of calibration-host benign host-days alarmed),
clean victim alarm / block rates and test attack TPR (plain, gated); writes gate_hostday_{ds}.csv.
--tau-day without --clean-only: the full K4 run at tau_day (CONF C3); --champion-a: champion A (bytes kept, cert-mode
host-context clip at inference only, CONF amendment 1); output gate_k4_{ds}[_A][_day][_tag].csv.
--m-attack M.. (PREREG §9 K17): inject M flows per key (box stays --m); --random N: N random peer / port / fail
mixtures per M; --no-bytes-grid: nbytes axis dropped (host-context bytes are zeroed before every detector); every M
also gets an "adaptive" row (per victim, a block under any pattern). --at (K18): also train the PGD adversarial-training
twin of the genome; detectors at_plain (block = alarm) and at_gated (IBP block). --throttle k.. (K16): attack rows with
each attacking source throttled to <= k attack flows per 240 s; writes gate_throttle_{ds}[_tag].csv.
--two-channel (K19): detector b2 = B's certified context channel at a 0.5% host-day tau plus a flow-only twin (the
victim's own 7 flow features, which injected flows cannot change) at a 0.5% host-day tau; alarm / block = either
channel (scores shifted by their taus, so b2's tau is 0).
Usage: .venv/bin/python scripts/gate_attack.py nf_ctu13 [--m 10] [--genome '{"cert": 4, "rdet": 4}'] [--clean-only]
"""
import argparse
import itertools
import json
from collections import deque
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bowl  # noqa: E402
import conf_clean  # noqa: E402
import torch  # noqa: E402
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
    ap.add_argument("--tau-day", action="store_true")
    ap.add_argument("--champion-a", action="store_true")  # CONF amendment 1: A keeps bytes, inference-only hc clip
    ap.add_argument("--m-attack", type=int, nargs="*")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--no-bytes-grid", action="store_true")
    ap.add_argument("--at", action="store_true")
    ap.add_argument("--throttle", type=int, nargs="*")
    ap.add_argument("--two-channel", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    g = (conf_clean.A if a.champion_a else G) | json.loads(a.genome)
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
    nb = [] if a.champion_a else [i for i, c in enumerate(cols) if c.startswith("hc_") and "_bytes_" in c]

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
    if a.champion_a:
        lo, hi = bowl.hc_clip(z)
        m.register_buffer("clo", torch.as_tensor(lo, dtype=torch.float32))
        m.register_buffer("chi", torch.as_tensor(hi, dtype=torch.float32))
    bi, nl = z["bi"], (lambda p: -np.log(np.maximum(p, 1e-45)))
    rf, xg = models.rf(sa.SEED).fit(Xf, yf), models.xgb(sa.SEED).fit(Xf, yf)
    def net(m, h, head):
        return lambda X: h.score(m, X, "cpu"), lambda X: fx.score_lower(m, *bowl.dbox(z, X, a.m), bi, head)
    dets = {"gated_net": net(m, h, g["head"]),
            "rf": (lambda X: nl(rf.predict_proba(X)[:, bi]), lambda X: ct.rf_lower(rf, *bowl.dbox(z, X, a.m), bi)),
            "xgb": (lambda X: nl(xg.predict_proba(X)[:, bi]),
                    lambda X: ct.xgb_upper(xg, *bowl.dbox(z, X, a.m), bi, z["k"], lower=True))}
    if a.at:  # PREREG §9 K18: same genome, PGD adversarial training in the same boxes instead of interval bounds
        ga = g | {"cert": 0, "rdet": 0, "adv": g.get("cert", 4), "advd": g.get("rdet", 2)}
        ma, hha, _ = bowl.run_pair(ga, z, "cpu", sa.SEED, 30)
        f_at, b_at = net(ma, hha, ga["head"])
        dets["at_plain"], dets["at_gated"] = (f_at, f_at), (f_at, b_at)
    meta = np.load(ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(meta["ips"], sa.SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(meta["src"][meta["fam"] != "Benign"]))
    cal = bank.cal[cal_host[meta["src"][bank.cal]] & benign_host[meta["src"][bank.cal]]]
    Xcal = tx(bank.frame(cal, P))
    tau = {n: np.quantile(f(Xcal), 0.99) for n, (f, _) in dets.items()}
    if a.tau_day and not a.clean_only:  # CONF C3: block tau per the HOSTDAY outcome
        ck = meta["src"][cal].astype(np.int64) * hs.MUL + meta["t"][cal] // hs.DAY_MS
        tau = {n: float(hs.threshold(hs.day_max(ck, f(Xcal)))) for n, (f, _) in dets.items()}
    if a.two_channel:  # PREREG §9 K19: spoof-invariant flow channel + certified context channel, 0.5% host-days each
        nf = z["nflow"]
        zf = {"X_fit": Xf[:, :nf], "y_fit": yf, "k": z["k"], "bi": bi, "nflow": nf, "cols": cols[:nf],
              "teacher": models.oof_proba(Xf[:, :nf], yf, sa.SEED).astype(np.float32)}
        for k in ("cal", "pseudo", "es"):
            zf[f"X_{k}"] = zf["X_fit"][:10]
        gf = g | {"cert": 0, "rdet": 0}
        mf, hf, _ = bowl.run_pair(gf, zf, "cpu", sa.SEED, 30)
        fB, bB = dets["gated_net"]
        fF = lambda X: hf.score(mf, X[:, :nf], "cpu")  # noqa: E731
        ck = meta["src"][cal].astype(np.int64) * hs.MUL + meta["t"][cal] // hs.DAY_MS
        tB, tF = (float(hs.threshold(hs.day_max(ck, f(Xcal)), 0.005)) for f in (fB, fF))
        dets["b2"] = (lambda X: np.maximum(fB(X) - tB, fF(X) - tF), lambda X: np.maximum(bB(X) - tB, fF(X) - tF))
        tau["b2"] = 0.0
        print(f"K19 taus: context {tB:.4f} (B alone {tau['gated_net']:.4f}), flow {tF:.4f}", flush=True)
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
    if a.throttle:
        throttle(a, bank, light, meta, dets, tau, tx, held)
        return
    if a.clean_only and a.tau_day:
        ck = meta["src"][cal].astype(np.int64) * hs.MUL + meta["t"][cal] // hs.DAY_MS
        rng = np.random.default_rng(0)
        cap = lambda r: np.sort(rng.permutation(r)[:5000])  # noqa: E731  shortcut: 5k rows per family keeps tree bounds fast
        T = {"unk": cap(bank.test[held]),
             "known": np.concatenate([cap(r) for f, r in bank.test.items() if f not in (held, "Benign")])}
        XT = {k: tx(bank.frame(r, ())) for k, r in T.items()}
        res = []
        for n, (f, b) in dets.items():
            td = float(hs.threshold(hs.day_max(ck, f(Xcal))))
            s0, b0 = f(X0), b(X0)
            sT = {k: (f(X), b(X)) for k, X in XT.items()}
            for mode, t in (("flow", tau[n]), ("day", td)):
                r = {"detector": n, "tau": mode, "tau_val": float(t), "alarm": (s0 > t).mean(), "block": (b0 > t).mean(),
                     "alarm_day": day(s0 > t), "block_day": day(b0 > t), "violations": int(((b0 > t) & ~(s0 > t)).sum())}
                for k, (s, bb) in sT.items():
                    r[f"tpr_{k}"], r[f"gtpr_{k}"] = (s > t).mean(), (bb > t).mean()
                r["gtpr_macro"] = (r["gtpr_unk"] + r["gtpr_known"]) / 2
                res.append(r)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        pd.DataFrame(res).to_csv(sa.OUT / f"gate_hostday_{a.dataset}.csv", index=False)
        return
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
    for r in res:
        r["m_attack"] = a.m
    Vs = V.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    for M in a.m_attack or [a.m]:
        pats = [((side, peer, port, fail, nbytes), None) for side, peer, port, fail, nbytes in itertools.product(
            ("dst", "src", "both"), (0, 1), (0, 1), (0, 1), (0,) if a.no_bytes_grid else (0, 1_000, 10**9))]
        rng = np.random.default_rng([0, M])
        pats += [((side, "mix", "mix", "mix", 0), rng.random(3)) for side in rng.choice(["dst", "src", "both"], a.random)]
        worst = {n: np.zeros(len(X0), bool) for n in dets}
        for (side, peer, port, fail, nbytes), p in pats:
            if p is None:
                inj = sa.inject(Vs, side, M, peer, port, fail, nbytes)
            else:  # per-flow mixture of the all-0 and all-1 injections (same times: the greedy schedule ignores flags)
                i0, i1 = sa.inject(Vs, side, M, 0, 0, 0, 0), sa.inject(Vs, side, M, 1, 1, 1, 0)
                assert (i0.FLOW_START_MILLISECONDS.to_numpy() == i1.FLOW_START_MILLISECONDS.to_numpy()).all()
                inj, u = i0.copy(), rng.random((len(i0), 3)) < p
                for j, cs in enumerate((["IPV4_SRC_ADDR", "IPV4_DST_ADDR"], ["L4_DST_PORT"], ["OUT_PKTS"])):
                    inj.loc[u[:, j], cs] = i1.loc[u[:, j], cs]
                peer, port, fail = (f"{q:.2f}" for q in p)
            X = base.copy()
            X[hostctx.COLS] = sa.host_context(S, inj, vmask)
            X1 = tx(X)
            for n, (f, b) in dets.items():
                al, bl = f(X1) > tau[n], b(X1) > tau[n]
                worst[n] |= bl
                res.append({"detector": n, "pattern": f"{side}/peer{peer}/port{port}/fail{fail}/b{nbytes}", "m_attack": M,
                            "alarm": al.mean(), "block": bl.mean(), "alarm_day": day(al), "block_day": day(bl),
                            "violations": int((bl & ~alarm0[n]).sum())})
            print(json.dumps({r["detector"]: [round(r["alarm_day"], 3), round(r["block_day"], 3)]
                              for r in res[-len(dets):]}), M, f"{side}/{peer}/{port}/{fail}/{nbytes}",
                  f"{time.time() - t0:.0f}s", flush=True)
        for n, w in worst.items():  # adaptive attacker: per victim flow, its best pattern
            res.append({"detector": n, "pattern": "adaptive", "m_attack": M, "alarm": np.nan, "block": w.mean(),
                        "alarm_day": np.nan, "block_day": day(w), "violations": int((w & ~alarm0[n]).sum())})
    R = pd.DataFrame(res)
    R.to_csv(sa.OUT / f"gate_k4_{a.dataset}{'_A' if a.champion_a else ''}{'_day' if a.tau_day else ''}"
                      f"{'_' + a.tag if a.tag else ''}.csv", index=False)
    print(f"{a.dataset} m={a.m} victims={len(rows)} flows / {len(np.unique(key))} host-days")
    for (n, M), D in R[R.pattern != "clean"].groupby(["detector", "m_attack"], sort=False):
        c, w, ad = R[(R.detector == n) & (R.pattern == "clean")].iloc[0], D[D.pattern != "adaptive"], D[D.pattern == "adaptive"].iloc[0]
        print(f"{n} M={M}: clean alarm {c.alarm:.4f} block {c.block:.4f} | host-day {c.alarm_day:.3f} / {c.block_day:.3f}; "
              f"worst attack alarm {w.alarm.max():.4f} block {w.block.max():.4f} | host-day {w.alarm_day.max():.3f} / "
              f"{w.block_day.max():.3f}; adaptive block_day {ad.block_day:.3f}; K4-gate violations "
              f"{int(w.violations.sum())} (adaptive {int(ad.violations)})")


def throttle(a, bank, light, meta, dets, tau, tx, held):
    """PREREG §9 K16: alarm / certified-block TPR on test attack rows when every attacking source keeps <= k of its
    attack flows per 240 s (greedy in time order); host context recomputed on the throttled raw flows."""
    rng = np.random.default_rng(0)
    cap = lambda r: np.sort(rng.permutation(r)[:5000])  # noqa: E731  same rows as the HOSTDAY test TPR
    unk = cap(bank.test[held])
    rows = np.sort(np.concatenate([unk] + [cap(r) for f, r in bank.test.items() if f not in (held, "Benign")]))
    is_unk = np.isin(rows, unk)
    raw = bank.kept[rows]
    V = light.iloc[raw]
    S_idx = np.flatnonzero((light.IPV4_SRC_ADDR.isin(set(V.IPV4_SRC_ADDR))
                            | light.IPV4_DST_ADDR.isin(set(V.IPV4_DST_ADDR))).to_numpy())
    S = light.iloc[S_idx].reset_index(drop=True)
    tmask = np.isin(S_idx, raw)
    fam = np.asarray(bank.host.family)[S_idx]
    attack = (fam != "Benign") & S.IPV4_SRC_ADDR.isin(set(V.IPV4_SRC_ADDR)).to_numpy()
    base = bank.frame(rows, ())
    t, src = S.FLOW_START_MILLISECONDS.to_numpy(), S.IPV4_SRC_ADDR.to_numpy()
    res = []
    for k in list(a.throttle) + [None]:
        keep = np.ones(len(S), bool)
        if k is not None:
            for _, idx in pd.Series(np.flatnonzero(attack)).groupby(src[attack], sort=False):
                idx = idx.to_numpy()[np.argsort(t[idx.to_numpy()], kind="stable")]
                q = deque()
                for i in idx:
                    while q and q[0] <= t[i] - sa.BUDGET_MS:
                        q.popleft()
                    if len(q) < k:
                        q.append(t[i])
                    else:
                        keep[i] = False
        kt = keep[tmask]  # scored rows that the attacker still sends
        H = sa.host_context(S[keep].reset_index(drop=True), light.iloc[:0], tmask[keep])
        if k is None:
            assert np.allclose(H, base[hostctx.COLS].to_numpy(), rtol=1e-4, atol=1e-3), "subset host context != bank"
        X = base[kt].copy()
        X[hostctx.COLS] = H
        X1 = tx(X)
        r = {"k": "inf" if k is None else k, "kept_attack_share": float(keep[attack].mean()), "rows": int(kt.sum())}
        for n, (f, b) in dets.items():
            s, lb = f(X1), b(X1)
            for part, msk in (("unk", is_unk[kt]), ("known", ~is_unk[kt])):
                r[f"{n}_tpr_{part}"] = float((s[msk] > tau[n]).mean()) if msk.any() else np.nan
                r[f"{n}_btpr_{part}"] = float((lb[msk] > tau[n]).mean()) if msk.any() else np.nan
            r[f"{n}_tpr_macro"] = np.nanmean([r[f"{n}_tpr_unk"], r[f"{n}_tpr_known"]])
            r[f"{n}_btpr_macro"] = np.nanmean([r[f"{n}_btpr_unk"], r[f"{n}_btpr_known"]])
        res.append(r)
        print(json.dumps({q: round(v, 4) if isinstance(v, float) else v for q, v in r.items()}), flush=True)
    pd.DataFrame(res).to_csv(sa.OUT / f"gate_throttle_{a.dataset}{'_' + a.tag if a.tag else ''}.csv", index=False)


if __name__ == "__main__":
    main()
