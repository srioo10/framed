"""Few-host calibration kill test (PREREG §9, CAL rows). No training: re-reads score banks.

For each (dataset, seed, outer fold) and each draw of m benign calibration hosts, the per-host CUSUM of
X = log g(u) is thresholded four ways on identical test streams:
  M  Markov-Lundberg anytime threshold c_t = [log(K/alpha) + log(t(t+1))] / theta*  (new)
  E  empirical quantile of the m hosts' host-day maxima (current OnePass rule)
  V  Ville rule AV
  B  moving-block bootstrap of the m hosts' X sequences
Usage: taskset -c 0-11 .venv/bin/python scripts/calib.py run --jobs 12
       .venv/bin/python scripts/calib.py report
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402

NB, BLOCK, NBOOT, MS, DRAWS, ALPHA = 8, 32, 2000, (5, 20, 50), 10, 0.01
OUT = ROOT / "results" / "calib"
RELABEL = ROOT / "data" / "processed" / "nf_cic_relabel.npz"


def logg(u):
    return np.log(np.mean(hs.KAPPAS[:, None] * u[None, :] ** (hs.KAPPAS[:, None] - 1), axis=0))


def pvals(x, ref):
    ref = np.sort(ref)
    return (1 + len(ref) - np.searchsorted(ref, x, side="left")) / (len(ref) + 1)


def cusum(X, start):
    """C_t = max(0, C_{t-1} + X_t) per stream and the 1-based flow index t; rows already in stream order."""
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    S = np.cumsum(X)
    S = S - np.r_[0.0, S][first][gid]
    C = S - np.minimum(0, pd.Series(S).groupby(gid).cummin().to_numpy())
    return C, np.arange(len(X)) - first[gid] + 1


def fit_markov(X, start):
    """theta*, K of the bin-state Markov-additive model of X (rows in stream order; start marks stream starts).
    State = pooled quantile bin of X; within a state, increments are drawn from that bin's empirical values, so
    the tilted matrix is P * diag(mgf_b(theta)). Returns (inf, 1) if no positive increment exists, (None, None) if
    the stationary drift is >= 0 (no root: not calibratable)."""
    if X.max() <= 0:
        return np.inf, 1.0
    edges = np.unique(np.quantile(X, np.arange(1, NB) / NB))
    b = np.searchsorted(edges, X, side="right")
    nb = len(edges) + 1
    P = np.ones((nb, nb))
    ok = ~start[1:]
    np.add.at(P, (b[:-1][ok], b[1:][ok]), 1)
    P /= P.sum(1, keepdims=True)
    vals = [X[b == j] for j in range(nb)]

    def tilted(th):
        mgf = np.array([np.exp(th * v).mean() if len(v) else 0.0 for v in vals])
        return P * mgf[None, :]

    def rho(th):
        return np.max(np.abs(np.linalg.eigvals(tilted(th))))
    w, V = np.linalg.eig(P.T)
    pi = np.abs(np.real(V[:, np.argmax(np.real(w))]))
    pi /= pi.sum()
    if pi @ np.array([v.mean() if len(v) else 0.0 for v in vals]) >= 0:
        return None, None
    lo, hi = 0.0, 1.0
    while rho(hi) < 1:
        lo, hi = hi, hi * 2
        if hi > 1e3:
            return np.inf, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rho(mid) < 1 else (lo, mid)
    th = hi
    w, V = np.linalg.eig(tilted(th))
    h = np.abs(np.real(V[:, np.argmax(np.real(w))]))
    return th, float(h.max() / max(h.min(), 1e-300))


def m_threshold(t, th, K):
    return (np.log(K / ALPHA) + np.log(t * (t + 1.0))) / th


def boot_threshold(X, start, rng):
    """99% quantile of CUSUM maxima over NBOOT synthetic host-days built from moving blocks of the m hosts' X."""
    first = np.flatnonzero(start)
    ends = np.r_[first[1:], len(X)]
    lens = ends - first
    blocks = [X[s:s + BLOCK] for a, e in zip(first, ends) for s in range(a, max(a + 1, e - BLOCK + 1))]
    L = rng.choice(lens, NBOOT)

    def draw(n):  # blocks from short host-days are shorter than BLOCK, so draw until n values exist
        out, k = [], 0
        while k < n:
            b = blocks[rng.integers(len(blocks))]
            out.append(b)
            k += len(b)
        return np.concatenate(out)[:n]
    seq = np.concatenate([draw(n) for n in L])
    st = np.zeros(len(seq), bool)
    st[np.r_[0, np.cumsum(L)[:-1]]] = True
    C = cusum(seq, st)[0]
    return float(np.quantile(np.maximum.reduceat(C, np.flatnonzero(st)), 1 - ALPHA))


def job(ds, seed, held):
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    src, t_all, fam = z["src"], z["t"], z["fam"]
    part = pd.read_parquet(sb.OUT / f"{ds}_host_cap40000_s{seed}" / f"config=outer_{held}" / "part.parquet",
                           columns=["row_id", "role", "mlp_sn_gda"])
    if ds == "nf_cic":
        part = part[~np.load(RELABEL)["drop"][part.row_id.to_numpy()]]
    rid, role, s = part.row_id.to_numpy(), part.role.to_numpy(), part.mlp_sn_gda.to_numpy(np.float64)
    h, t = src[rid], t_all[rid]
    key = h.astype(np.int64) * hs.MUL + t // hs.DAY_MS
    o = np.lexsort((t, key))  # stream order, fixed for every draw
    rid, role, s, h, t, key = rid[o], role[o], s[o], h[o], t[o], key[o]
    cal_host = hs.host_split(z["ips"], seed)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(src[fam != "Benign"]))
    ch = cal_host[h] & benign_host[h]
    te = np.isin(role, ["test_benign", "test_known", "test_unknown"])
    unk = role[te] == "test_unknown"
    bt = (role[te] == "test_benign") & ~cal_host[h[te]] & benign_host[h[te]]
    kte = key[te]
    start_te = np.r_[True, kte[1:] != kte[:-1]]

    def rates(alarm):
        fpr = pd.Series(alarm[bt]).groupby(kte[bt]).any().mean()
        tpr = pd.Series(alarm[unk]).groupby(kte[unk]).any().mean()
        return float(fpr), float(tpr)

    def streams(mask, X):
        k = key[mask]
        st = np.r_[True, k[1:] != k[:-1]]
        return X[mask], st, k
    rows = []
    # Full: current rule with every calibration host
    ref = s[(role == "rank") & ch]
    X = logg(pvals(s, ref))
    Xc, stc, kc = streams((role == "cal") & ch, X)
    th = hs.threshold(hs.day_max(kc, cusum(Xc, stc)[0]))
    C, _ = cusum(X[te], start_te)
    f, tp = rates(C > th)
    rows.append({"ds": ds, "seed": seed, "fold": held, "m": "all", "draw": 0, "method": "Full", "fpr": f, "tpr": tp,
                 "n_cal_days": int(len(np.unique(kc)))})
    pool = np.unique(h[(role == "cal") & ch])
    for m in MS:
        for d in range(DRAWS):
            rng = np.random.default_rng([seed, m, d])
            pick = rng.choice(pool, m, replace=False)
            sel = np.isin(h, pick)
            ref = s[(role == "rank") & sel]
            u = pvals(s, ref)
            X = logg(u)
            Xc, stc, kc = streams((role == "cal") & sel, X)
            C, tix = cusum(X[te], start_te)
            base = {"ds": ds, "seed": seed, "fold": held, "m": m, "draw": d, "n_cal_days": int(len(np.unique(kc)))}
            thE = hs.threshold(hs.day_max(kc, cusum(Xc, stc)[0]))
            rows.append({**base, "method": "E", **dict(zip(("fpr", "tpr"), rates(C > thE)))})
            av = (C >= np.log(2 / ALPHA)) | (u[te] <= ALPHA / 2)
            rows.append({**base, "method": "V", **dict(zip(("fpr", "tpr"), rates(av)))})
            thM, K = fit_markov(Xc, stc)
            alarm = np.ones(len(C), bool) if thM is None else C > m_threshold(tix, thM, K)
            rows.append({**base, "method": "M", "theta": thM, "K": K, **dict(zip(("fpr", "tpr"), rates(alarm)))})
            thB = boot_threshold(Xc, stc, rng)
            rows.append({**base, "method": "B", **dict(zip(("fpr", "tpr"), rates(C > thB)))})
    return rows


def folds(ds):
    return sorted(p.parent.name.removeprefix("config=outer_")
                  for p in (sb.OUT / f"{ds}_host_cap40000_s1").glob("config=outer_*/part.parquet"))


def run(jobs):
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(ds, seed, f) for ds in ("nf_cic", "nf_ton") for seed in (1, 2, 3, 4) for f in folds(ds)
            if not (OUT / f"{ds}_s{seed}_{f}.json").exists()]
    with ProcessPoolExecutor(jobs) as ex:
        for (ds, seed, f), rows in zip(todo, ex.map(job, *zip(*todo))):
            (OUT / f"{ds}_s{seed}_{f}.json").write_text(json.dumps(rows))
            print(f"{ds} s{seed} {f} done", flush=True)


def report():
    res = pd.DataFrame([r for p in sorted(OUT.glob("nf_*_s*_*.json")) for r in json.loads(p.read_text())])
    res["m"] = res.m.astype(str)
    ok, L = {}, ["# Few-host calibration (PREREG §9, CAL)\n"]
    for ds, g in res.groupby("ds"):
        # macro over folds within (seed, m, draw, method), then mean over draws and seeds
        mac = g.groupby(["m", "method", "seed", "draw"])[["fpr", "tpr"]].mean().groupby(["m", "method"]).mean()
        nr = g[g.method == "M"].theta.isna().mean() if "theta" in g else 0
        L += [f"## {ds}\n", "| m | method | benign host-day alarm rate | macro host-day TPR |", "|---|---|---|---|"]
        L += [f"| {m} | {meth} | {r.fpr:.4f} | {r.tpr:.3f} |" for (m, meth), r in mac.iterrows()]
        L.append(f"\nM not calibratable (no root) in {nr:.1%} of draws\n")
        ok[ds] = mac
    full = ok["nf_cic"].loc[("all", "Full"), "tpr"]
    rate_ok = {meth: all(ok[ds].loc[(m, meth), "fpr"] <= 0.015 for ds in ok for m in ("5", "20")) for meth in "MB"}
    tpr_ok = {meth: all(ok["nf_cic"].loc[(m, meth), "tpr"] >= full - 0.05 for m in ("5", "20")) for meth in "MB"}
    passM = rate_ok["M"] and tpr_ok["M"]
    L.append(f"**CAL primary (M) {'PASS' if passM else 'FAIL'}**: rate <= 0.015 at m = 5, 20 on both datasets "
             f"{'yes' if rate_ok['M'] else 'no'}; nf_cic TPR >= Full - 0.05 ({full - 0.05:.3f}) {'yes' if tpr_ok['M'] else 'no'}")
    if passM:
        mt = min(ok["nf_cic"].loc[(m, "M"), "tpr"] - ok["nf_cic"].loc[(m, "B"), "tpr"] for m in ("5", "20"))
        novel = not rate_ok["B"] or mt >= 0
        L.append(f"Novelty condition: {'met' if novel else 'not met: block bootstrap suffices'} "
                 f"(B rate rule {'pass' if rate_ok['B'] else 'fail'}; min M - B nf_cic TPR {mt:+.3f})")
    text = "\n".join(L)
    print(text)
    (OUT / "report.md").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    return run(a.jobs) if a.stage == "run" else report()


if __name__ == "__main__":
    main()
