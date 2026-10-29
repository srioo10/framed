"""Per-host conformal CUSUM kill test (PREREG §9, per-host conformal CUSUM row). No training: re-reads a score bank.

Stage 1 (--meta): load + clean nf_cic once and save, per cleaned row, the source-host code, start time and family.
Stage 2: per held-out family, evaluate per-flow (P), EWMA (E), window mean (W), conformal CUSUM (A) and its Ville
version (A-V) on host-day streams with host-disjoint calibration, at a 1% benign host-day alarm budget.
Usage: .venv/bin/python scripts/host_stream.py --meta
       .venv/bin/python scripts/host_stream.py --variant none --member mlp_ce --out results/host_stream/nf_cic_none
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bank_report as br  # noqa: E402
import score_bank as sb  # noqa: E402

META = ROOT / "data" / "processed" / "nf_cic_hostmeta.npz"
DAY_MS = 86_400_000
UNIT_MS = {"day": DAY_MS, "hour": 3_600_000}
MUL = 10**7  # key = host * MUL + time unit index (hours since epoch < 10^7)
BUDGET, LAM, WIN, KAPPAS, ALPHA_V = 0.01, 0.1, 32, np.array([0.05, 0.1, 0.2, 0.5]), 0.01
METHODS = ["P", "E", "W", "A"]
MCDE_COLS = ["mcde_sup", "mcde_dist", "mcde_mode"]


def build_meta(ds="nf_cic"):
    from nids import clean, load
    raw = load.LOADERS[ds]()
    ips, src = np.unique(raw["IPV4_SRC_ADDR"].astype(str).to_numpy().astype("U"), return_inverse=True)
    t = raw["FLOW_START_MILLISECONDS"].to_numpy(np.int64)
    df, rep = clean.clean(raw, ds)
    k = rep["kept_rows"]
    meta = ROOT / "data" / "processed" / f"{ds}_hostmeta.npz"
    np.savez_compressed(meta, ips=ips, src=src[k].astype(np.int32), t=t[k],
                        fam=df["family"].astype(str).to_numpy().astype("U"))
    print(f"saved {meta}: {len(k)} rows, {len(ips)} source hosts")


def host_split(ips, seed=0):
    """True = calibration host (seeded hash of the IP string; 50/50)."""
    return np.array([hashlib.md5(f"{seed}:{ip}".encode()).digest()[0] % 2 == 0 for ip in ips])


def stats(key, t, s, ref):
    """Statistic of every method for every flow. Streams = key groups (host-day) in time order; ref = benign
    reference scores (rank role, calibration hosts) for the conformal p-value."""
    o = np.lexsort((t, key))
    k, x = key[o], s[o].astype(np.float64)
    ref = np.sort(ref)
    u = (1 + len(ref) - np.searchsorted(ref, x, side="left")) / (len(ref) + 1)  # #{ref >= s}
    q = -np.log(u)
    logg = np.log(np.mean(KAPPAS[:, None] * u[None, :] ** (KAPPAS[:, None] - 1), axis=0))
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    # CUSUM C_t = S_t - min(0, min_{j<=t} S_j), S = within-stream cumulative sum of log g
    S = np.cumsum(logg)
    S = S - np.r_[0, S][first][gid]
    A = S - np.minimum(0, pd.Series(S).groupby(gid).cummin().to_numpy())
    # window mean of q over the last WIN flows of the stream
    cq = np.r_[0, np.cumsum(q)]
    lo = np.maximum(np.arange(len(k)) - WIN + 1, first[gid])
    Wm = (cq[np.arange(len(k)) + 1] - cq[lo]) / (np.arange(len(k)) + 1 - lo)
    # EWMA of q, reset per stream, started at the benign mean E[q] = 1
    E = np.empty(len(k))
    e = 1.0
    for i, (st, qi) in enumerate(zip(start.tolist(), q.tolist())):
        e = LAM * qi + (1 - LAM) * (1.0 if st else e)
        E[i] = e
    out = {"P": x, "E": E, "W": Wm, "A": A, "AV": (A >= np.log(2 / ALPHA_V)) | (u <= ALPHA_V / 2)}
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return {m: v[inv] for m, v in out.items()}


def day_max(key, v):
    return pd.Series(v.astype(np.float64)).groupby(key).max()


def threshold(m):
    """Smallest h with share(per-unit max > h) <= BUDGET."""
    m = np.sort(m.to_numpy())[::-1]
    j = int(np.floor(BUDGET * len(m)))
    return m[j] if j < len(m) else -np.inf


def fold(part, s, meta, cal_host, held, unit="day"):
    src, t, fam_all = meta["src"], meta["t"], meta["fam"]
    rid = part.row_id.to_numpy()
    h = src[rid]
    U = UNIT_MS[unit]
    key = h.astype(np.int64) * MUL + t[rid] // DAY_MS  # stream: host-day (state resets each UTC day)
    ukey = h.astype(np.int64) * MUL + t[rid] // U  # budget / detection unit: host-day or host-hour
    role = part.role.to_numpy()
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(src[fam_all != "Benign"]))
    ch = cal_host[h] & benign_host[h]
    ref = s[(role == "rank") & ch]
    # calibration streams: cal-role rows of calibration hosts
    m_cal = (role == "cal") & ch
    st_cal = stats(key[m_cal], t[rid][m_cal], s[m_cal], ref)
    th = {m: threshold(day_max(ukey[m_cal], st_cal[m])) for m in METHODS}
    # test streams: all test rows; benign host-days of test hosts and attacker host-days of the held-out family
    m_te = np.isin(role, ["test_benign", "test_known", "test_unknown", "test_cover"])
    st = stats(key[m_te], t[rid][m_te], s[m_te], ref)
    k_te, unk = ukey[m_te], role[m_te] == "test_unknown"
    att = unk | (role[m_te] == "test_cover")  # ADV cover flows from the attacker count for its host-day
    bt = (role[m_te] == "test_benign") & ~cal_host[h[m_te]] & benign_host[h[m_te]]
    res = {"fold": held, "n_ref": int(len(ref)), "n_cal_days": int(len(np.unique(ukey[m_cal])))}
    bkeys = np.unique(k_te[bt])
    res["n_test_benign_days"] = int(len(bkeys))
    akeys = np.unique(k_te[unk])
    ahost = akeys // MUL
    res["n_attacker_hosts"], res["n_attacker_days"] = int(len(np.unique(ahost))), int(len(akeys))
    # clean history: no other attack family from that host on that day or the day before
    other = fam_all != "Benign"
    other &= fam_all != held
    okeys = np.unique(src[other].astype(np.int64) * MUL + t[other] // U)
    clean_day = ~(np.isin(akeys, okeys) | np.isin(akeys - 1, okeys))
    res["n_clean_days"] = int(clean_day.sum())
    det = {}
    for m in METHODS + ["AV"]:
        v = st[m].astype(np.float64)
        alarm = v > th[m] if m != "AV" else v > 0
        fp = pd.Series(alarm[bt]).groupby(k_te[bt]).any()
        res[f"fpr_{m}"] = float(fp.mean())
        hit = pd.Series(alarm[att]).groupby(k_te[att]).any().reindex(akeys).to_numpy()
        det[m] = hit
        res[f"tpr_{m}"] = float(hit.mean())
        res[f"tpr_clean_{m}"] = float(hit[clean_day].mean()) if clean_day.any() else float("nan")
        res[f"flowtpr_{m}"] = float(alarm[unk].mean())
        if m == "AV":
            res["flowfpr_AV"] = float(alarm[bt].mean())
    flows = pd.DataFrame({"fold": held, "row_id": rid[m_te][unk], **{m: (st[m][unk] > th[m]) if m != "AV" else st[m][unk]
                                                                   for m in METHODS + ["AV"]}})
    return res, pd.DataFrame({"fold": held, "host": ahost, "clean": clean_day, **{m: det[m] for m in det}}), flows


def boot(days, a, b, reps=2000, seed=0):
    """95% CI of macro (over families) TPR difference a - b, resampling attacker hosts within each family."""
    rng = np.random.default_rng(seed)
    per = []
    for _, g in days.groupby("fold"):
        hs = g.groupby("host")[[a, b]].agg(["sum", "count"])
        per.append((hs[(a, "sum")].to_numpy(), hs[(b, "sum")].to_numpy(), hs[(a, "count")].to_numpy()))
    macro = []
    for _ in range(reps):
        d = []
        for sa, sb_, n in per:
            i = rng.integers(0, len(n), len(n))
            d.append((sa[i].sum() - sb_[i].sum()) / n[i].sum())
        macro.append(np.mean(d))
    return np.percentile(macro, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--ds", default="nf_cic")
    ap.add_argument("--variant", default="none")
    ap.add_argument("--member", default="mlp_ce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unit", default="day", choices=list(UNIT_MS))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.meta:
        return build_meta(a.ds)
    z = np.load(ROOT / "data" / "processed" / f"{a.ds}_hostmeta.npz", allow_pickle=False)
    meta = {"src": z["src"], "t": z["t"], "fam": z["fam"]}
    cal_host = host_split(z["ips"], a.seed)
    run = sb.OUT / f"{a.ds}_{a.variant}_cap40000_s{a.seed}"
    rows, days, flows = [], [], []
    for p in sorted(run.glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        cols = MCDE_COLS if a.member == "mcde" else [a.member]
        part = sb.read_part(p, ["row_id", "role", "family"] + cols)
        assert (meta["fam"][part.row_id.to_numpy()] == part.family.to_numpy()).all(), "meta rows != bank rows"
        if a.member == "mcde":
            s = br.mcde_score(part[part.role == "rank"], part, 0.01 if a.ds == "unsw_raw" else 0.001)
        else:
            s = part[a.member].to_numpy(np.float64)
        r, d, fl = fold(part, s, meta, cal_host, held, a.unit)
        rows.append(r)
        days.append(d)
        flows.append(fl)
        print(json.dumps(r), flush=True)
    res, days = pd.DataFrame(rows).set_index("fold"), pd.concat(days, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(f"{a.out}.json", orient="index", indent=1)
    days.to_csv(f"{a.out}_days.csv", index=False)
    pd.concat(flows, ignore_index=True).to_parquet(f"{a.out}_flows.parquet", index=False)
    L = [f"# Per-host conformal CUSUM — {a.ds} {a.variant}, {a.member}, seed {a.seed}\n",
         "Host-day TPR (share of attacker host-days of the held-out family with ≥ 1 alarm on a held-out flow) at a 1% "
         "benign host-day alarm budget calibrated on disjoint hosts. Clean = clean-history subset.\n",
         "| family | hosts | host-days | clean | " + " | ".join(METHODS) + " | A clean | P clean | A-V | flow TPR A / P |",
         "|---|---|---|---|" + "---|" * (len(METHODS) + 4)]
    for k, r in res.iterrows():
        L.append(f"| {k} | {r.n_attacker_hosts} | {r.n_attacker_days} | {r.n_clean_days} | " +
                 " | ".join(f"{r[f'tpr_{m}']:.3f}" for m in METHODS) +
                 f" | {r.tpr_clean_A:.3f} | {r.tpr_clean_P:.3f} | {r.tpr_AV:.3f} | {r.flowtpr_A:.3f} / {r.flowtpr_P:.3f} |")
    mac = {m: res[f"tpr_{m}"].mean() for m in METHODS}
    L.append("\nmacro: " + ", ".join(f"{m} {v:.3f}" for m, v in mac.items()))
    L.append("realised test benign host-day alarm rate: " +
             ", ".join(f"{m} {res[f'fpr_{m}'].mean():.4f}" for m in METHODS + ["AV"]) +
             f"; A-V per-flow benign alarm rate {res.flowfpr_AV.mean():.4f}")
    ew = max(["E", "W"], key=lambda m: mac[m])
    for b in ["P", ew]:
        lo, hi = boot(days, "A", b)
        L.append(f"A − {b}: macro {mac['A'] - mac[b]:+.3f} [95% host-bootstrap CI {lo:+.3f}, {hi:+.3f}]")
    # gates (PREREG): against P and against the stronger of E/W
    for b in ["P", ew]:
        g1 = mac["A"] - mac[b] >= 0.10
        g2 = all(res.loc[f, "tpr_clean_A"] - res.loc[f, f"tpr_clean_{b}"] >= 0.10 for f in ("Bot", "DoS") if f in res.index)
        g3 = all(res[f"tpr_A"] - res[f"tpr_{b}"] >= -0.05)
        g4 = res["fpr_A"].mean() <= 0.015
        L.append(f"gates vs {b}: G1 {'PASS' if g1 else 'FAIL'}, G2 {'PASS' if g2 else 'FAIL'}, "
                 f"G3 {'PASS' if g3 else 'FAIL'}, G4 {'PASS' if g4 else 'FAIL'}")
    under = [k for k, r in res.iterrows() if r.n_attacker_hosts < 5]
    L.append(f"underpowered (< 5 attacker hosts): {under}")
    text = "\n".join(L)
    print(text)
    Path(f"{a.out}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
