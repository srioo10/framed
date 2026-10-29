"""Model selection / confirmation with the fixed referee (PREREG §9 SEL / §9 CONF). Reads candidate parts (tag "cand")
joined to the base bank parts; no training.

Flow TPR at a matched 1% benign flow FPR (threshold = 99th percentile of cal-role benign flows of calibration hosts),
macro over held-out families; host-day TPR (host_stream.fold) with a powered flag (>= 10 attacker host-days in the
test split); 95% bootstrap CIs of differences resampling attacker hosts within (seed, family).
Usage: .venv/bin/python scripts/model_eval.py dev            (seed 1, selection)
       .venv/bin/python scripts/model_eval.py confirm --arm tabm --method P   (seeds 2-4)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import discover  # noqa: E402
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import load  # noqa: E402

DATASETS = ("nf_ugr16", "nf_cic", "nf_ctu13", "nf_iot23", "nf_ton", "nf_bot")
CANDS = ["tabm", "tabm_min", "tabm_raw", "ple_ce", "lxgb_hc", "tabm_kd", "tabm_kd_min"]
BASE = ["rf", "xgb", "kitnet", "mlp_sn_gda"]
TAG, REPS, POWER, MARGIN = "cand", 2000, 10, 0.05
MIN_UNK, MIN_BEN_FLOWS, MIN_BEN_HOSTS = 100, 1000, 20  # §9 SEL amendment 2: data-only eligibility
OUT = ROOT / "results" / "model_eval"


def flow_threshold(s, cal_benign):
    return np.quantile(s[cal_benign], 0.99)


def roc_detect(s, benign, fpr=0.01):
    """Per-flow detection weight at exactly `fpr` on the benign flows: 1 above the threshold, a shared fraction on
    ties at it (randomised-threshold ROC point), 0 below."""
    q = np.quantile(s[benign], 1 - fpr)
    above, tie = (s[benign] > q).mean(), (s[benign] == q).mean()
    return (s > q) + (s == q) * (max(fpr - above, 0) / tie if tie > 0 else 0)


def per_host(ds, seed, dets):
    """Long table, one row per (seed, family, attacker host, detector): flows, flows detected, host-days, host-days
    hit (P, A); plus per (seed, family, detector) flow FPR and host-day FPR."""
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    meta = {"src": z["src"], "t": z["t"], "fam": z["fam"]}
    cal_host = hs.host_split(z["ips"], seed)
    att = z["fam"] != "Benign"
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(z["src"][att]))
    nfam = pd.Series(z["fam"][att]).groupby(z["src"][att]).nunique()
    fresh = np.zeros(len(cal_host), bool)
    fresh[nfam.index[nfam == 1]] = True  # attacker hosts that run exactly one family
    both = None
    if ds == "nf_ugr16":
        w = pd.read_parquet(load.CACHE / f"{ds}.parquet", columns=["family", "src"]).astype(str)
        both = set(w.family[w.src == "july_week5"]) & set(w.family[w.src.str.startswith("august")])
    run = sb.OUT / f"{ds}_host_cap40000_s{seed}"
    H, F = [], []
    for p in sorted((sb.OUT / f"{run.name}_{TAG}").glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        if both is not None and held not in both:
            continue
        cn = pq.read_schema(p).names
        base = sb.read_part(run / p.parent.name / "part.parquet",
                            ["row_id", "role", "family"] + [b for b in BASE if b not in cn])
        part = base.merge(pd.read_parquet(p, columns=["row_id", "role"] + [c for c in dets if c in cn]),
                          on=["row_id", "role"], how="inner", validate="one_to_one")
        assert len(part) == len(base), f"{ds} {held}: candidate rows != base rows"
        if ds == "nf_cic":  # T0 relabel, as discover.hits
            drop, r = np.load(discover.RELABEL)["drop"], part.row_id.to_numpy()
            part = part[~drop[r]].reset_index(drop=True)
        stale = (part.role.to_numpy() == "test_unknown") & ~fresh[z["src"][part.row_id.to_numpy()]]
        part = part[~stale].reset_index(drop=True)  # unknown = only attackers never seen running another family
        role, rid = part.role.to_numpy(), part.row_id.to_numpy()
        h = z["src"][rid]
        unk = role == "test_unknown"
        if not unk.any():
            continue
        cal_b = (role == "cal") & cal_host[h] & benign_host[h]
        te_b = (role == "test_benign") & ~cal_host[h] & benign_host[h]
        akey = h.astype(np.int64) * hs.MUL + z["t"][rid] // hs.DAY_MS
        for d in dets:
            s = part[d].to_numpy(np.float64)
            det = roc_detect(s, te_b)  # §9 SEL amendment: matched 1% on test benign hosts (ROC TPR@1%FPR)
            dep = s > flow_threshold(s, cal_b)  # deployable threshold from calibration hosts, reported only
            r, days, _ = hs.fold(part, s, meta, cal_host, held)
            ak = np.unique(akey[unk])
            hd = pd.DataFrame({"host": ak // hs.MUL, "P": days["P"].astype(float), "A": days["A"].astype(float)})
            g = pd.DataFrame({"host": h[unk], "det": det[unk]}).groupby("host").det.agg(["size", "sum"])
            hh = hd.groupby("host").agg(days=("P", "size"), hit_P=("P", "sum"), hit_A=("A", "sum"))
            t = g.join(hh, how="outer").fillna(0).reset_index()
            t[["seed", "fold", "detector"]] = seed, held, d
            H.append(t.rename(columns={"size": "flows", "sum": "flows_det"}))
            F.append({"seed": seed, "fold": held, "detector": d, "flow_fpr": det[te_b].mean(),
                      "deploy_fpr": dep[te_b].mean(), "deploy_tpr": dep[unk].mean(), "unk_flows": int(unk.sum()),
                      "eligible": unk.sum() >= MIN_UNK and te_b.sum() >= MIN_BEN_FLOWS
                      and len(np.unique(h[te_b])) >= MIN_BEN_HOSTS,
                      "fpr_P": r["fpr_P"], "fpr_A": r["fpr_A"], "att_days": r["n_attacker_days"]})
    return (pd.concat(H, ignore_index=True), pd.DataFrame(F)) if H else (None, None)


def macro(t, num, den):
    """Macro over (seed, fold) of sum(num)/sum(den), then mean."""
    g = t.groupby(["seed", "fold"])[[num, den]].sum()
    return (g[num] / g[den].where(g[den] > 0)).mean()


def boot_diff(H, a, b, num, den, reps=REPS, seed=0):
    """CI of macro(a) - macro(b), resampling attacker hosts within (seed, fold)."""
    A = H[H.detector == a].set_index(["seed", "fold", "host"])[[num, den]]
    B = H[H.detector == b].set_index(["seed", "fold", "host"])[[num, den]].reindex(A.index)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(m) for m in pd.Series(range(len(A)), index=A.index).groupby(level=[0, 1]).indices.values()]
    an, ad, bn, bd = (x.to_numpy(float) for x in (A[num], A[den], B[num], B[den]))
    out = np.empty(reps)
    for r in range(reps):
        da, db = [], []
        for g in groups:
            i = g[rng.integers(0, len(g), len(g))]
            if ad[i].sum() > 0:
                da.append(an[i].sum() / ad[i].sum())
                db.append(bn[i].sum() / bd[i].sum())
        out[r] = np.mean(da) - np.mean(db)
    return np.quantile(out, [0.025, 0.975])


def dev():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        if not (sb.OUT / f"{ds}_host_cap40000_s1_{TAG}" / "manifest_outer.json").exists():
            print(f"{ds}: no candidate run yet")
            continue
        H, F = per_host(ds, 1, CANDS + BASE)
        if H is None:
            continue
        ok = F[F.eligible][["seed", "fold"]].drop_duplicates()
        print(f"{ds}: eligible families {sorted(ok.fold)}")
        for d in CANDS + BASE:
            t = H[H.detector == d].merge(ok, on=["seed", "fold"])
            if t.empty:
                continue
            rows.append({"ds": ds, "detector": d, "flow_tpr": macro(t, "flows_det", "flows"),
                         "flow_fpr": F[F.detector == d].flow_fpr.mean(),
                         "deploy_fpr": F[F.detector == d].deploy_fpr.mean(), "deploy_tpr": F[F.detector == d].deploy_tpr.mean(),
                         "day_tpr_P": macro(t, "hit_P", "days"), "day_tpr_A": macro(t, "hit_A", "days")})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "dev.csv", index=False)
    best = R[R.detector.isin(BASE)].groupby("ds").flow_tpr.max()
    R["gap"] = R.flow_tpr - R.ds.map(best)
    fl = pd.read_csv(ROOT / "results" / "engine" / "flood_detectors.csv")
    f1 = fl[fl.k == 1].set_index(["detector", "n"])

    def affected(d):
        if (d, 10) not in f1.index:
            return None
        return any(f1.loc[(d, 10), f"ben_{m}"] >= 0.10 and f1.loc[(d, 10), f"ben_{m}"] >= 5 * f1.loc[(d, 0), f"ben_{m}"]
                   for m in ("A", "P"))
    S = R[R.detector.isin(CANDS)].groupby("detector").agg(mean_gap=("gap", "mean"), datasets=("ds", "nunique"),
                                                          day_P=("day_tpr_P", "mean"), day_A=("day_tpr_A", "mean"))
    S["flood_affected"] = [affected(d) for d in S.index]
    ok = S[S.flood_affected.eq(False)]
    pick = ok.mean_gap.idxmax() if len(ok) else None
    L = ["# Development selection (PREREG §9 SEL), seed 1\n",
         "Flow TPR at a matched 1% benign flow FPR on test hosts (macro over held-out families):\n",
         R.pivot_table(index="detector", columns="ds", values="flow_tpr").round(3).to_string(),
         "\nHost-day TPR, better of P / A:\n",
         R.assign(day=R[["day_tpr_P", "day_tpr_A"]].max(axis=1)).pivot_table(index="detector", columns="ds",
                                                                         values="day").round(3).to_string(),
         "\nDeployable threshold (99th pct of calibration hosts): realised test flow FPR\n",
         R.pivot_table(index="detector", columns="ds", values="deploy_fpr").round(4).to_string(),
         "\nCandidates:\n", S.round(4).to_string(),
         f"\n**Selected: {pick}**, host-day method "
         f"{'P' if pick and S.loc[pick, 'day_P'] >= S.loc[pick, 'day_A'] else 'A'}\n"]
    (OUT / "dev_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def confirm(arm, method, seeds=(2, 3, 4)):
    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"# Confirmation (PREREG §9 CONF): {arm}, host-day method {method}, seeds {list(seeds)}\n"]
    V = []
    for ds in DATASETS:
        parts = [per_host(ds, s, [arm] + BASE) for s in seeds
                 if (sb.OUT / f"{ds}_host_cap40000_s{s}_{TAG}" / "manifest_outer.json").exists()]
        parts = [p for p in parts if p[0] is not None]
        if not parts:
            continue
        H = pd.concat([p[0] for p in parts], ignore_index=True)
        F = pd.concat([p[1] for p in parts], ignore_index=True)
        pw = F[F.detector == arm].set_index(["seed", "fold"]).att_days >= POWER
        for b in BASE:
            lo, hi = boot_diff(H, arm, b, "flows_det", "flows")
            row = {"ds": ds, "baseline": b, "flow_diff_lo": lo, "flow_diff_hi": hi,
                   "flow_noninferior": lo >= -MARGIN, "flow_beats": lo > 0}
            Hp = H.set_index(["seed", "fold"]).loc[pw[pw].index].reset_index() if pw.any() else None
            if Hp is not None and len(Hp):
                dlo, dhi = boot_diff(Hp, arm, b, f"hit_{method}", "days")
                row |= {"day_diff_lo": dlo, "day_diff_hi": dhi, "day_noninferior": dlo >= -MARGIN}
            V.append(row)
    V = pd.DataFrame(V)
    per_ds = V.groupby("ds").flow_noninferior.all()
    L += [V.round(4).to_string(index=False), f"\nDatasets non-inferior to all four baselines (flow): "
          f"{int(per_ds.sum())} of {len(per_ds)} (needs >= 4)\n"]
    V.to_csv(OUT / "confirm.csv", index=False)
    (OUT / "confirm_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "confirm"])
    ap.add_argument("--arm")
    ap.add_argument("--method", choices=["P", "A"])
    a = ap.parse_args()
    dev() if a.mode == "dev" else confirm(a.arm, a.method)


if __name__ == "__main__":
    main()
