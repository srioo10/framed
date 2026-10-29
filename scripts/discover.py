"""Discovery harness and blind-spot kill test (PREREG §9 DISC). Reads score banks; no training.

Per (dataset, seed, outer fold, detector): host_stream.fold thresholds P (host-day max score) and A (per-host conformal
CUSUM) at the 1% benign host-day budget. An attacker host-day is all-missed when no (detector, method) alarms.
B1: all-missed share (macro over families, mean over seeds) >= 0.15 on >= 2 datasets.
B2: one fixed descriptor separates all-missed from caught host-days, host-grouped 5-fold CV AUC >= 0.80, on >= 2 datasets.
Usage: .venv/bin/python scripts/discover.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import load  # noqa: E402

DATASETS = ("nf_ctu13", "nf_iot23", "nf_cic")
DETECTORS = ("mlp_sn_gda", "rf", "xgb", "iforest", "kitnet")
DESC = ("rate", "port_entropy", "iat_cv", "fanout", "med_bytes", "enc_share", "med_dur")
ENC = [443, 8443, 993, 995, 22]
OUT = ROOT / "results" / "discover"
RELABEL = ROOT / "data" / "processed" / "nf_cic_relabel.npz"


def hits(ds, seed, tag=""):
    """tag: suffix of an ADV perturbed bank (e.g. "_dilate10"); its adv_meta.npz replaces times and adds cover flows."""
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    meta = {"src": z["src"], "t": z["t"], "fam": z["fam"]}
    run = sb.OUT / f"{ds}_host_cap40000_s{seed}{tag}"
    if (run / "adv_meta.npz").exists():
        a = np.load(run / "adv_meta.npz", allow_pickle=False)
        assert len(a["t_kept"]) == len(z["t"]), "perturbed bank and host meta disagree on kept rows"
        meta = {"src": np.r_[z["src"], np.searchsorted(z["ips"], a["cover_src"])].astype(np.int32),
                "t": np.r_[a["t_kept"], a["cover_t"]].astype(np.int64), "fam": np.r_[z["fam"], a["cover_fam"]]}
    cal_host = hs.host_split(z["ips"], seed)
    rows, fprs = [], []
    for p in sorted(run.glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        dets = [d for d in DETECTORS if d in pq.read_schema(p).names]
        part = sb.read_part(p, ["row_id", "role", "family"] + dets)
        if ds == "nf_cic":
            drop, r = np.load(RELABEL)["drop"], part.row_id.to_numpy()
            part = part[~(r < len(drop)) | ~drop[np.minimum(r, len(drop) - 1)]].reset_index(drop=True)  # ADV cover rows kept
        rid = part.row_id.to_numpy()
        unk = part.role.to_numpy() == "test_unknown"
        if not unk.any():
            continue
        akeys = np.unique(meta["src"][rid[unk]].astype(np.int64) * hs.MUL + meta["t"][rid[unk]] // hs.DAY_MS)
        cols = {}
        for d in dets:
            r, days, _ = hs.fold(part, part[d].to_numpy(np.float64), meta, cal_host, held)
            for m in ("P", "A"):
                cols[f"{d}_{m}"] = days[m].to_numpy(bool)
                fprs.append({"ds": ds, "seed": seed, "fold": held, "detector": d, "method": m,
                             "fpr": r[f"fpr_{m}"], "tpr": r[f"tpr_{m}"], "n_test_benign_days": r["n_test_benign_days"]})
        f = pd.DataFrame({"ds": ds, "seed": seed, "fold": held, "key": akeys, "host": z["ips"][akeys // hs.MUL],
                          **cols})
        f["all_missed"] = ~f[list(cols)].any(axis=1)
        f["n_det"] = len(dets)
        rows.append(f)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fprs)


def descriptors(ds, keys):
    """Fixed descriptors of each attacker host-day, from that host's raw flows of the held-out family that day."""
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    raw = pd.read_parquet(load.CACHE / f"{ds}.parquet", columns=[
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "FLOW_START_MILLISECONDS", "IN_BYTES",
        "FLOW_DURATION_MILLISECONDS", "family"])
    raw = raw[raw.family.astype(str).isin(keys.fold.unique())]  # only held-out attack families are described
    code = np.searchsorted(z["ips"], raw["IPV4_SRC_ADDR"].astype(str).to_numpy().astype("U"))
    raw["key"] = code.astype(np.int64) * hs.MUL + raw["FLOW_START_MILLISECONDS"].to_numpy(np.int64) // hs.DAY_MS
    out = []
    for (key, fold), _ in keys.groupby(["key", "fold"]):
        g = raw[(raw.key.to_numpy() == key) & (raw.family.astype(str).to_numpy() == fold)]
        t = np.sort(g.FLOW_START_MILLISECONDS.to_numpy(np.int64)) / 1000
        iat = np.diff(t)
        pc = g.L4_DST_PORT.value_counts(normalize=True).to_numpy()
        out.append({"key": key, "fold": fold, "rate": len(g) / max(t[-1] - t[0], 1.0) if len(t) else 0.0,
                    "port_entropy": float(-(pc * np.log(pc)).sum()),
                    "iat_cv": float(iat.std() / iat.mean()) if len(iat) > 1 and iat.mean() > 0 else 0.0,
                    "fanout": g.IPV4_DST_ADDR.nunique(), "med_bytes": float(g.IN_BYTES.median()),
                    "enc_share": float(g.L4_DST_PORT.isin(ENC).mean()),
                    "med_dur": float(g.FLOW_DURATION_MILLISECONDS.median())})
    return pd.DataFrame(out).fillna(0.0)


def cv_auc(x, y, groups):
    """Host-grouped 5-fold CV AUC of one descriptor; direction set on each training fold, AUC pooled over test folds."""
    s = np.empty(len(x))
    for tr, te in GroupKFold(min(5, len(np.unique(groups)))).split(x, y, groups):
        sign = 1 if len(np.unique(y[tr])) < 2 or roc_auc_score(y[tr], x[tr]) >= 0.5 else -1
        s[te] = sign * x[te]
    return roc_auc_score(y, s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L, b1, b2 = ["# Discovery harness and blind-spot kill test (PREREG §9 DISC)\n"], {}, {}
    for ds in DATASETS:
        parts = [hits(ds, s) for s in (1, 2, 3, 4) if (sb.OUT / f"{ds}_host_cap40000_s{s}").exists()]
        H, F = pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])
        D = descriptors(ds, H[["key", "fold"]].drop_duplicates())
        H = H.merge(D, on=["key", "fold"], how="left")
        H.to_parquet(OUT / f"{ds}_miss_table.parquet", index=False)
        F.to_csv(OUT / f"{ds}_detectors.csv", index=False)
        share = H.groupby(["seed", "fold"]).all_missed.mean().groupby("seed").mean().mean()
        b1[ds] = share
        L += [f"## {ds}\n", f"Seeds {sorted(H.seed.unique())}; detectors per fold: {sorted(H.n_det.unique())}; "
              f"attacker host-days (summed over seeds) {len(H)}; all-missed {int(H.all_missed.sum())}\n",
              "| family | attacker host-days | all-missed share |", "|---|---|---|"]
        L += [f"| {f} | {len(g)} | {g.all_missed.mean():.3f} |" for f, g in H.groupby("fold")]
        L.append(f"\n**All-missed share (macro over families, mean over seeds): {share:.3f}** (B1 needs >= 0.15)\n")
        tab = F.groupby(["detector", "method"])[["fpr", "tpr"]].mean()
        L += ["| detector | method | realised benign host-day FPR | macro host-day TPR |", "|---|---|---|---|"]
        L += [f"| {d} | {m} | {r.fpr:.4f} | {r.tpr:.3f} |" for (d, m), r in tab.iterrows()]
        y = H.all_missed.to_numpy(int)
        nm, nc = int(y.sum()), int(len(y) - y.sum())
        if nm < 5 or nc < 5:
            b2[ds] = {}
            L.append(f"\nB2 not evaluable: {nm} all-missed, {nc} caught host-days (rule: B2 fails for this dataset)\n")
            continue
        aucs = {d: cv_auc(H[d].to_numpy(float), y, H.host.to_numpy()) for d in DESC}
        b2[ds] = aucs
        L += ["\n| descriptor | host-grouped CV AUC (all-missed vs caught) |", "|---|---|"]
        L += [f"| {d} | {a:.3f} |" for d, a in aucs.items()]
        L.append("")
    pass1 = sum(v >= 0.15 for v in b1.values()) >= 2
    good = {d: [ds for ds in b2 if b2[ds].get(d, 0) >= 0.80] for d in DESC}
    pass2 = any(len(v) >= 2 for v in good.values())
    L.append(f"**B1 {'PASS' if pass1 else 'FAIL'}**: all-missed share " +
             ", ".join(f"{ds} {v:.3f}" for ds, v in b1.items()))
    L.append(f"\n**B2 {'PASS' if pass2 else 'FAIL'}**: descriptors with AUC >= 0.80 per dataset: " +
             json.dumps({d: v for d, v in good.items() if v}))
    L.append(f"\n**DISC verdict: {'proceed to step 3' if pass1 and pass2 else 'stop: no algorithm claim'}**")
    text = "\n".join(L)
    print(text)
    (OUT / "report.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
