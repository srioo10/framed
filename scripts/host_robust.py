"""Robustness of the host-level budget (paper §6): where benign alarms fall and how dependent benign streams are.

Per fold of nf_cic (host variant, seeds 1-4): benign test host-days that alarm under A, alarms per benign host
(share from the top 10% of alarming hosts), lag-1 autocorrelation of the conformal p-value within benign host-day
streams, and the realised per-flow alarm rate of the Ville rule (target alpha = 0.01 if flows were exchangeable).
Usage: .venv/bin/python scripts/host_robust.py --member mlp_sn_gda --out results/onepass/robust_nf_cic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nf_cic")
    ap.add_argument("--member", default="mlp_ce")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(ROOT / "data" / "processed" / f"{a.ds}_hostmeta.npz")
    src, t, fam = z["src"], z["t"], z["fam"]
    benign_host = ~np.isin(np.arange(len(z["ips"])), np.unique(src[fam != "Benign"]))
    rows, per_host = [], []
    for seed in a.seeds:
        cal_host = hs.host_split(z["ips"], seed)
        for p in sorted((sb.OUT / f"{a.ds}_host_cap40000_s{seed}").glob("config=outer_*/part.parquet")):
            part = sb.read_part(p, ["row_id", "role", a.member])
            rid, role, s = part.row_id.to_numpy(), part.role.to_numpy(), part[a.member].to_numpy(np.float64)
            h = src[rid]
            key = h.astype(np.int64) * hs.MUL + t[rid] // hs.DAY_MS
            ch = cal_host[h] & benign_host[h]
            ref = s[(role == "rank") & ch]
            mc = (role == "cal") & ch
            th = hs.threshold(hs.day_max(key[mc], hs.stats(key[mc], t[rid][mc], s[mc], ref)["A"]))
            bt = (role == "test_benign") & ~cal_host[h] & benign_host[h]
            st = hs.stats(key[bt], t[rid][bt], s[bt], ref)
            alarm_day = pd.Series(st["A"] > th).groupby(key[bt]).any()
            hosts = alarm_day.index.to_numpy() // hs.MUL
            cnt = pd.Series(alarm_day.to_numpy()).groupby(hosts).sum()
            # conformal p-values of benign test flows, in stream order, and their lag-1 autocorrelation
            o = np.lexsort((t[rid][bt], key[bt]))
            r = np.sort(ref)
            u = (1 + len(r) - np.searchsorted(r, s[bt][o], side="left")) / (len(r) + 1)
            same = key[bt][o][1:] == key[bt][o][:-1]
            ac = float(np.corrcoef(u[1:][same], u[:-1][same])[0, 1]) if same.sum() > 2 else float("nan")
            alarmed = cnt[cnt > 0].sort_values(ascending=False)
            top = int(np.ceil(0.1 * len(alarmed))) if len(alarmed) else 0
            rows.append({"seed": seed, "fold": p.parent.name.removeprefix("config=outer_"),
                         "benign_test_host_days": int(len(alarm_day)), "alarm_host_days": int(alarm_day.sum()),
                         "benign_hosts": int(len(cnt)), "alarming_hosts": int((cnt > 0).sum()),
                         "max_alarm_days_one_host": int(cnt.max()),
                         "top10pct_host_share": float(alarmed.iloc[:top].sum() / alarmed.sum()) if top else float("nan"),
                         "lag1_autocorr_u": ac, "ville_flow_fpr": float(st["AV"].mean())})
            per_host.append(cnt)
    r = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(f"{a.out}_{a.member}.csv", index=False)
    allc = pd.concat(per_host)
    text = "\n".join([
        f"# Host-level budget robustness — {a.ds}, {a.member}, seeds {a.seeds}\n",
        f"benign test host-days per fold (mean): {r.benign_test_host_days.mean():.0f}; alarming: {r.alarm_host_days.mean():.1f} "
        f"({(r.alarm_host_days / r.benign_test_host_days).mean():.4f})",
        f"benign test hosts per fold (mean): {r.benign_hosts.mean():.0f}; hosts with >= 1 alarm: {r.alarming_hosts.mean():.1f}",
        f"max alarming host-days on one host (max over folds): {r.max_alarm_days_one_host.max()}",
        f"share of alarm host-days from the top 10% alarming hosts (mean): {r.top10pct_host_share.mean():.3f}",
        f"alarm host-days per alarming host, distribution: {allc[allc > 0].value_counts().sort_index().to_dict()}",
        f"lag-1 autocorrelation of benign conformal p-values within host-day streams (mean): {r.lag1_autocorr_u.mean():.3f}",
        f"Ville rule realised per-flow benign alarm rate: {r.ville_flow_fpr.mean():.4f} (exchangeable target <= 0.01)"])
    print(text)
    Path(f"{a.out}_{a.member}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
