"""UGR'16 forward-time false-alarm test (PREREG §9 FWD). Reads nf_ugr16 score banks; no training.

Thresholds as host_stream.fold (cal role of June calibration hosts, 1% host-day budget). Per (detector, method):
  forward FPR  = alarmed share of August benign host-days of non-calibration hosts (the claim quantity)
  time-only    = August benign host-days of calibration hosts (same hosts, 2 months later)
  host-only    = June cal-role benign host-days of non-calibration hosts (same period, other hosts)
Each host-day's alarm is averaged over folds and seeds; 95% CI by resampling hosts. Claim: CI upper <= 1.5%.
Host-day TPR per held-out family from host_stream.fold.
Usage: .venv/bin/python scripts/forward.py
"""
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

DS = "nf_ugr16"
METHODS = ("P", "A")  # PREREG §9 FWD methods
OUT = ROOT / "results" / "forward"


def alarms(part, s, meta, cal_host):
    """{split: Series host-day key -> alarmed} for each method."""
    src, t, fam = meta["src"], meta["t"], meta["fam"]
    rid = part.row_id.to_numpy()
    h, role = src[rid], part.role.to_numpy()
    key = h.astype(np.int64) * hs.MUL + t[rid] // hs.DAY_MS
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(src[fam != "Benign"]))
    ch = cal_host[h] & benign_host[h]
    ref = s[(role == "rank") & ch]
    m_cal = (role == "cal") & ch
    st_cal = hs.stats(key[m_cal], t[rid][m_cal], s[m_cal], ref)
    th = {m: hs.threshold(hs.day_max(key[m_cal], st_cal[m])) for m in METHODS}
    m_te = np.isin(role, ["test_benign", "test_known", "test_unknown"])
    m_ho = (role == "cal") & ~cal_host[h] & benign_host[h]
    out = {}
    for name, m, keep in (("forward", m_te, lambda r: (role[r] == "test_benign") & ~cal_host[h[r]] & benign_host[h[r]]),
                          ("time_only", m_te, lambda r: (role[r] == "test_benign") & cal_host[h[r]] & benign_host[h[r]]),
                          ("host_only", m_ho, lambda r: np.ones(len(r), bool))):
        r = np.flatnonzero(m)
        st = hs.stats(key[r], t[rid][r], s[r], ref)
        k = keep(r)
        for meth in METHODS:
            out[(name, meth)] = pd.Series(st[meth][k] > th[meth]).groupby(key[r][k]).any()
    return out


def ci(day_rate, reps=2000, seed=0):
    host = day_rate.index.to_numpy() // hs.MUL
    g = pd.DataFrame({"h": host, "a": day_rate.to_numpy()}).groupby("h").a.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(g), (reps, len(g)))
    b = g["sum"].to_numpy()[i].sum(1) / g["count"].to_numpy()[i].sum(1)
    return np.quantile(b, [0.025, 0.975])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data" / "processed" / f"{DS}_hostmeta.npz", allow_pickle=False)
    meta = {"src": z["src"], "t": z["t"], "fam": z["fam"]}
    w = pd.read_parquet(load.CACHE / f"{DS}.parquet", columns=["family", "src"]).astype(str)
    both = set(w.family[w.src == "july_week5"]) & set(w.family[w.src.str.startswith("august")])
    acc, tpr, seeds = {}, [], []
    for seed in (1, 2, 3, 4):
        run = sb.OUT / f"{DS}_host_cap40000_s{seed}"
        if not (run / "manifest_outer.json").exists():
            continue
        seeds.append(seed)
        cal_host = hs.host_split(z["ips"], seed)
        for p in sorted(run.glob("config=outer_*/part.parquet")):
            held = p.parent.name.removeprefix("config=outer_")
            if held not in both:  # PREREG §9 FWD: a family needs July week 5 and August rows to be scored
                continue
            dets = [d for d in discover.DETECTORS if d in pq.read_schema(p).names]
            part = sb.read_part(p, ["row_id", "role", "family"] + dets)
            for d in dets:
                s = part[d].to_numpy(np.float64)
                for (split, meth), a in alarms(part, s, meta, cal_host).items():
                    acc.setdefault((d, meth, split), []).append(a.astype(float))
                if (part.role == "test_unknown").any():
                    r = hs.fold(part, s, meta, cal_host, held)[0]
                    tpr += [{"seed": seed, "fold": held, "detector": d, "method": m, "tpr": r[f"tpr_{m}"],
                             "attacker_days": r["n_attacker_days"], "attacker_hosts": r["n_attacker_hosts"]}
                            for m in METHODS]
    rows = []
    for (d, meth, split), parts in acc.items():
        rate = pd.concat(parts, axis=1).mean(axis=1)  # host-day alarm averaged over folds and seeds
        lo, hi = ci(rate)
        rows.append({"detector": d, "method": meth, "split": split, "fpr": rate.mean(), "lo": lo, "hi": hi,
                     "host_days": len(rate), "hosts": len(np.unique(rate.index.to_numpy() // hs.MUL))})
    F = pd.DataFrame(rows)
    T = pd.DataFrame(tpr)
    F.to_csv(OUT / "fpr.csv", index=False)
    T.to_csv(OUT / "tpr.csv", index=False)
    L = [f"# UGR'16 forward-time test (PREREG §9 FWD)\n", f"Seeds {seeds}. Budget 1%. Claim: forward FPR 95% CI upper <= 1.5%.\n",
         "| detector | method | forward FPR [95% CI] (host-days) | holds | time-only | host-only |", "|---|---|---|---|---|---|"]
    for (d, meth), g in F.groupby(["detector", "method"]):
        g = g.set_index("split")
        f = g.loc["forward"]
        L.append(f"| {d} | {meth} | {100 * f.fpr:.2f}% [{100 * f.lo:.2f}, {100 * f.hi:.2f}] ({f.host_days}) | "
                 f"{'yes' if f.hi <= 0.015 else 'no'} ({f.fpr / 0.01:.1f}x) | {100 * g.loc['time_only'].fpr:.2f}% | "
                 f"{100 * g.loc['host_only'].fpr:.2f}% |")
    if len(T):
        L += ["\n## Host-day TPR per held-out family (mean over seeds)\n"]
        P = T.groupby(["fold", "detector", "method"]).agg(tpr=("tpr", "mean"), days=("attacker_days", "mean"),
                                                         hosts=("attacker_hosts", "mean")).reset_index()
        L += ["| family | attacker host-days | " + " | ".join(f"{d} {m}" for d in discover.DETECTORS for m in METHODS) + " |",
              "|---|---|" + "---|" * (len(discover.DETECTORS) * len(METHODS))]
        for fam, g in P.groupby("fold"):
            g = g.set_index(["detector", "method"])
            L.append(f"| {fam} | {g.days.iloc[0]:.0f} | " + " | ".join(
                f"{g.tpr.get((d, m), np.nan):.2f}" for d in discover.DETECTORS for m in METHODS) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
