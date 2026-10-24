"""OnePass-IDS claim table (PREREG OnePass rows, C1-C7) for one dataset, pooled over seeds.

Runs host_stream.py for every (variant, member) the claims need, then pools host-days over seeds. Macro = mean over
families of (detected host-days / host-days); CI = 95% bootstrap resampling attacker hosts (all their seeds) within
each family. C3 reading: host gain = A(host) - A(none) and perm gain = A(host_perm) - A(none), both with mlp_ce,
so the identity control isolates the features, not the aggregator.
Usage: .venv/bin/python scripts/onepass_claims.py nf_ton --seeds 0 1 2 3 4 --out results/onepass/nf_ton
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {
    "nf_cic": {"DoS": "Flood", "DDoS": "Flood", "BruteForce": "Access", "Web": "Access", "Bot": "Bot",
               "Infiltration": "Infiltration"},
    "nf_ton": {"dos": "Flood", "ddos": "Flood", "password": "Access", "xss": "Access", "injection": "Access",
               "Backdoor": "Malware", "ransomware": "Malware", "scanning": "scanning", "mitm": "mitm"},
    "nf_bot": {"DoS": "Flood", "DDoS": "Flood", "Reconnaissance": "Reconnaissance", "Theft": "Theft"},
}
HOST = {"unsw_raw": ("ct_drop", "ct_causal")}  # dataset: (none variant, host variant)


def stream(ds, variant, member, seed, out):
    f = out / f"{variant}_{member}_s{seed}"
    if not Path(f"{f}_days.csv").exists():
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "host_stream.py"), "--ds", ds,
                        "--variant", variant, "--member", member, "--seed", str(seed), "--out", str(f)],
                       check=True, stdout=subprocess.DEVNULL)
    days = pd.read_csv(f"{f}_days.csv")
    res = pd.read_json(f"{f}.json", orient="index")
    return days.assign(seed=seed), res.assign(seed=seed)


def pooled(ds, variant, member, seeds, out):
    d, r = zip(*(stream(ds, variant, member, s, out) for s in seeds))
    return pd.concat(d, ignore_index=True), pd.concat(r)


def macro(days, m):
    return days.groupby("fold")[m].mean().mean()


def boot_diff(da, ma, db, mb, reps=2000, seed=0):
    """CI of macro(da[ma]) - macro(db[mb]); da and db hold the same host-days in the same order."""
    rng = np.random.default_rng(seed)
    per = []
    for k, g in da.groupby("fold"):
        h = db[db.fold == k]
        assert (h.host.to_numpy() == g.host.to_numpy()).all(), "host-days differ between arms"
        x = pd.DataFrame({"host": g.host.to_numpy(), "a": g[ma].to_numpy(float), "b": h[mb].to_numpy(float)})
        s = x.groupby("host").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
        per.append(s.to_numpy())
    out = []
    for _ in range(reps):
        v = []
        for s in per:
            i = s[rng.integers(0, len(s), len(s))]
            v.append((i[:, 0].sum() - i[:, 1].sum()) / i[:, 2].sum())
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


def backoff(ds, seeds, out, variant):
    par = PARENTS[ds]
    rows = []
    for s in seeds:
        fl = pd.read_parquet(out / f"{variant}_mlp_ce_s{s}_flows.parquet")
        run = Path("/home/srioo/Desktop/Projects/root/BigP/runs/bank") / f"{ds}_{variant}_cap40000_s{s}"
        for k, g in fl[fl.A].groupby("fold"):
            part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet")
            part = part[part.role == "test_unknown"].set_index("row_id").loc[g.row_id]
            known = [c[6:] for c in part.columns if c.startswith("mlp_p_") and c[6:] != "Benign"]
            groups = sorted({par[f] for f in known})
            mass = np.column_stack([part[[f"mlp_p_{f}" for f in known if par[f] == p]].sum(1) for p in groups])
            pred = np.array(groups)[mass.argmax(1)]
            sibling = any(par[f] == par[k] for f in known)
            rows.append({"fold": k, "seed": s, "n": len(pred), "leaf": sibling, "chance": 1 / len(groups),
                         "acc": float((pred == par[k]).mean()) if sibling else float("nan")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    none, host = HOST.get(a.ds, ("none", "host"))
    P = lambda v, m: pooled(a.ds, v, m, a.seeds, a.out)  # noqa: E731
    one, one_r = P(host, "mlp_ce")
    base, _ = P(none, "mlp_ce")
    L = [f"# OnePass-IDS claims — {a.ds}, seeds {a.seeds}\n"]
    fam = one.groupby("fold").agg(hosts=("host", "nunique"), host_days=("host", "size"))
    tab = {"OnePass (host, mlp_ce, A)": (one, "A"), "per-flow MLP (none, P)": (base, "P")}
    trees = {}
    for m in ("rf", "xgb", "mcde"):
        d, _ = P(none, m)
        trees[m] = d
        tab[f"{m} (none, P)"] = (d, "P")
    best = max(trees, key=lambda m: macro(trees[m], "P"))
    for m in ("rf", "xgb"):
        d, _ = P(host, m)
        tab[f"{m} (host, A) upper ref"] = (d, "A")
    extra = {}
    for m in ("mlp_ce_int8", "mlp_sn_gda"):
        try:
            extra[m], _ = P(host, m)
            tab[f"{m} (host, A)"] = (extra[m], "A")
        except (subprocess.CalledProcessError, ValueError, KeyError):
            L.append(f"(no {m} column in these banks)")
    perm = None
    if a.ds.startswith("nf_"):
        try:
            perm, _ = P("host_perm", "mlp_ce")
            tab["host_perm (mlp_ce, A)"] = (perm, "A")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            L.append("(no host_perm banks)")
    noneA = macro(base, "A")
    L += ["| method | " + " | ".join(fam.index) + " | macro |", "|---|" + "---|" * (len(fam) + 1),
          "| attacker hosts / host-days | " + " | ".join(f"{r.hosts} / {r.host_days}" for r in fam.itertuples())
          + " | |"]
    for name, (d, m) in tab.items():
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in d.groupby("fold")[m].mean().reindex(fam.index))
                 + f" | {macro(d, m):.3f} |")
    mo, mb, mt = macro(one, "A"), macro(base, "P"), macro(trees[best], "P")
    lo, hi = boot_diff(one, "A", base, "P")
    L.append(f"\nC1 OnePass − per-flow MLP: {mo - mb:+.3f} [{lo:+.3f}, {hi:+.3f}] → "
             f"{'PASS' if mo - mb >= 0.10 and lo > 0 else 'FAIL'}")
    lo, hi = boot_diff(one, "A", trees[best], "P")
    L.append(f"C2 OnePass − best large per-flow baseline ({best}): {mo - mt:+.3f} [{lo:+.3f}, {hi:+.3f}] → "
             f"{'PASS' if mo - mt >= -0.02 else 'FAIL'}")
    if perm is not None:
        gh, gp = mo - noneA, macro(perm, "A") - noneA
        L.append(f"C3 host gain {gh:+.3f}, perm gain {gp:+.3f} → {'PASS' if gp < 0.5 * gh else 'FAIL'}")
    fpr = one_r["fpr_A"].mean()
    L.append(f"C4 realised benign host-day alarm rate {fpr:.4f} → {'PASS' if fpr <= 0.015 else 'FAIL'}")
    if "mlp_ce_int8" in extra:
        d8 = mo - macro(extra["mlp_ce_int8"], "A")
        L.append(f"C5 int8 loss {d8:+.3f} → {'PASS' if d8 <= 0.02 else 'FAIL'}")
    if "mlp_sn_gda" in extra:
        g = macro(extra["mlp_sn_gda"], "A") - mo
        lo, hi = boot_diff(extra["mlp_sn_gda"], "A", one, "A")
        L.append(f"C6 DDU − OnePass {g:+.3f} [{lo:+.3f}, {hi:+.3f}] → {'PASS' if g >= 0.03 and lo > 0 else 'FAIL'}")
    if a.ds in PARENTS:
        try:
            b = backoff(a.ds, a.seeds, a.out, host)
            leaf = b[b.leaf]
            if len(leaf):
                acc = (leaf.acc * leaf.n).sum() / leaf.n.sum()
                ch = leaf.chance.mean()
                L.append(f"C7 back-off accuracy (leaf case: {sorted(leaf.fold.unique())}) {acc:.3f}, chance {ch:.3f} → "
                         f"{'PASS' if acc >= 0.6 and acc >= 2 * ch else 'FAIL'}")
                L.append("   per family: " + ", ".join(f"{k} {(g.acc * g.n).sum() / g.n.sum():.3f} (n {g.n.sum()})"
                                                       for k, g in leaf.groupby("fold")))
        except (FileNotFoundError, KeyError, ValueError) as e:
            L.append(f"(back-off not available: {e})")
    under = [k for k, r in fam.iterrows() if r.hosts < 5]
    L.append(f"underpowered (< 5 attacker hosts): {under}")
    text = "\n".join(L)
    print(text)
    (a.out / "claims.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
