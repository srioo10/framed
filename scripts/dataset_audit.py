"""Pre-evaluation audit of the §9 SEL datasets (seed 1 base bank): benign share, family overlap of attacker hosts,
attacker hosts that also send benign, test-split sizes, and score ties at the 1% test-benign threshold per baseline.
Usage: .venv/bin/python scripts/dataset_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import model_eval as me  # noqa: E402
import score_bank as sb  # noqa: E402


def audit(ds, seed=1):
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    src, fam = z["src"], pd.Series(z["fam"])
    att = fam.values != "Benign"
    hosts = pd.Series(src[att]).groupby(fam.values[att]).apply(lambda s: set(s.unique()))
    shared = {f: len(h & set().union(*[g for k, g in hosts.items() if k != f])) / len(h) for f, h in hosts.items()}
    ben_hosts = set(np.unique(src[~att]))
    att_hosts = set(np.unique(src[att]))
    cal = hs.host_split(z["ips"], seed)
    bh = ~np.isin(np.arange(len(cal)), list(att_hosts))
    rows = []
    for p in sorted((sb.OUT / f"{ds}_host_cap40000_s{seed}").glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        part = sb.read_part(p, ["row_id", "role"] + [d for d in me.BASE if d in pq.read_schema(p).names])
        h = src[part.row_id.to_numpy()]
        role = part.role.to_numpy()
        te = (role == "test_benign") & ~cal[h] & bh[h]
        unk = role == "test_unknown"
        r = {"fold": held, "unk_flows": int(unk.sum()), "unk_hosts": len(np.unique(h[unk])),
             "te_ben_flows": int(te.sum()), "te_ben_hosts": len(np.unique(h[te])),
             "shared_att_hosts": round(shared.get(held, np.nan), 2)}
        for d in me.BASE:
            if d in part and te.any():
                s = part[d].to_numpy(np.float64)
                r[f"tie_{d}"] = round(float((s[te] == np.quantile(s[te], 0.99)).mean()), 4)
        rows.append(r)
    head = {"ds": ds, "flows": len(src), "benign_share": round(float((~att).mean()), 4),
            "benign_hosts": len(ben_hosts), "attacker_hosts": len(att_hosts),
            "attackers_also_benign": len(ben_hosts & att_hosts)}
    return head, pd.DataFrame(rows)


if __name__ == "__main__":
    for ds in me.DATASETS:
        head, t = audit(ds)
        print(head)
        print(t.to_string(index=False), "\n", flush=True)
