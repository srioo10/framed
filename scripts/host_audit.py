"""Dataset entry audit for the OnePass-IDS row (PREREG §9): after §3 cleaning, per family the rows, attacker source
hosts and host-days; benign hosts and host-days; then the single-feature fingerprint audit (feature_audit.audit).
Saves the per-row host metadata used by host_stream.py.
Usage: .venv/bin/python scripts/host_audit.py nf_ton --out results/host_stream/audit_nf_ton
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
from nids import clean, load  # noqa: E402
import feature_audit  # noqa: E402

DAY_MS = 86_400_000
KEYS = {"unsw_raw": ("srcip", "stime", 1000)}  # dataset: (source column, start-time column, ms per unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-features", action="store_true")
    a = ap.parse_args()
    raw = load.LOADERS[a.dataset]()
    sc, tc, mul = KEYS.get(a.dataset, ("IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS", 1))
    ips, src = np.unique(raw[sc].astype(str).to_numpy().astype("U"), return_inverse=True)
    t = raw[tc].to_numpy(np.int64) * mul
    df, rep = clean.clean(raw, a.dataset)
    del raw
    k = rep["kept_rows"]
    src, t, fam = src[k].astype(np.int32), t[k], df["family"].astype(str).to_numpy().astype("U")
    np.savez_compressed(ROOT / "data" / "processed" / f"{a.dataset}_hostmeta.npz", ips=ips, src=src, t=t, fam=fam)
    day = t // DAY_MS
    g = pd.DataFrame({"fam": fam, "host": src, "day": day})
    attack_hosts = set(g.host[g.fam != "Benign"])
    out = {"dataset": a.dataset, "rows": int(len(g)), "days": int(g.day.nunique()), "families": {}}
    for f, d in g.groupby("fam"):
        out["families"][f] = {"rows": int(len(d)), "hosts": int(d.host.nunique()),
                              "host_days": int(len(d.drop_duplicates(["host", "day"])))}
    b = g[(g.fam == "Benign") & ~g.host.isin(attack_hosts)]
    out["benign_only_hosts"] = int(b.host.nunique())
    out["benign_only_host_days"] = int(len(b.drop_duplicates(["host", "day"])))
    if not a.no_features:
        out["single_feature_tpr"] = feature_audit.audit(df)[:15]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
