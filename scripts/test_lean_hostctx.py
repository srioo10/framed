"""Exactness check for the memory-lean hostctx.HostFlows and spoof_attack.setup (2026-11-01): on each dataset, the
lean and reference (scripts/hostctx_ref.py) implementations must give bit-identical host codes and host features for
every population used (all flows, each single family removed, and isolated on small data), and the 6-column light
frame must equal the old full-load one. Exits non-zero on any difference.
Usage: .venv/bin/python scripts/test_lean_hostctx.py nf_ctu13 [nf_ugr16 ...]
"""
import sys

import numpy as np
import pandas as pd

import hostctx_ref
import spoof_attack as sa
from nids import hostctx, load


def check(ds):
    raw = load.LOADERS[ds]()
    old_light = pd.DataFrame({c: raw[c].to_numpy() for c in sa.COLS})
    for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
        old_light[c] = old_light[c].astype(str)
    new_light = pd.read_parquet(load.CACHE / f"{ds}.parquet", columns=sa.COLS)
    for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
        new_light[c] = new_light[c].astype(str)
    pd.testing.assert_frame_equal(old_light, pd.DataFrame({c: new_light[c].to_numpy() for c in sa.COLS}))
    del old_light, new_light
    ref, new = hostctx_ref.HostFlows(raw), hostctx.HostFlows(raw)
    for k in ("src", "dst"):
        for a in ("key", "pair_port", "pair_peer"):
            np.testing.assert_array_equal(getattr(ref, a)[k], getattr(new, a)[k], err_msg=f"{ds} {a} {k}")
    np.testing.assert_array_equal(ref.family, new.family)
    fams = [()] + [(f,) for f in sorted(set(ref.family))]
    for R in fams:
        pop = ~np.isin(ref.family, list(R))
        np.testing.assert_array_equal(ref.features(pop), new.features(~new._fam_in(R)), err_msg=f"{ds} R={R}")
        print(ds, "R =", R, "identical", flush=True)
    if len(raw) < 2_000_000:
        np.testing.assert_array_equal(ref.isolated(), new.isolated())
        print(ds, "isolated identical", flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        check(ds)
    print("LEAN_EXACT_OK", flush=True)
