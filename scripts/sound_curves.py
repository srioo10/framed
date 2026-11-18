"""PREREG SOUND curves: harm (innocent clean host-days blocked) vs macro TPR as tau sweeps, per detector, from
gate_attack.py --clean-only --dump npz files. Block readout = b (certified bound for Veto / trees / shifted rules);
alarm readout = s (an IPS that blocks on alarm). Prints TPR at 1, 2 and 5% clean host-day harm; writes results/cert/sound_curves.csv.
Usage: python scripts/sound_curves.py results/cert/scores_{ds}_soundtpr.npz ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = (0.01, 0.02, 0.05)


def curve(key, v0, vu, vk):
    """(harm, tpr) at every distinct threshold: block iff value > t."""
    k, inv = np.unique(key, return_inverse=True)
    dmax = np.full(len(k), -np.inf)
    np.maximum.at(dmax, inv, np.nan_to_num(v0, nan=-np.inf))
    ts = np.unique(np.concatenate([dmax[np.isfinite(dmax)], [np.inf]]))
    harm = np.array([(dmax > t).mean() for t in ts])
    tpr = np.array([((vu > t).mean() + (vk > t).mean()) / 2 for t in ts])
    return harm, tpr


def at(harm, tpr, h):
    ok = harm <= h + 1e-12
    return float(tpr[ok].max()) if ok.any() else np.nan


def main(paths):
    rows = []
    for p in paths:
        z = np.load(p)
        ds = Path(p).stem.split("scores_")[1].rsplit("_", 1)[0]
        dets = sorted({k.split("__")[0] for k in z.files if "__" in k})
        for n in dets:
            for mode, w in (("block", "b"), ("alarm", "s")):
                if mode == "alarm" and n in ("veto_sound",) or (mode == "block" and n.startswith("rule_") and not n.endswith("_fp")):
                    continue
                h, t = curve(z["key"], z[f"{n}__{w}0"], z[f"{n}__unk_{w}"], z[f"{n}__known_{w}"])
                rows.append({"ds": ds, "detector": n, "readout": mode, **{f"tpr@{int(l * 100)}%": at(h, t, l) for l in LEVELS}})
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    out = Path("results/cert/sound_curves.csv")
    T.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
