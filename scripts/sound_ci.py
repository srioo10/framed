"""PREREG SOUND-CI: paired bootstrap CI of Veto's equal-harm lead (scores_*_soundtpr.npz, pair 0 seed 1).
Resample clean host-days (keys) and unknown/known attack flows with replacement, same draws for every detector;
per draw, each detector's tau is the smallest threshold with clean host-day harm <= h; lead = Veto TPR - best rival TPR,
rival fixed = highest point estimate among rf, xgb, rule_*_fp. Usage: python scripts/sound_ci.py npz..."""
import sys

import numpy as np

RIVALS = ("rf", "xgb", "rule_f2b_fp", "rule_rate_fp", "rule_scan_fp", "rule_union_fp")


def dmax_per_key(key, v0):
    k, inv = np.unique(key, return_inverse=True)
    d = np.full(len(k), -np.inf)
    np.maximum.at(d, inv, np.nan_to_num(v0, nan=-np.inf))
    return d


def tpr_at(d, vu, vk, h):
    s = np.sort(d)[::-1]
    j = int(np.floor(h * len(s) + 1e-9))
    t = s[j] if j < len(s) else -np.inf
    return ((vu > t).mean() + (vk > t).mean()) / 2


def main(paths, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    for p in paths:
        z = np.load(p)
        dets = ["veto_sound"] + [r for r in RIVALS if f"{r}__b0" in z.files]
        D = {n: (dmax_per_key(z["key"], z[f"{n}__b0"]), z[f"{n}__unk_b"], z[f"{n}__known_b"]) for n in dets}
        nk, nu, nkn = len(D["veto_sound"][0]), len(D["veto_sound"][1]), len(D["veto_sound"][2])
        for h in (0.01, 0.05):
            pt = {n: tpr_at(*D[n], h) for n in dets}
            riv = max(RIVALS, key=lambda r: pt.get(r, -1))
            diff = []
            for _ in range(B):
                ik, iu, ikn = rng.integers(0, nk, nk), rng.integers(0, nu, nu), rng.integers(0, nkn, nkn)
                f = lambda n: tpr_at(D[n][0][ik], D[n][1][iu], D[n][2][ikn], h)
                diff.append(f("veto_sound") - f(riv))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            print(f"{p.split('scores_')[1].rsplit('_', 1)[0]} harm {h:.0%}: Veto {pt['veto_sound']:.3f} vs {riv} {pt[riv]:.3f}; "
                  f"lead {pt['veto_sound'] - pt[riv]:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'LEADS' if lo > 0 else ('BEHIND' if hi < 0 else 'LEVEL')}")


if __name__ == "__main__":
    # self-check: identical detectors give a zero lead
    d = np.array([0.1, 0.5, 0.9]); assert tpr_at(d, np.array([0.95, 0.2]), np.array([1.0]), 0.34) == 0.75
    main(sys.argv[1:])
