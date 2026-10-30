"""Exploratory probe (dev pairs 0-1 per dataset, cheap arena, seed 0; disclosed): robust-alarm gating with the
robust-detection loss (rdet) vs benign cert loss alone. Host-context bytes columns zeroed (the fit-range clip then
pins them, so no box is wide there). Prints plain fitness and robust fitness at m = 1 / 5 / 10."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import arena  # noqa: E402
import bowl  # noqa: E402

torch.set_num_threads(4)
G = {"family": "mlp", "enc": "raw", "e": 2, "w": 64, "depth": 2, "sn": 0, "act": "relu", "members": 1, "head": "nlog",
     "kd": 0.5, "syn": 0.5, "ls": 0.0, "focal": 2.0, "lr": 0.003, "clip": 1}
VARIANTS = {k: dict(zip(("cert", "rdet"), v)) for k, v in
            {"c4": (4, 0), "r1": (0, 1), "r4": (0, 4), "c2r2": (2, 2), "c4r4": (4, 4), "c4r8": (4, 8), "c8r4": (8, 4),
             "c8r8": (8, 8)}.items()}
SEED = int(__import__("os").environ.get("SEED", 0))
names = sys.argv[1:] or list(VARIANTS)
for z in [z for z in bowl.load_pairs(bowl.ARENA, cheap=True) if z["name"] in ("pair_0", "pair_1")]:
    j = [i for i, c in enumerate(z["cols"]) if c.startswith("hc_") and "_bytes_" in c]
    for k in [k for k in z if k.startswith("X_")]:
        z[k] = z[k].copy()
        z[k][:, j] = 0
    for v in names:
        g = G | VARIANTS[v]
        m, h, s = bowl.run_pair(g, z, "cpu", SEED, 15)
        f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
        r = {"pair": f"{z['ds']}/{z['name']}", "v": v, "seed": SEED, "plain": [round(f, 3), round(u, 3), round(k_, 3)]}
        for mm in (1, 5, 10):
            r[f"rob{mm}"] = [round(x, 3) for x in bowl.robust_fitness(g, m, z, s["cal"], "cpu", mm)]
        print(json.dumps(r), flush=True)
