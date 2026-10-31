"""K10 (PREREG §9 K10): champions A and B at the widths the bowl never searched (width gene was {16, 32, 64}, size
cap 64 KB). Fixed configs, full arena, seeds 1-2, 30 epochs; a config over 64 KB is dropped. Appends to
results/bowl/k10.json after each config.
Usage: .venv/bin/python scripts/k10_capacity.py [--only A_w128 ...] [--device cuda] [--arena DIR] [--out k10.json]
"""
import argparse
import json
from pathlib import Path

import bowl
import fighters as fx

A = {"family": "mlp", "enc": "raw", "e": 2, "w": 64, "depth": 2, "sn": 0, "act": "relu", "members": 1, "head": "energy",
     "kd": 0.5, "syn": 0.5, "ens": 1, "ls": 0.0, "focal": 2.0, "lr": 0.003}
B = {"family": "tabm", "enc": "raw", "e": 4, "w": 32, "depth": 2, "sn": 0, "act": "relu", "members": 8, "head": "nlog",
     "kd": 0.9, "syn": 0.5, "clip": 1, "cert": 4, "gate": 1, "rdet": 2, "ls": 0.0, "focal": 0.0, "lr": 0.003}
CONFIGS = {"A_w128": A | {"w": 128}, "A_w192": A | {"w": 192}, "A_w128_d3": A | {"w": 128, "depth": 3},
           "B_w64": B | {"w": 64}, "B_w128": B | {"w": 128}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arena", type=Path, default=bowl.ARENA)
    ap.add_argument("--out", default="k10.json")
    a = ap.parse_args()
    bowl.ARENA = a.arena
    bowl.OUT.mkdir(parents=True, exist_ok=True)
    out = bowl.OUT / a.out
    done = json.loads(out.read_text()) if out.exists() else {}
    full = bowl.load_pairs(bowl.ARENA, cheap=False)
    gated = bowl.drop_bytes([dict(z) for z in full], {"drop_bytes": 1})
    for name, g in CONFIGS.items():
        if name in done or (a.only and name not in a.only):
            continue
        pairs = gated if g.get("gate") else full
        size = fx.size_bytes(bowl.build(g, pairs[0], 1))
        if size > bowl.SIZE_MAX:
            done[name] = {"dropped": f"size {size} B > {bowl.SIZE_MAX}"}
        else:
            res, _ = bowl.evaluate(g, pairs, a.device, seeds=(1, 2), epochs=30)
            done[name] = {"genome": g, **res}
        print(name, json.dumps(done[name]), flush=True)
        out.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
