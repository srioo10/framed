"""Deployment cost of the OnePass models (PREREG OnePass row, C5 reporting): serialized size and single-thread CPU
latency per flow (batch 1) for the MLP (fp32, int8) and the DDU arm, plus RF/XGB at the bank's settings for scale.
Shapes match nf_cic host bank inputs; weights are random (cost does not depend on them).
Usage: taskset -c 11 .venv/bin/python scripts/onepass_cost.py --out results/onepass/cost
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import models  # noqa: E402


def size(obj):
    b = io.BytesIO()
    torch.save(obj.state_dict(), b) if hasattr(obj, "state_dict") else pickle.dump(obj, b)
    return len(b.getvalue())


def lat(f, X, reps=2000):
    for x in X[:100]:
        f(x[None])
    t = time.perf_counter()
    for x in X[:reps]:
        f(x[None])
    return (time.perf_counter() - t) / reps * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=67)  # nf_cic host bank: 47 flow + 20 host-context features
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, a.d)).astype(np.float32)
    y = rng.integers(0, a.k, len(X))
    m = models.train_mlp(X, y, X[:1000], y[:1000], a.k, epochs=1).eval()
    q = models.int8(m)
    sn = models.train_mlp(X, y, X[:1000], y[:1000], a.k, epochs=1, arch=models.SNMLP).eval()
    gda = models.FeatureGDA().fit(sn, X, y, a.k)
    rf = models.rf(0).set_params(n_jobs=1).fit(X, y)
    xg = models.xgb(0).set_params(n_jobs=1).fit(X, y)
    with torch.no_grad():
        rows = [("MLP fp32", size(m), lat(lambda x: m(torch.from_numpy(x)), X)),
                ("MLP int8 (OnePass)", size(q), lat(lambda x: q(torch.from_numpy(x)), X)),
                ("DDU MLP + class Gaussians", size(sn) + sum(t.numel() * 8 for t in gda.prec + gda.mu),
                 lat(lambda x: gda.score(sn, x), X)),
                ("RF (300 trees)", size(rf), lat(rf.predict_proba, X, 300)),
                ("XGB", size(xg), lat(xg.predict_proba, X, 300))]
    L = [f"# OnePass deployment cost (d = {a.d}, k = {a.k}; single CPU thread, batch 1)\n",
         "| model | serialized size | latency per flow |", "|---|---|---|"]
    L += [f"| {n} | {s / 1024:.1f} KB | {t:.1f} µs |" for n, s, t in rows]
    L.append("\nHost state: one float32 CUSUM value + day stamp per source host (≤ 10 bytes). Latency is PyTorch / "
             "sklearn on this CPU, not an MCU measurement.")
    text = "\n".join(L)
    print(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
