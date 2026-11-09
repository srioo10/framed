"""Export champion B for the C engine (engine/certids.c, engine/certb.h; PREREG §9 DEPLOY) and check parity.

B is trained on one arena / CONF pair exactly as the death match (bytes context zeroed, seed --seed, 30 epochs). Body
Linear weights and member heads are rounded to int8 (per-tensor symmetric scale); member input scales r and biases
stay float. tau = clean 1%-FPR threshold of the quantized model on the pair's calibration rows. Parity: on --rows
prepped rows (pseudo / es / cal), the C engine's score and certified lower bound (--prepped) must match Python's
(fighters Head / score_lower over bowl.dbox at m) on the same quantized weights; asserted.
Usage: .venv/bin/python scripts/engine_export_b.py nf_cic/pair_0 [--arena data/arena] [--out engine/build_b]
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

import bowl
import fighters as fx
from conf_clean import B
from nids import hostctx

ROOT = Path(__file__).resolve().parents[1]


def q8(t):
    w = t.detach().double()
    s = float(w.abs().max() / 127)
    q = torch.round(w / s).clamp(-127, 127)
    with torch.no_grad():
        t.copy_((q * s).float())
    return q.numpy().astype(np.int64), s


def quantise(m):
    """Round B to the deployed int8 weights in place: body Linear weights and member heads (r and biases stay float)."""
    assert isinstance(m, fx.Body) and m.M > 1 and len(m.body) == 2 * m.depth, "engine supports one TabM body"
    lins = [m.body[i] for i in range(0, len(m.body), 2)]
    return lins, [q8(lin.weight) for lin in lins], q8(m.hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--arena", type=Path, default=bowl.ARENA)
    ap.add_argument("--out", type=Path, default=ROOT / "engine" / "build_b")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--rows", type=int, default=3000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ds, name = a.pair.split("/")
    bowl.ARENA, bowl.DATASETS = a.arena, (ds,)
    z = bowl.drop_bytes([p for p in bowl.load_pairs(a.arena, cheap=False) if p["name"] == name], {"drop_bytes": 1})[0]
    F = z["nflow"]
    assert z["cols"][F:] == hostctx.COLS, "engine expects the flow features then hostctx.COLS in order"
    torch.set_num_threads(4)
    m, h, _ = bowl.run_pair(B, z, "cpu", a.seed, 30)
    m.eval()
    lins, Q, HW = quantise(m)
    bi, k, M, W, D = z["bi"], z["k"], m.M, m.w, len(z["cols"])
    s_cal = h.score(m, z["X_cal"], "cpu")
    tau = float(np.sort(s_cal)[::-1][int(np.floor(0.01 * len(s_cal)))])
    clo = np.clip(m.clo.numpy().astype(np.float64), -1e30, 1e30)
    chi = np.clip(m.chi.numpy().astype(np.float64), -1e30, 1e30)
    zero = np.array([c.startswith("hc_") and "_bytes_" in c for c in z["cols"]], int)
    f8 = lambda v: " ".join(f"{x:.17g}" for x in np.ravel(v))  # noqa: E731
    with open(a.out / "model_b.txt", "w") as f:
        f.write(f"certb {D} {F} {bi} {k} {M} {W} {len(lins)} {a.m} {tau:.17g}\n")
        for nm, v in (("mu", z["mu"]), ("sd", z["sd"]), ("clo", clo), ("chi", chi), ("zero", zero)):
            f.write(f"{nm} {f8(v)}\n")
        for r in m.r.detach().double().numpy():
            f.write(f"r {f8(r)}\n")
        for lin, (q, s) in zip(lins, Q):
            f.write(f"layer {q.shape[0]} {q.shape[1]} {s:.17g}\n{' '.join(map(str, q.ravel()))}\n"
                    f"{f8(lin.bias.detach().double().numpy())}\n")
        f.write(f"head {HW[1]:.17g}\n{' '.join(map(str, HW[0].ravel()))}\n{f8(m.hb.detach().double().numpy())}\n")
    int8 = sum(q.size for q, _ in Q) + HW[0].size
    # parity on prepped rows (already bytes-zeroed): score and certified lower bound
    rng = np.random.default_rng(0)
    X = np.vstack([z["X_pseudo"], z["X_es"], z["X_cal"]])
    X = X[np.sort(rng.choice(len(X), min(a.rows, len(X)), replace=False))].astype(np.float32)
    s_py = h.score(m, X, "cpu")
    lo_py = fx.score_lower(m, *bowl.dbox(z, X, a.m), bi, "nlog")
    np.savetxt(a.out / "parity_x.csv", X, delimiter=",", fmt="%.9g")
    exe = a.out / "certids"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(ROOT / "engine" / "certids.c"), "-lm"], check=True)
    r = subprocess.run([str(exe), "--model", str(a.out / "model_b.txt"), "--prepped"], input=(a.out / "parity_x.csv")
                       .read_text(), capture_output=True, text=True, check=True)
    C = np.array([[float(v) for v in line.split(",")] for line in r.stdout.split()])
    ds_, dl = np.abs(C[:, 0] - s_py), np.abs(C[:, 1] - lo_py)
    fin = np.isfinite(lo_py)
    dec_py, dec_c = lo_py > tau, C[:, 1] > tau
    print(f"B exported: {int8} int8 weights (+ {M * D} r, biases), tau {tau:.6g}; parity on {len(X)} rows: "
          f"max |score diff| {ds_.max():.2e}, max |lower diff| {dl[fin].max():.2e}, block flips {int((dec_py != dec_c).sum())}, "
          f"blocks {int(dec_c.sum())}")
    assert ds_.max() < 1e-4 and dl[fin].max() < 1e-3, "C engine disagrees with Python"
    assert (np.isfinite(C[:, 1]) == fin).all()


if __name__ == "__main__":
    main()
