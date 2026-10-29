"""Export one OnePass model for the C engine (engine/onepass.c) plus a parity stream with Python reference values.

Model: the bank's d-64-32-k MLP (CE) of one outer config, weights rounded to int8 (per-tensor symmetric scale), Prep
parameters, benign reference scores (rank role, calibration hosts), and the P / A thresholds at the 1% host-day
budget from the cal role of calibration hosts, exactly as host_stream.fold computes them.
Stream: every raw flow of a few whole captures (host keys are capture-prefixed, so their populations are complete),
time ordered, with the Python reference host features and score for the parity test.
Usage: .venv/bin/python scripts/engine_export.py nf_iot23 --held Okiru --out engine/build
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import hostctx, load, models  # noqa: E402

STREAM = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "OUT_PKTS", "IN_BYTES"]


def quantize(m):
    """[(W int8, scale, bias)] for body Linear layers and head; float model with the same rounded weights."""
    layers = [m.body[0], m.body[2], m.head]
    out = []
    for lin in layers:
        w = lin.weight.detach().double()
        s = float(w.abs().max() / 127)
        q = torch.round(w / s).clamp(-127, 127)
        out.append((q.numpy().astype(np.int8), s, lin.bias.detach().numpy().astype(np.float64)))
        with torch.no_grad():
            lin.weight.copy_((q * s).float())
    return out


def score_fn(m, bi):
    return lambda X: 1 - models.mlp_proba(m, X)[:, bi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--held", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--captures", default="c7,c34,ch4", help="host-key prefixes whose flows form the parity stream")
    ap.add_argument("--out", type=Path, default=ROOT / "engine" / "build")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bank = sb.Bank(a.dataset, a.seed, "host", 40000, 200000)
    P = (a.held,)
    fit, es = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    bi, k = code["Benign"], len(known)
    dfit = bank.frame(fit, P)
    prep = models.Prep().fit(dfit)
    assert not prep.codes, f"categorical features need a code table in the engine: {list(prep.codes)}"
    Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit]])
    es = es[np.isin(bank.fam[es], known)]
    Xe, ye = prep.transform(bank.frame(es, P)), np.array([code[f] for f in bank.fam[es]])
    mlp = models.train_mlp(Xf, yf, Xe, ye, k, seed=a.seed).eval()
    q = quantize(mlp)
    score = score_fn(mlp, bi)

    # calibration exactly as host_stream.fold: reference = rank role, cal streams = cal role, calibration hosts only
    z = np.load(ROOT / "data" / "processed" / f"{a.dataset}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(z["ips"], a.seed)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(z["src"][z["fam"] != "Benign"]))
    ch = lambda r: cal_host[z["src"][r]] & benign_host[z["src"][r]]  # noqa: E731
    rank, cal = bank.rank[ch(bank.rank)], bank.cal[ch(bank.cal)]
    ref = np.sort(score(prep.transform(bank.frame(rank, P))))
    s_cal = score(prep.transform(bank.frame(cal, P)))
    key = z["src"][cal].astype(np.int64) * hs.MUL + z["t"][cal] // hs.DAY_MS
    st = hs.stats(key, z["t"][cal], s_cal, ref)
    th = {m: float(hs.threshold(hs.day_max(key, st[m]))) for m in ("P", "A")}

    with open(a.out / "model.txt", "w") as f:
        d = Xf.shape[1]
        f.write(f"dims {d} {len(bank.feats)} {len(hostctx.COLS)} benign {bi} classes {k}\n")
        for name, v in (("mean", prep.mean), ("mu", prep.mu), ("sd", prep.sd)):
            f.write(name + " " + " ".join(f"{x:.9g}" for x in v) + "\n")
        for W, s, b in q:
            f.write(f"layer {W.shape[0]} {W.shape[1]} {s:.9g}\n" + " ".join(map(str, W.ravel())) + "\n" +
                    " ".join(f"{x:.9g}" for x in b) + "\n")
        f.write(f"ref {len(ref)}\n" + " ".join(f"{x:.9g}" for x in ref) + "\n")
        f.write("kappas " + " ".join(map(str, hs.KAPPAS)) + "\n")
        f.write(f"thresholds {th['P']:.9g} {th['A']:.9g}\n")

    # parity stream: all raw flows of the chosen captures, time ordered
    raw = load.LOADERS[a.dataset]()
    cap = raw["IPV4_SRC_ADDR"].astype(str).str.split(":").str[0]
    m = cap.isin(a.captures.split(",")).to_numpy()
    rows = np.flatnonzero(m)
    rows = rows[np.argsort(raw["FLOW_START_MILLISECONDS"].to_numpy()[rows], kind="stable")]
    hf = hostctx.HostFlows(raw.iloc[rows].reset_index(drop=True))
    H = hf.features(np.ones(len(rows), bool))  # test-role population: every flow
    X = raw.iloc[rows][bank.feats].reset_index(drop=True)
    X[hostctx.COLS] = H
    s = score(prep.transform(X))
    out = raw.iloc[rows][STREAM + bank.feats].reset_index(drop=True)
    out.to_csv(a.out / "stream.csv", index=False, header=False)
    pd.DataFrame(np.column_stack([H, s]), columns=hostctx.COLS + ["score"]).to_csv(a.out / "reference.csv", index=False)
    print(f"model d={d} classes={k} thresholds P={th['P']:.4g} A={th['A']:.4g}; stream {len(rows)} flows "
          f"from {a.captures}; per-flow features {bank.feats}")


if __name__ == "__main__":
    main()
