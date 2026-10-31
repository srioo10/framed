"""K11 (PREREG §9 ENGINE state exhaustion): can a spoofed-key flood make the bounded-memory engine block a benign flow
that its clean replay would not alarm on?

Stream: every raw flow touching the K1 victims' addresses (seed-1 benign test host-days, spoof_attack.victims), time
ordered, in certids' input format, scored by the exported champion B (engine/build_b, from scripts/engine_export_b.py
on a pair of the same dataset). Floods, all inside the threat model (<= m flows per host key per 240 s):
  evict: before every victim flow, E fresh spoofed keys (each used once, between two outside addresses) in the second
         before it, so a --hosts table is overwritten again and again;
  k1+evict: the K1 injection pattern (spoof_attack.inject at m on the victims' keys) plus the evict flood.
Violation = a victim flow blocked under a flood whose clean replay (same table, same age setting) does not alarm.
Run with and without the age rule (--no-age). Writes results/cert/engine_k11_{ds}.csv.
Usage: .venv/bin/python scripts/engine_k11.py nf_ugr16 [--hosts 64] [--evict 128] [--model engine/build_b/model_b.txt]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import spoof_attack as sa
from nids import load

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["L4_DST_PORT", "PROTOCOL", "FLOW_DURATION_MILLISECONDS", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]
ENGINE = sa.COLS + FEATS  # t, src, dst, dst_port, out_pkts, in_bytes, then the model's flow features


def injected(df):
    """Attacker-chosen flow fields for injected flows (DataFrame in sa.COLS)."""
    out = df.copy()
    out["PROTOCOL"], out["FLOW_DURATION_MILLISECONDS"], out["OUT_BYTES"], out["IN_PKTS"] = 6, 0, 0, 1
    return out


def evict_flood(times, e, rng):
    t = np.repeat(times, e) - rng.integers(1, 1000, len(times) * e)
    i = np.arange(len(t))
    src = [f"7.{(j >> 16) & 255}.{(j >> 8) & 255}.{j & 255}" for j in i]
    dst = [f"8.{(j >> 16) & 255}.{(j >> 8) & 255}.{j & 255}" for j in i]
    return injected(pd.DataFrame({"FLOW_START_MILLISECONDS": t, "IPV4_SRC_ADDR": src, "IPV4_DST_ADDR": dst,
                                  "L4_DST_PORT": 80, "OUT_PKTS": 1, "IN_BYTES": 60}))


def run(exe, model, df, hosts, age):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        df[ENGINE].to_csv(f, index=False, header=False)
    cmd = [str(exe), "--model", str(model), "--hosts", str(hosts)] + ([] if age else ["--no-age"])
    r = subprocess.run(cmd, stdin=open(f.name), capture_output=True, text=True, check=True)
    Path(f.name).unlink()
    out = np.array([[float(v) for v in line.split(",")] for line in r.stdout.split()])
    return out[:, 1] > 0, out[:, 3] > 0, r.stderr.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--hosts", type=int, default=64)
    ap.add_argument("--evict", type=int, default=128)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--model", type=Path, default=ROOT / "engine" / "build_b" / "model_b.txt")
    a = ap.parse_args()
    exe = a.model.parent / "certids_k11"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(ROOT / "engine" / "certids.c"), "-lm"], check=True)
    raw = load.LOADERS[a.dataset]()
    light, bank = sa.setup(a.dataset)
    meta = np.load(ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    rows, _ = sa.victims(bank, meta)
    vrows = bank.kept[rows]
    ips = set(light.IPV4_SRC_ADDR.iloc[vrows]) | set(light.IPV4_DST_ADDR.iloc[vrows])
    S_idx = np.flatnonzero((light.IPV4_SRC_ADDR.isin(ips) | light.IPV4_DST_ADDR.isin(ips)).to_numpy())
    S = light.iloc[S_idx].copy()
    for c in FEATS:
        S[c] = raw[c].to_numpy()[S_idx]
    del raw
    S["victim"] = np.isin(S_idx, vrows)
    S["inj"] = False
    vt = S.FLOW_START_MILLISECONDS.to_numpy()[S.victim.to_numpy()]
    rng = np.random.default_rng(0)
    floods = {"evict": evict_flood(vt, a.evict, rng)}
    V = light.iloc[vrows].sort_values("FLOW_START_MILLISECONDS", kind="stable")
    for side in ("dst", "src", "both"):
        k1 = injected(sa.inject(V, side, a.m, 1, 1, 1, 0))
        floods[f"k1_{side}+evict"] = pd.concat([k1, floods["evict"]], ignore_index=True)
    res = []
    for age in (1, 0):
        al0, bl0, log0 = run(exe, a.model, S.sort_values("FLOW_START_MILLISECONDS", kind="stable"), a.hosts, age)
        Sv = S.sort_values("FLOW_START_MILLISECONDS", kind="stable").victim.to_numpy()
        res.append({"age_rule": age, "flood": "clean", "victim_flows": int(Sv.sum()), "alarm": al0[Sv].mean(),
                    "block": bl0[Sv].mean(), "violations": 0, "engine": log0})
        print(res[-1], flush=True)
        for name, fl in floods.items():
            fl = fl.assign(victim=False, inj=True)
            for c in FEATS:
                fl[c] = fl[c].fillna(0) if c in fl else 0
            D = pd.concat([S, fl], ignore_index=True).sort_values("FLOW_START_MILLISECONDS", kind="stable")
            al, bl, log = run(exe, a.model, D, a.hosts, age)
            real = ~D.inj.to_numpy()
            al, bl, vic = al[real], bl[real], D.victim.to_numpy()[real]  # real flows keep their clean order
            res.append({"age_rule": age, "flood": name, "victim_flows": int(vic.sum()), "alarm": al[vic].mean(),
                        "block": bl[vic].mean(), "violations": int((bl & ~al0 & vic).sum()), "engine": log})
            print(res[-1], flush=True)
    R = pd.DataFrame(res)
    R.to_csv(sa.OUT / f"engine_k11_{a.dataset}.csv", index=False)
    print(R.drop(columns="engine").to_string())
    print("K11", "PASS" if R[R.age_rule == 1].violations.sum() == 0 else "FAIL",
          "| without the age rule:", int(R[R.age_rule == 0].violations.sum()), "violations")


if __name__ == "__main__":
    main()
