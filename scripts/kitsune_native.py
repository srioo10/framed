"""K-NATIVE kill test (PREREG 2026-11-04): host-context poisoning against the ORIGINAL Kitsune pipeline.

Unmodified Kitsune-py components (third_party/Kitsune-py: netStat/AfterImage damped statistics + KitNET), the
example defaults (maxAE 10, FMgrace 5,000, ADgrace 50,000), on Kitsune's own Mirai capture (first 70,000 packets,
which the Kitsune example states are clean), parsed by tshark exactly as Kitsune's FeatureExtractor does (its TSV
branch is copied in `vec`, because FeatureExtractor imports scapy at module level and we parse with tshark only).
Shims, no behaviour change: numpy 2 removed np.Inf (AfterImage default arg); pyximport is stubbed (AfterImage is the
pure-python module, nothing is compiled).
Split: packets [0, 55k) train (Kitsune's grace periods), [55k, 62.5k) calibration (tau = 99% RMSE quantile),
[62.5k, 70k) victims. Victims = IPv4 hosts with >= 20 own (source) packets in the victim segment. Attack: per victim
own packet, spoofed packets with the victim's source IP and MAC to fresh destination IPs / ports, 1 ms apart just
before it, greedy, <= k per victim per 240 s. Readout per k: victim packet alarm share and victims with >= 1 alarm.
Usage: .venv/bin/python -I scripts/kitsune_native.py new_dataset/kitsune/clean70k.tsv
"""
import csv
import json
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
np.Inf = np.inf  # shim: numpy 2 removed the alias AfterImage uses as a default argument
sys.modules.setdefault("pyximport", types.SimpleNamespace(install=lambda *a, **k: None))  # shim: nothing to compile
sys.path.insert(0, str(ROOT / "third_party" / "Kitsune-py"))
import netStat as ns  # noqa: E402
from KitNET.KitNET import KitNET  # noqa: E402

FM, AD, CAL, END = 5_000, 50_000, 62_500, 70_000
BUDGET_S = 240.0


def vec(nstat, row):
    """Kitsune FeatureExtractor.get_next_vector, TSV branch, verbatim logic."""
    IPtype, timestamp, framelen, srcIP, dstIP = np.nan, row[0], row[1], "", ""
    if row[4] != "":
        srcIP, dstIP, IPtype = row[4], row[5], 0
    elif row[17] != "":
        srcIP, dstIP, IPtype = row[17], row[18], 1
    srcproto, dstproto = row[6] + row[8], row[7] + row[9]
    srcMAC, dstMAC = row[2], row[3]
    if srcproto == "":
        if row[12] != "":
            srcproto = dstproto = "arp"
            srcIP, dstIP, IPtype = row[14], row[16], 0
        elif row[10] != "":
            srcproto = dstproto = "icmp"
            IPtype = 0
        elif srcIP + srcproto + dstIP + dstproto == "":
            srcIP, dstIP = row[2], row[3]
    return nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp))


def inject(rows, victims, k, seed=0, ppf=1, reply=False):
    """Spoofed rows (victim src IP + MAC, 60-byte TCP) placed 1 ms apart before victim packets; a fresh dst IP / port
    every ppf packets (ppf packets per flow), <= k packets per victim per 240 s. reply: each injected packet gets a
reply from its peer 0.4 ms later (an established flow, as from a host sharing the victim's address behind a NAT)."""
    rng, out = np.random.default_rng(seed), []
    for v, mac in victims.items():
        q, n = deque(), 0
        for i in range(CAL, len(rows)):
            r = rows[i]
            if r[4] != v:
                continue
            t = float(r[0])
            for j in range(k, 0, -1):
                tau = t - 0.001 * j
                while q and q[0] <= tau - BUDGET_S:
                    q.popleft()
                if len(q) >= k or (q and tau <= q[-1]):
                    continue
                if n % ppf == 0:
                    tup = (f"6.6.{rng.integers(256)}.{rng.integers(256)}", str(int(rng.integers(40000, 65000))),
                           str(int(rng.integers(1, 65535))))
                q.append(tau)
                n += 1
                f = [""] * 19
                f[0], f[1], f[2], f[3] = f"{tau:.6f}", "60", mac, r[3]
                f[4], (f[5], f[6], f[7]) = v, tup
                out.append((tau, f))
                if reply:  # same flow, so not counted against the budget
                    g = [""] * 19
                    g[0], g[1], g[2], g[3] = f"{tau + 0.0004:.6f}", "60", r[3], mac
                    g[4], g[5], g[6], g[7] = tup[0], v, tup[2], tup[1]
                    out.append((tau + 0.0004, g))
    return out


def run(rows, extra):
    """RMSE of every original row from FM+AD on (NaN before), Kitsune trained on rows[:FM+AD] (no injection there)."""
    stream = [(float(r[0]), i, r) for i, r in enumerate(rows)] + [(t, -1, f) for t, f in extra]
    stream.sort(key=lambda x: (x[0], x[1] < 0 and 0 or 1))  # stable by time; injected rows precede ties
    nstat = ns.netStat(np.nan, 100000000000, 100000000000)
    kit = KitNET(len(nstat.getNetStatHeaders()), 10, FM, AD, 0.1, 0.75)
    rmse = np.full(len(rows), np.nan)
    for t, i, r in stream:
        s = kit.process(vec(nstat, r))
        if i >= FM + AD:
            rmse[i] = s
    return rmse


def main():
    with open(sys.argv[1], encoding="utf8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        rows = [r + [""] * (19 - len(r)) for r in rd][:END]
    assert float(rows[FM + AD][0]) < float(rows[CAL][0])
    cnt, mac = {}, {}
    for r in rows[CAL:]:
        if r[4]:
            cnt[r[4]] = cnt.get(r[4], 0) + 1
            mac.setdefault(r[4], r[2])
    victims = {v: mac[v] for v, c in cnt.items() if c >= 20}
    own = np.array([i >= CAL and rows[i][4] in victims for i in range(len(rows))])
    res = []
    clean = run(rows, [])
    tau = float(np.quantile(clean[FM + AD:CAL], 0.99))  # calibration segment, clean run
    ks = [(int(a.split("x")[0]), int(a.split("x")[1].rstrip("r")), a.endswith("r")) for a in sys.argv[2:]] \
        or [(0, 1, False), (10, 1, False), (100, 1, False)]
    for k, ppf, rep in ks:  # args like 30x3 = 30 packets per 240 s in flows of 3; 30x3r = with peer replies
        r = clean if k == 0 else run(rows, inject(rows, victims, k, ppf=ppf, reply=rep))
        al = r[own] > tau
        vic = np.array([rows[i][4] for i in np.flatnonzero(own)])
        per = {v: bool(al[vic == v].any()) for v in victims}
        res.append({"k": k, "ppf": ppf, "reply": rep, "tau": tau, "victims": len(victims), "victim_packets": int(own.sum()),
                    "packet_alarm": float(al.mean()), "victim_alarm": float(np.mean(list(per.values())))})
        print(json.dumps(res[-1]), flush=True)
    name = "mirai" if len(sys.argv) < 3 else "sharedkey" if any(a.endswith("r") for a in sys.argv[2:]) else "flow"
    out = ROOT / "results" / "cert" / f"kitsune_native_{name}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
