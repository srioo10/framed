"""NATIVE-HE (PREREG 2026-11-06): host-context poisoning against the ORIGINAL HorusEye pipeline, on its own IoT data.

Unmodified HorusEye code (third_party/HorusEye @ ee3b8cc): stage 1 = iForest_detect (burst features pk_num, sum_len per
5-tuple-hash key; a key is forwarded to stage 2 if > 95% of its bursts are anomalous; port-0 packets such as ARP are
always forwarded), stage 2 = Magnifier autoencoder on per-packet AfterImage features (HorusEye's netStat) + ports.
Packets are parsed with tshark instead of scapy (scapy is not installed); `vec` reproduces FE_HorusEye.get_next_vector's
scapy branch field by field, and `check` asserts that it reproduces the authors' released feature CSV for the same pcap.
Shims, no behaviour change: pyximport stubbed (AfterImage is the pure-python module); numpy 2 aliases.
Usage: .venv/bin/python -I scripts/horuseye_native.py check <device> <file index> [rows]
"""
import csv
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HE = ROOT / "third_party" / "HorusEye"
np.Inf = np.inf  # shim: numpy 2 removed the alias
sys.modules.setdefault("pyximport", types.SimpleNamespace(install=lambda *a, **k: None))  # shim: nothing to compile
sys.path.insert(0, str(HE))
import netStat as ns  # noqa: E402  HorusEye's copy (Kitsune netStat + AfterImage)

FIELDS = ["frame.time_epoch", "frame.cap_len", "eth.src", "eth.dst", "ip.src", "ip.dst", "ip.proto", "tcp.srcport",
          "tcp.dstport", "udp.srcport", "udp.dstport", "icmp.type", "arp.opcode", "arp.src.proto_ipv4",
          "arp.dst.proto_ipv4", "ipv6.src", "ipv6.dst", "icmpv6.type"]
T, LEN, ESRC, EDST, IPS, IPD, PROTO, TSP, TDP, USP, UDP_DP, ICMP, ARP, ARPS, ARPD, V6S, V6D, ICMP6 = range(len(FIELDS))


def pcap_rows(pcap):
    """One list of FIELDS strings per frame (first occurrence of each field, as scapy's first matching layer)."""
    cmd = ["tshark", "-r", str(pcap), "-T", "fields", "-E", "occurrence=f", "-E", "separator=\t"]
    for f in FIELDS:
        cmd += ["-e", f]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [(line.split("\t") + [""] * len(FIELDS))[:len(FIELDS)] for line in out.splitlines()]


def row_args(r):
    """Parse a tshark row to (ts, key, srcport, dstport, update_args), update_args = FE_HorusEye's updateGetStats args.

    Factored out of vec so a merged/synthetic packet stream can be driven through one shared netStat (injection).
    """
    srcport = dstport = proto = 0
    IPtype = np.nan
    if r[IPS]:  # IPv4
        srcIP, dstIP, proto, IPtype = r[IPS], r[IPD], int(r[PROTO]), 0
    elif r[V6S]:
        srcIP, dstIP, IPtype = r[V6S], r[V6D], 1
    else:
        srcIP = dstIP = ""
    inner = bool(r[ICMP] or r[ICMP6])  # tshark shows an ICMP error's quoted TCP/UDP header; scapy's haslayer does not
    if r[TSP] and not inner:
        srcproto, dstproto = r[TSP], r[TDP]
        srcport, dstport = int(r[TSP]), int(r[TDP])
    elif r[USP] and not inner:
        srcproto, dstproto = r[USP], r[UDP_DP]
        srcport, dstport = int(r[USP]), int(r[UDP_DP])
    else:
        srcproto = dstproto = ""
    srcMAC, dstMAC = r[ESRC], r[EDST]
    if srcproto == "":
        if r[ARP]:
            srcproto = dstproto = "arp"
            srcIP, dstIP, IPtype = r[ARPS], r[ARPD], 0
        elif r[ICMP]:
            srcproto = dstproto = "icmp"
            IPtype = 0
        elif srcIP + srcproto + dstIP + dstproto == "":
            srcIP, dstIP = r[ESRC], r[EDST]
    h1 = hashlib.md5(str([srcIP, srcport, proto]).encode("utf-8"))
    h2 = hashlib.md5(str([dstIP, dstport, proto]).encode("utf-8"))
    key = int.from_bytes(h1.digest(), byteorder="big") + int.from_bytes(h2.digest(), byteorder="big")
    args = (IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, int(r[LEN]), float(r[T]))
    return float(r[T]), key, srcport, dstport, args


def vec(nstat, r):
    """FE_HorusEye.get_next_vector (scapy branch), on tshark fields: [hash, srcport, dstport, *netStat features]."""
    _, key, srcport, dstport, args = row_args(r)
    return key, srcport, dstport, nstat.updateGetStats(*args)


def features(rows):
    """Per-frame (key, sport, dport, stats) over a stream, with FE's netStat limits."""
    nstat = ns.netStat(np.nan, 100000000000, 100000000000)
    return [vec(nstat, r) for r in rows]


def device_files(dev):
    pcaps = sorted((HE / "DataSets" / "Pcap" / "Normal" / dev).glob("*.pcap"))
    csvs = sorted((HE / "DataSets" / "normal-kitsune_test" / dev).glob("*.csv"))
    return pcaps, csvs


def check(dev, i, n):
    """Our extractor vs the authors' released CSV for pcap i of device dev (first n frames)."""
    pcaps, csvs = device_files(dev)
    rows = pcap_rows(pcaps[i])[:n]
    ours = features(rows)
    with open(csvs[i]) as f:
        ref = [next(csv.reader([line])) for _, line in zip(range(n), f)]
    assert len(ref) == len(ours), (len(ref), len(ours))
    bad_key = sum(int(r[0]) != o[0] or int(r[1]) != o[1] or int(r[2]) != o[2] for r, o in zip(ref, ours))
    diff = max(float(np.max(np.abs(np.array(r[3:], float) - np.asarray(o[3])))) for r, o in zip(ref, ours))
    print(f"{dev}[{i}] {pcaps[i].name}: {n} frames, key/port mismatches {bad_key}, max |feature diff| {diff:.3g}")
    return bad_key, diff


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 20000)
