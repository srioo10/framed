"""Bidirectional flow meter for the OnePass lab (PREREG §9 DEPLOY): pcap -> NF-v3-style flow CSV, standard library only.

Reads tcpdump's rotating pcaps (EN10MB, any snaplen) in name order, keeps one flow table across files and appends
finished flows to <out>/flows-YYYYMMDD.csv with the columns hostctx.HostFlows and B need. A flow = (proto, initiator
ip:port, responder ip:port); the initiator is the sender of the first packet seen, except a TCP flow first seen without
a bare SYN, where the side with the larger port is taken as the client. IN_* = initiator -> responder, OUT_* = reply;
bytes are IP total length (L3, as NetFlow). A flow ends on TCP RST, FIN from both sides, IDLE_S without packets or
ACTIVE_S after its start (then it continues as a new flow). Time is packet time, so a replayed pcap gives the same
flows as the live run.
  live:   python3 flowmeter.py --watch /opt/onepass/live/pcap --out /opt/onepass/live/flows
  replay: python3 flowmeter.py --pcap a.pcap [b.pcap ...] --out DIR
"""
import argparse
import csv
import socket
import struct
import time
from pathlib import Path

IDLE_S, ACTIVE_S = 30.0, 120.0  # shortcut: nProbe-like defaults; the ESP32 meter must use the same two numbers
COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR",
        "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS", "END_REASON"]


def packets(path):
    """Yield (ts, src, dst, proto, sport, dport, iplen, tcpflags) for each IPv4 packet of a pcap file."""
    with open(path, "rb") as f:
        g = f.read(24)
        if len(g) < 24:
            return
        magic = struct.unpack("<I", g[:4])[0]
        if magic in (0xa1b2c3d4, 0xa1b23c4d):
            e = "<"
        elif magic in (0xd4c3b2a1, 0x4d3cb2a1):
            e = ">"
        else:
            raise ValueError(f"{path}: not a pcap file")
        scale = 1e-9 if magic in (0xa1b23c4d, 0x4d3cb2a1) else 1e-6
        if struct.unpack(e + "I", g[20:24])[0] != 1:
            raise ValueError(f"{path}: link type is not Ethernet")
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            sec, frac, incl, _ = struct.unpack(e + "IIII", h)
            d = f.read(incl)
            if len(d) < incl:
                return  # file still being written: the caller only reads closed files
            o, et = 14, struct.unpack("!H", d[12:14])[0] if len(d) >= 14 else 0
            while et in (0x8100, 0x88a8) and len(d) >= o + 4:
                et, o = struct.unpack("!H", d[o + 2:o + 4])[0], o + 4
            if et != 0x0800 or len(d) < o + 20:
                continue
            ihl = (d[o] & 15) * 4
            iplen, frag, proto = struct.unpack("!H", d[o + 2:o + 4])[0], struct.unpack("!H", d[o + 6:o + 8])[0], d[o + 9]
            src, dst = socket.inet_ntoa(d[o + 12:o + 16]), socket.inet_ntoa(d[o + 16:o + 20])
            sport = dport = flags = 0
            l4 = o + ihl
            if frag & 0x1fff == 0 and proto in (6, 17) and len(d) >= l4 + 4:
                sport, dport = struct.unpack("!HH", d[l4:l4 + 4])
                if proto == 6 and len(d) >= l4 + 14:
                    flags = d[l4 + 13]
            yield sec + frac * scale, src, dst, proto, sport, dport, iplen, flags


class Meter:
    def __init__(self, emit):
        self.flows, self.emit, self.now = {}, emit, 0.0

    def add(self, ts, src, dst, proto, sport, dport, iplen, flags):
        self.now = max(self.now, ts)
        a, b = (src, sport), (dst, dport)
        k = (proto,) + (a + b if a <= b else b + a)
        fl = self.flows.get(k)
        if fl is not None and (ts - fl["last"] > IDLE_S or ts - fl["start"] > ACTIVE_S):
            self.close(k, "idle" if ts - fl["last"] > IDLE_S else "active")
            fl = None
        if fl is None:
            client = a
            if proto == 6 and not (flags & 0x02 and not flags & 0x10):  # not a bare SYN: larger port = client
                client = a if sport >= dport else b
            fl = self.flows[k] = {"cli": client, "srv": b if client == a else a, "start": ts, "last": ts,
                                  "ib": 0, "ip": 0, "ob": 0, "op": 0, "flags": 0, "fin": set()}
        fwd = a == fl["cli"]
        fl["ib" if fwd else "ob"] += iplen
        fl["ip" if fwd else "op"] += 1
        fl["flags"] |= flags
        fl["last"] = ts
        if proto == 6:
            if flags & 0x01:
                fl["fin"].add(fwd)
            if flags & 0x04 or len(fl["fin"]) == 2:
                self.close(k, "rst" if flags & 0x04 else "fin")

    def expire(self, now):
        for k in [k for k, f in self.flows.items() if now - f["last"] > IDLE_S]:
            self.close(k, "idle")

    def close(self, k, why):
        f = self.flows.pop(k)
        ms = lambda t: int(round(t * 1000))
        self.emit([ms(f["start"]), ms(f["last"]), f["cli"][0], f["cli"][1], f["srv"][0], f["srv"][1], k[0], f["ib"],
                   f["ip"], f["ob"], f["op"], f["flags"], ms(f["last"]) - ms(f["start"]), why])

    def flush(self):
        for k in list(self.flows):
            self.close(k, "flush")


def writer(out):
    out.mkdir(parents=True, exist_ok=True)

    def emit(row):
        p = out / time.strftime("flows-%Y%m%d.csv", time.localtime(row[0] / 1000))
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLS)
            w.writerow(row)
    return emit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", nargs="+", help="replay these files, then flush")
    ap.add_argument("--watch", help="directory of rotating pcaps (all but the newest are closed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = Meter(writer(Path(a.out)))
    if a.pcap:
        for p in a.pcap:
            for pk in packets(p):
                m.add(*pk)
        m.flush()
        return
    done_log = Path(a.out) / "processed.txt"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    while True:
        files = sorted(Path(a.watch).glob("*.pcap"))
        for p in files[:-1]:  # the newest file is still open in tcpdump
            if p.name in done:
                continue
            for pk in packets(p):
                m.add(*pk)
            m.expire(m.now)
            done.add(p.name)
            with open(done_log, "a") as f:
                f.write(p.name + "\n")
            print(f"{p.name}: {len(m.flows)} open flows", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
