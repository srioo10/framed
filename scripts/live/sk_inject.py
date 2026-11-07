"""PREREG §9 LIVE-SK: shared-key host-context poisoning on the isolated lab only (own devices, no internet).

The attacker is an ordinary client on the gateway's IoT Wi-Fi (same /24 as the victim sensor). It makes --n real,
fully established TCP connections (handshake, HTTP GET, reply, FIN) from its OWN address to --n distinct lab peers,
so nothing is spoofed and a handshake / established-only filter passes every flow. With the gateway built with
CONFIG_GW_KEY_PREFIX24 the sensor and the attacker share one host-context key (as behind NAT / CGNAT), so these
flows raise the sensor's newpeer / count features. The readout is the sensor's own next benign flow in the gateway
log: its score (ordinary IPS) and its certified lower bound (Veto). Within budget n <= 10, over budget n = 50.

Peers: one per /24, because the PREFIX24 build keys peers by /24 too (peer novelty uses the masked server key), so n
peers inside one /24 would count as a single new peer. On the Pi, for k = first..first+n-1:
`sudo ip addr add 192.168.<k>.1/32 dev <lab-iface>`, and `python3 -m http.server 8080` (binds all addresses).
Usage: python3 sk_inject.py --n 10
"""
import argparse
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=120, help="first peer /24: peers are 192.168.<first+i>.1")
    ap.add_argument("--n", type=int, default=10, help="distinct peers = injected established flows (<=10 in budget)")
    ap.add_argument("--dport", type=int, default=8080)
    a = ap.parse_args()
    assert 100 <= a.first and a.first + a.n <= 200, "lab-only peer /24s 192.168.100-199 (none used by the testbed)"
    t0, ok = int(time.time() * 1000), 0
    for i in range(a.n):
        dst = f"192.168.{a.first + i}.1"
        try:
            with socket.create_connection((dst, a.dport), timeout=3) as s:
                s.sendall(b"GET / HTTP/1.0\r\nHost: lab\r\n\r\n")
                while s.recv(4096):
                    pass
            ok += 1
        except OSError as e:
            print(f"  {dst}: {e}")
    print(f"{t0} established {ok}/{a.n} flows -> 192.168.{a.first}..{a.first + a.n - 1}.1:{a.dport}")
    if ok < a.n:  # not fatal: an ordinary IPS may block the attacker mid-run; the log says which flows were metered
        print("WARNING: not every flow was established (Pi aliases down, or the attacker itself was blocked)")


if __name__ == "__main__":
    main()
