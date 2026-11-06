"""SHED live test: saturate the gateway with real TCP connection attempts, then (optionally) run an attack.

python scripts/live/flood.py <target_ip> <rate_per_s> <seconds> [--ports 1-1024]

Opens non-blocking TCP connections to target:port at <rate_per_s> (each attempt = one flow at the gateway), so the
gateway's decision rate (about 17 certified decisions per second) is exceeded at rate >= 34. Run the real attack
(sk_inject.py or e3_inject.py) from a second source while this runs; the gateway's Q,<t>,<qdrop>,<nshed> lines and
F lines tell which attack flows were scored and blocked.
"""
import argparse
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("rate", type=float)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--ports", default="1-1024")
    a = ap.parse_args()
    lo, hi = map(int, a.ports.split("-"))
    assert 0 < a.rate <= 2000 and a.seconds <= 600, "lab limits: <= 2000/s, <= 10 min"
    socks, n, t0, port = [], 0, time.time(), lo
    while time.time() - t0 < a.seconds:
        s = socket.socket()
        s.setblocking(False)
        s.connect_ex((a.target, port))
        socks.append(s)
        n, port = n + 1, lo + (port - lo + 1) % (hi - lo + 1)
        if len(socks) > 512:  # close old attempts so the laptop does not run out of descriptors
            socks.pop(0).close()
        time.sleep(max(0.0, t0 + n / a.rate - time.time()))
    for s in socks:
        s.close()
    print(f"sent {n} connection attempts in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
