"""LIVE-NAT-ENF (PREREG 2026-11-10): open TCP listeners on Pi #1, ports 2000-2099 on every address (the peer
aliases included), so the attacker's variant-B flows are fully established. Ports 3000+ stay closed (variant A).
Each connection gets a short HTTP reply and is closed. Usage (Pi #1): python3 listeners.py"""
import socket
import threading


def serve(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(16)
    while True:
        c, _ = s.accept()
        with c:
            c.settimeout(2)
            try:
                c.recv(1024)
                c.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok")
            except OSError:
                pass


for p in range(2000, 2100):
    threading.Thread(target=serve, args=(p,), daemon=True).start()
print("listening on 2000-2099")
threading.Event().wait()
