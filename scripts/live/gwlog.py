"""Record the Veto gateway's console on the Pi (PREREG §9 DEPLOY): each line gets the Pi's wall clock in ms as a first
field and is appended to <out>/gw-YYYYMMDD.log, so gateway decisions line up with the Pi's pcaps and the attack logs.
Reopens the port when the board resets or is reflashed.
Usage: python gwlog.py --port /dev/ttyUSB0 --out /opt/onepass/live/gw
"""
import argparse
import time
from pathlib import Path

import serial  # pyserial, installed with esptool in /opt/onepass/live/venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            with serial.Serial(a.port, a.baud, timeout=1) as s:
                buf = b""
                while True:
                    buf += s.read(4096)
                    *lines, buf = buf.split(b"\n")
                    if not lines:
                        continue
                    now = time.time()
                    with open(out / time.strftime("gw-%Y%m%d.log", time.localtime(now)), "a") as f:
                        for ln in lines:
                            f.write(f"{int(now * 1000)},{ln.decode('utf-8', 'replace').rstrip()}\n")
        except (serial.SerialException, OSError):
            time.sleep(2)  # board unplugged, resetting or being flashed


if __name__ == "__main__":
    main()
