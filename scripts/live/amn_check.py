"""PREREG LIVE-AMNESTY bars A1-A4 from a run dir (fair_runs.txt, gw-*.log, sub.log). Pi clock throughout.
Usage: python scripts/live/amn_check.py <dir> <build>   (build 'col' replays LIVE-COLLATERAL as the baseline)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_replay import runs  # noqa: E402

KEY, TTL = "192.168.4.2", 300_000


def main(d, build):
    d = Path(d)
    lines = [ln.split(",") for f in sorted(d.glob("gw-*.log")) for ln in f.read_text(errors="replace").splitlines()]
    B = sorted(int(p[0]) for p in lines if len(p) >= 5 and p[1] == "B" and p[3] == KEY)
    F = [p for p in lines if len(p) >= 17 and p[1] == "F" and p[3] == KEY]
    pub = sorted(float(ln.split()[0]) for ln in (d / "sub.log").read_text(errors="replace").splitlines()
                 if ln[:1].isdigit() and "up_ms" in ln)  # sensor publishes only
    a1 = a2 = a3 = nP = 0; a4 = True
    for var in ("P", "PC"):
        for rep, (t0, t1) in sorted(runs(d, build, var).items()):
            b = [t for t in B if t0 * 1000 <= t < t1 * 1000]
            # P: gap from the first block to run end; PC: from the sensor's first publish after the reset
            w0 = b[0] / 1000 if var == "P" and b else next((x for x in pub if x >= t0), float(t1))
            ps = [w0] + [x for x in pub if w0 <= x < t1] + [float(t1)]
            gap = max(y - x for x, y in zip(ps, ps[1:]))
            new = [p for p in F if p[6] != "1883" and any(bt <= int(p[0]) - int(p[12]) < bt + TTL for bt in b)]
            if var == "PC":
                a4 &= not b and gap <= 30
            else:
                nP += 1; a1 += bool(b) and gap <= 30; a2 += bool(b); a3 += bool(b) and not new
            print(f"{build} {var} {rep}: blocks {len(b)} first {round(b[0] / 1000 - t0) if b else '-'} s; "
                  f"longest sensor gap {round(gap)} s; new-conn records in block {len(new)}")
    print(f"A1 {a1}/{nP} (need >=4)  A2 {a2}/{nP} (5)  A3 {a3}/{nP} (5)  A4 {a4}")


if __name__ == "__main__":
    main(*sys.argv[1:])
