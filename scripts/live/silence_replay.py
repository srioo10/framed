"""PREREG SILENCE: replay the LIVE-POS P runs with S one-flow sprays to fresh destinations from the attacker's address
2 s before each attack cluster; report whether the attacker is still blocked (engine/replay_live.c, pos build).
Usage: python scripts/live/silence_replay.py results/live/pos [tau]"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_check import build, flows  # noqa: E402
from fair_report import runs  # noqa: E402

ATT, SENSOR, BROKER = "192.168.4.3", "192.168.4.2", "192.168.77.1"


def clusters(rows, gap_ms=30_000):
    """Device-ms start of each run of attacker non-MQTT flows (bursts and the scan)."""
    t = sorted(int(p[2]) for p in rows if p[3] == ATT and p[6] != "1883")
    return [x for i, x in enumerate(t) if i == 0 or x - t[i - 1] > gap_ms]


def spray(starts, S, run_i):
    out = []
    for c, t in enumerate(starts):
        for j in range(S):
            dst = f"10.{99 - run_i}.{c}.{j + 1}"  # never-seen destinations, one flow each (within m per key)
            out.append([None, None, str(t - 2000 + j), ATT, str(40000 + j), dst, "80", "6", "60", "1", "0", "0", "0"])
    return out


def main(d, tau="0.478"):
    F, R = flows(d), runs(d)
    exe = str(Path(tempfile.mkdtemp()) / "pos")
    build("pos", exe)
    P = R[(R.build == "pos") & (R.variant == "P")].reset_index(drop=True)
    print("S,runs_att_blocked,runs_att_blocked_scan,att_blocks,sensor_blocks,broker_blocks,abstains,max_evictions")
    for S in (0, 8, 16, 32, 64):
        tot = dict(runs=0, scan=0, att=0, sen=0, brk=0, ab=0, ev=0)
        for i, r in P.iterrows():
            rows = [p for p in F if r.t0 * 1000 <= int(p[0]) < r.t1 * 1000]
            off = int(rows[0][0]) - int(rows[0][2])  # Pi epoch ms - device ms
            allr = sorted(rows + spray(clusters(rows), S, i), key=lambda p: int(p[2]))
            out = subprocess.run([exe, tau], input="".join(",".join(p[2:13]) + "\n" for p in allr),
                                 capture_output=True, text=True, check=True).stdout.splitlines()
            assert len(out) == len(allr)
            if S == 0:  # must reproduce the logged decisions
                assert all(o.split(",")[7] == p[16] for p, o in zip(allr, out)), "S=0 replay differs from the log"
            ab = att = scan = 0
            for o in out:
                o = o.split(",")
                ts = (int(o[0]) + off) / 1000 - r.t0
                blk = o[7] == "1"
                att += blk and o[1] == ATT
                scan |= blk and o[1] == ATT and 1020 <= ts < 1200
                tot["sen"] += blk and o[1] == SENSOR
                tot["brk"] += blk and o[1] == BROKER
                ab += o[8] == "1"
                tot["ev"] = max(tot["ev"], int(o[9]))
            tot["runs"] += att > 0; tot["scan"] += bool(scan); tot["att"] += att; tot["ab"] += ab
        print(f"{S},{tot['runs']},{tot['scan']},{tot['att']},{tot['sen']},{tot['brk']},{tot['ab']},{tot['ev']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
