"""Re-derive every logged live block decision from the gateway's own flow records (engine/replay_live.c), per run.
Usage: python scripts/live/replay_check.py results/live/fair     Prints per build: runs, flows, score / decision mismatches."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fair_report import runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"veto": [], "sound": ["-DGW_SOUND", "-DHC_FLAG_MS=250000"], "scan": ["-DGW_SOUND", "-DHC_FLAG_MS=250000", "-DHC_BLOOM"],
         "pos": ["-DGW_SOUND", "-DHC_FLAG_MS=250000", "-DHC_BLOOM"], "pos_mon": ["-DGW_SOUND", "-DHC_FLAG_MS=250000", "-DHC_BLOOM"],
         "ord": ["-DORDINARY"]}


def build(name, out):
    subprocess.run(["cc", "-O1", *FLAGS[name], "-I", str(ROOT / "engine"), "-I", str(ROOT / "firmware/gateway/main"),
                    str(ROOT / "engine/replay_live.c"), "-lm", "-o", out], check=True)


def flows(d):
    F = []
    for f in sorted(Path(d).glob("gw-*.log")):
        for ln in f.read_bytes().decode(errors="replace").splitlines():
            p = ln.split(",")
            if len(p) > 18 and p[1] == "F":
                F.append(p)
    return F


def main(d, tau="0.442"):
    F, R, tmp = flows(d), runs(d), Path(tempfile.mkdtemp())
    for b in sorted(set(R.build) & set(FLAGS)):
        exe = str(tmp / b); build(b, exe)
        n = ms = md = 0
        for _, r in R[R.build == b].iterrows():
            rows = [p for p in F if r.t0 * 1000 <= int(p[0]) < r.t1 * 1000]
            out = subprocess.run([exe, tau], input="".join(",".join(p[2:13]) + "\n" for p in rows),
                                 capture_output=True, text=True, check=True).stdout.splitlines()
            assert len(out) == len(rows), (b, r.variant, r.rep)
            for p, o in zip(rows, out):
                o = o.split(",")
                n += 1; ms += abs(float(o[4]) - float(p[13])) > 1e-4
                md += o[7] != p[16]  # the log keeps the decision before a monitor build discards it
        print(f"{b}: {int((R.build == b).sum())} runs, {n} flows, score mismatches {ms}, block-decision mismatches {md}")


if __name__ == "__main__":
    main(*sys.argv[1:])
