"""PREREG RETRY-GATE: apply the block-removing retry gate to logged gateway decisions and count blocks per run.
Gate: suppress a block on flow f of key k iff f got no reply packet, every no-reply flow of k in the trailing 240 s
went to f's (dst, dport), and k had a flow to that (dst, dport) with reply bytes > 0 in the prior 600 s.
shortcut: applied to logged decisions, so traffic a suppressed block would have let through is not re-simulated.
Usage: python scripts/live/gate_replay.py <dir with fair_runs.txt + gw-*.log> <build> <variant> <key>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_check import flows  # noqa: E402


def runs(d, build, variant):
    cur, out = {}, []
    for ln in (Path(d) / "fair_runs.txt").read_text().splitlines():
        p = ln.split()
        if p[:1] != ["RUN"] or len(p) < 6 or p[1] != build or p[2] != variant:
            continue
        if p[4] == "start":
            cur[p[3]] = int(p[5])
        elif p[4] == "end" and p[3] in cur:
            out.append((p[3], cur.pop(p[3]), int(p[5])))
    return {r: (a, b) for r, a, b in out}  # last valid start/end per rep


V2 = "--v2" in sys.argv  # PREREG RETRY-GATE v2: no per-flow reply condition, all flows (not only no-reply) count


def gated(rows, i):
    """True if the block on rows[i] is suppressed. rows: F lines of one key, time-sorted."""
    p = rows[i]; t = int(p[0]); svc = (p[5], p[6])
    if not V2 and int(p[11]) > 0:
        return False
    for q in rows[:i + 1]:
        tq = int(q[0])
        if t - 240_000 <= tq <= t and (V2 or int(q[11]) == 0) and (q[5], q[6]) != svc:
            return False
    return any(t - 600_000 <= int(q[0]) < t and (q[5], q[6]) == svc and int(q[10]) > 0 for q in rows[:i])


def main(d, build, variant, key):
    F = flows(d)
    tot = [0, 0]
    for rep, (t0, t1) in sorted(runs(d, build, variant).items()):
        rows = sorted((p for p in F if p[3] == key and (t0 - 700) * 1000 <= int(p[0]) < t1 * 1000), key=lambda p: int(p[0]))
        inrun = [i for i, p in enumerate(rows) if int(p[0]) >= t0 * 1000]
        raw = [i for i in inrun if rows[i][16] == "1"]
        kept = [i for i in raw if not gated(rows, i)]
        tot[0] += bool(raw); tot[1] += bool(kept)
        print(f"{build} {variant} {rep}: blocks {len(raw)} -> {len(kept)} with gate; kept at "
              f"{[round(int(rows[i][0]) / 1000 - t0) for i in kept]}")
    print(f"runs with a block of {key}: {tot[0]} -> {tot[1]}")


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if a != "--v2"])
