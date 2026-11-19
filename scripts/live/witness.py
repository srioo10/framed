"""FRAME-ALARM: search a framing witness for every alarmed-but-unblocked flow of a key, from logged gateway flows.
A witness for flow f of key K is a set W of <= m earlier flows of K in [t_f - 240 s, t_f) such that replaying the run
without W through the deployed decision path (engine/replay_live.c) gives score(f) <= tau. W names the suspect flows.
Policies (model-free, fixed order, first hit wins): ranked (failed+new peer+new port), failed (no reply), new_port,
new_peer (suspects are context writes only). Deletion counts connections: all records of up to m distinct 5-tuples.
Usage: python scripts/live/witness.py <dir> <build> <tau> [key] [m]
Prints one line per gap flow (run, t, score, policy, n_named, n_named_injected) and per-run summary lines."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_check import build, flows  # noqa: E402
from fair_report import runs  # noqa: E402

W_MS = 240_000
V2 = "--v2" in sys.argv
if V2: sys.argv.remove("--v2")
def is_inj(dst):  # ground truth: the attacker's targets are lab peers 192.168.100-199.x (sk_inject.py asserts this range)
    o = dst.split(".")
    return dst.startswith("192.168.") and 100 <= int(o[2]) <= 199


def injected(p):
    return is_inj(p[5])


def replay(exe, tau, rows):
    out = subprocess.run([exe, tau], input="".join(",".join(p[2:13]) + "\n" for p in rows),
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(rows)
    return [o.split(",") for o in out]


def clears(exe, tau, rows, i, tup):
    """True if deleting every earlier record of the 5-tuples in tup gives score(rows[i]) <= tau."""
    S = {j for j in range(i) if tuple(rows[j][3:8]) in tup}
    keep = [p for j, p in enumerate(rows) if j not in S]
    return float(replay(exe, tau, keep)[i - len(S)][4]) <= float(tau)


def policies(rows, i, key, m):
    """Candidate deletion sets (indices into rows) for gap flow i; each of size <= m."""
    t = int(rows[i][2])
    C = [j for j in range(i) if t - W_MS <= int(rows[j][2]) < t and key in (rows[j][3], rows[j][5])]
    if V2:  # FRAME-WITNESS v2: a host's own connections to the alarmed flow's service are its behaviour, not suspects
        C = [j for j in C if (rows[j][5], rows[j][6]) != (rows[i][5], rows[i][6])]
    last_peer, last_port, newp, newo = {}, {}, set(), set()
    for j in range(i):  # novelty as the gateway's tables: not seen by the key in the prior 120 s
        if key not in (rows[j][3], rows[j][5]):
            continue
        tj, peer, port = int(rows[j][2]), rows[j][5], rows[j][6]
        if tj - last_peer.get(peer, -10**12) >= 120_000: newp.add(j)
        if tj - last_port.get((peer, port), -10**12) >= 120_000: newo.add(j)
        last_peer[peer] = tj; last_port[(peer, port)] = tj
    failed = {j for j in C if rows[j][11] == "0"}
    rank = sorted([j for j in C if j in failed or j in newp or j in newo],
                  key=lambda j: -((j in failed) + (j in newp) + (j in newo)))
    def first(L):  # the first m distinct 5-tuples of L, in L's order
        out, seen = [], set()
        for j in L:
            t5 = tuple(rows[j][3:8])
            if t5 not in seen and len(seen) < m: seen.add(t5); out.append(j)
        return out
    rev = lambda L: first(L[::-1])  # noqa: E731  (most recent first)
    return [("ranked", first(rank)), ("failed", rev([j for j in C if j in failed])),
            ("new_port", rev([j for j in C if j in newo])), ("new_peer", rev([j for j in C if j in newp]))]
    # suspects are context writes only (failed, new peer, new port): the way an in-budget injector moves context


def no_attacker_runs(d, b):
    """LIVE-FAULT runs have no attacker address by design, which fair_report.runs drops as INVALID."""
    import pandas as pd
    cur, out = {}, {}
    for ln in (Path(d) / "fair_runs.txt").read_text().splitlines():
        p = ln.split()
        if p[:2] != ["RUN", b] or len(p) < 6:
            continue
        if p[4] == "start": cur[p[3]] = int(p[5])
        elif p[4] == "end" and p[3] in cur: out[p[3]] = (cur.pop(p[3]), int(p[5]))
    return pd.DataFrame([{"build": b, "variant": "F", "rep": int(k), "t0": a, "t1": z} for k, (a, z) in out.items()])


def main(d, b, tau, key="192.168.4.2", m="10"):
    m = int(m); F = flows(d)
    R = runs(d) if b != "fault" else no_attacker_runs(d, b)
    exe = str(Path(tempfile.mkdtemp()) / b); build({"col": "pos", "amn2": "pos", "fault": "pos"}.get(b, b), exe)  # pos + enforcement-only flags
    for _, r in R[R.build == b].iterrows():
        rows = [p for p in F if r.t0 * 1000 <= int(p[0]) < r.t1 * 1000]
        base = replay(exe, tau, rows)
        bad = sum(abs(float(o[4]) - float(p[13])) >= 1e-4 for o, p in zip(base, rows))
        if bad or not rows:  # replay must reproduce the logged scores, else the run is not analysed
            print("SKIP", b, r.variant, r.rep, "flows", len(rows), "score mismatches", bad); continue
        gaps = [i for i, o in enumerate(base) if rows[i][3] == key and o[6] == "1" and o[7] == "0"]
        for i in [i for i, o in enumerate(base) if rows[i][3] == key and o[7] == "1"]:  # W5 soundness on blocks
            for name, S0 in policies(rows, i, key, m):
                tup = {tuple(rows[j][3:8]) for j in S0}
                nrec = sum(1 for j in range(i) if tuple(rows[j][3:8]) in tup)
                if S0 and clears(exe, tau, rows, i, tup):
                    print("BW", r.variant, r.rep, rows[i][2], name, len(tup), nrec); break
        n_alert = named = named_inj = 0
        for i in gaps:
            hit = None
            for name, S0 in policies(rows, i, key, m):
                if not S0:
                    continue
                # a flow is a connection: delete every earlier record of the <= m chosen 5-tuples (gateway logs
                # long-lived connections as several 'active' records)
                tup = [tuple(rows[j][3:8]) for j in S0]
                if clears(exe, tau, rows, i, set(tup)):
                    for t5 in list(tup):  # prune to a 1-minimal witness, fixed order
                        if clears(exe, tau, rows, i, set(tup) - {t5}):
                            tup.remove(t5)
                    hit = (name, [j for j in range(i) if tuple(rows[j][3:8]) in set(tup)]); break
            if hit:
                n_alert += 1; ni = sum(injected(rows[j]) for j in hit[1])
                named += len(hit[1]); named_inj += ni
            T = {tuple(rows[j][3:8]) for j in hit[1]} if hit else set()  # named connections
            print("G", r.variant, r.rep, rows[i][2], "inj" if injected(rows[i]) else "real", rows[i][5], rows[i][6],
                  base[i][4], hit[0] if hit else "-", len(T), sum(is_inj(t[2]) for t in T))
        n_inj = sum(injected(p) for p in rows)
        print("R", b, r.variant, r.rep, "gaps", len(gaps), "alerts", n_alert, "named", named, "named_inj", named_inj,
              "injected_flows", n_inj)


if __name__ == "__main__":
    main(*sys.argv[1:])
