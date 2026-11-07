"""SLIPS-FRAME (PREREG §9): frame benign hosts in an unmodified Stratosphere Slips with injected flows.

export: writes one Zeek conn.log directory per run (clean, the 8 src-key patterns at m = 10, and a fidelity run of
attack sources) from the K1 victims' stream (spoof_attack.victims, seed-1 test fold) plus spoof_attack.inject flows.
score: parses each run's Slips alerts.json and writes results/cert/slips_frame_{ds}.json.
Usage: .venv/bin/python scripts/slips_frame.py export nf_ctu13 OUTDIR
       (run Slips on OUTDIR/<run>/zeek, output to OUTDIR/<run>/slips)
       .venv/bin/python scripts/slips_frame.py score nf_ctu13 OUTDIR
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spoof_attack as sa  # noqa: E402
from spoof_attack import ROOT  # noqa: E402

M, DAY = 10, 86_400_000
PATTERNS = list(itertools.product((0, 1), (0, 1), (0, 1)))  # peer, port, fail
FIELDS = ("ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service duration orig_bytes resp_bytes conn_state "
          "local_orig local_resp missed_bytes history orig_pkts orig_ip_bytes resp_pkts resp_ip_bytes tunnel_parents")
TYPES = ("time string addr port addr port enum string interval count count string bool bool count string count count "
         "count count set[string]")
UNSET = ["service", "local_orig", "local_resp", "missed_bytes", "tunnel_parents"]  # not in NetFlow; written as '-'


ADDR = {}  # tagged address -> unique synthetic IP, so CTU-13 scenarios never merge into one Slips profile


def plain(ip):
    if ip in ADDR:
        return ADDR[ip]
    return ip.split(":", 1)[-1]  # untagged address passes through (UGR'16); bare tag strip as a fallback


def build_addr_map(*frames):
    """Assign every scenario-tagged address (contains ':') a distinct 100.64.x.y IP (CGNAT space, valid for Zeek)."""
    ADDR.clear()
    tagged = set()
    for df in frames:
        for col in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
            tagged |= {a for a in df[col].unique() if ":" in str(a)}
    assert len(tagged) <= 65536, "more tagged hosts than the 100.64.0.0/16 block holds"
    for i, a in enumerate(sorted(tagged)):
        ADDR[a] = f"100.64.{i >> 8}.{i & 255}"


def write_conn(df, d):
    d.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    proto = df.PROTOCOL.map({6: "tcp", 17: "udp", 1: "icmp"}).fillna("tcp")
    oneway = df.OUT_PKTS.to_numpy() == 0
    st = np.where(oneway, "S0", "SF")
    hist = np.where(oneway, "S", "ShADadFf")
    cols = [(df.FLOW_START_MILLISECONDS / 1000).map("{:.6f}".format), pd.Series([f"C{i}" for i in range(len(df))]),
            df.IPV4_SRC_ADDR.map(plain), df.L4_SRC_PORT.astype(int), df.IPV4_DST_ADDR.map(plain),
            df.L4_DST_PORT.astype(int), proto, "-", (df.FLOW_DURATION_MILLISECONDS / 1000).map("{:.6f}".format),
            df.IN_BYTES.astype(np.int64), df.OUT_BYTES.astype(np.int64), st, "-", "-", "-",
            np.where(proto == "tcp", hist, "-"), df.IN_PKTS.astype(np.int64), df.IN_BYTES.astype(np.int64),
            df.OUT_PKTS.astype(np.int64), df.OUT_BYTES.astype(np.int64), "(empty)"]
    out = pd.DataFrame({i: (c.to_numpy() if hasattr(c, "to_numpy") else c) for i, c in enumerate(cols)})
    with open(d / "conn.log", "w") as f:
        f.write("#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n"
                f"#fields\t{FIELDS.replace(' ', chr(9))}\n#types\t{TYPES.replace(' ', chr(9))}\n")
        out.to_csv(f, sep="\t", header=False, index=False)
        f.write("#close\t-\n")
    return len(out)


def export(ds, outdir, days=None):
    raw = pd.read_parquet(sa.load.CACHE / f"{ds}.parquet")
    bank = sa.sb.Bank(ds, sa.SEED, "host", 40000, 200000)
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    rows, key = sa.victims(bank, z)
    if days:  # validation only: keep the first `days` victim host-days to bound Slips runtime (NOT the registered run)
        keep_days = np.array(sorted(np.unique(key)))[:days]
        rows = rows[np.isin(key, keep_days)]
    V = raw.iloc[bank.kept[rows]]
    vday = set(zip(V.IPV4_SRC_ADDR, V.FLOW_START_MILLISECONDS // DAY))  # keyed on the tagged (unique) address
    day = raw.FLOW_START_MILLISECONDS // DAY
    ips = set(V.IPV4_SRC_ADDR)
    cand = raw[raw.IPV4_SRC_ADDR.isin(ips) | raw.IPV4_DST_ADDR.isin(ips)]
    dc = day[cand.index]
    keep = [(s, d) in vday or (t, d) in vday for s, t, d in zip(cand.IPV4_SRC_ADDR, cand.IPV4_DST_ADDR, dc)]
    S = cand[keep]  # every flow naming a victim on a victim day (the stream Slips sees for those profiles)
    build_addr_map(S, V)  # tagged -> unique IP; now plain() is 1-1 on victims, so a profile == one victim host
    pv = {(plain(i), int(d)) for i, d in vday}
    assert len(pv) == len(vday), "address remap collided two victim host-days"
    Vs = V.sort_values("FLOW_START_MILLISECONDS", kind="stable")
    meta = {"ds": ds, "victim_days": sorted([i, int(d)] for i, d in pv), "stream_flows": len(S), "runs": {},
            "unset_fields": UNSET}
    outdir = Path(outdir)
    runs = [("clean", None)] + [(f"p{p}{po}{f}", (p, po, f)) for p, po, f in PATTERNS]
    for name, pat in runs:
        if pat is None:
            df = S
        else:
            peer, port, fail = pat
            inj = sa.inject(Vs, "src", M, peer, port, fail, 0)  # src-key injection; bytes left at NetFlow default
            inj = inj.reindex(columns=raw.columns)
            inj["PROTOCOL"], inj["OUT_BYTES"], inj["IN_PKTS"], inj["IN_BYTES"] = 6, 0, 1, 0
            inj["FLOW_DURATION_MILLISECONDS"], inj["L4_SRC_PORT"] = 0.0, 40000
            df = pd.concat([S, inj], ignore_index=True)
        n = write_conn(df, outdir / name / "zeek")
        meta["runs"][name] = {"flows": n, "injected": 0 if pat is None else int(n - len(S))}
    # fidelity: does Slips alert on the dataset's real attack sources? (reported, not a rule)
    atk = raw[raw.family != "Benign"]
    fsrc = atk.IPV4_SRC_ADDR.value_counts().head(200).index
    fid = raw[raw.IPV4_SRC_ADDR.isin(set(fsrc)) | raw.IPV4_DST_ADDR.isin(set(fsrc))]
    meta["runs"]["fidelity"] = {"flows": write_conn(fid, outdir / "fidelity" / "zeek"),
                                "attack_srcs": sorted({plain(i) for i in fsrc})}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{ds}: {len(meta['runs'])} runs, {len(pv)} victim host-days -> {outdir}")


import calendar  # noqa: E402
import re  # noqa: E402

# slips.log: "... IP <ip> detected as malicious in timewindow N (start YYYY/MM/DD HH:MM:SS, stop ...) ..."
BLOCK_RE = re.compile(r"IP (\d+\.\d+\.\d+\.\d+)\s+detected as malicious in timewindow \d+ "
                      r"\(start (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", re.I)


def blocked_host_days(slips_dir):
    """The block signal as (ip, epoch_day): Slips declared ip malicious in a timewindow whose start falls on that
    day (the aggregated alert that drives blocking, in slips.log). Slips echoes the flow's own UTC time. None if no
    slips.log. Per-flow evidence is NOT a block."""
    f = Path(slips_dir) / "slips.log"
    if not f.exists():
        return None
    out = set()
    for line in open(f, errors="replace"):
        m = BLOCK_RE.search(line)
        if m:
            t = calendar.timegm(__import__("time").strptime(m.group(2), "%Y/%m/%d %H:%M:%S"))
            out.add((m.group(1), t * 1000 // 86_400_000))
    return out


def alerted_ips(slips_dir):
    d = blocked_host_days(slips_dir)
    return None if d is None else {ip for ip, _ in d}


def evidence_ips(slips_dir):
    """Secondary (reported, not the rule): source IPs with any evidence event (alerts.json, IDEA format)."""
    f = Path(slips_dir) / "alerts" / "alerts.json"
    if not f.exists():
        return set()
    ips = set()
    for line in open(f, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        for s in o.get("Source", []):
            if s.get("IP"):
                ips.add(s["IP"])
    return ips


def score(ds, outdir):
    outdir = Path(outdir)
    meta = json.loads((outdir / "meta.json").read_text())
    vdays = {(i, int(d)) for i, d in meta["victim_days"]}  # (victim ip, epoch day)
    victims = {i for i, _ in vdays}
    res = {"ds": ds, "n_victim_host_days": len(vdays), "runs": {}, "newly_framed": {}}
    for name in meta["runs"]:
        bhd = blocked_host_days(outdir / name / "slips")
        if bhd is None:
            res["runs"][name] = "NO_OUTPUT"
            continue
        if name == "fidelity":
            srcs = set(meta["runs"]["fidelity"]["attack_srcs"])
            ips = {ip for ip, _ in bhd}
            res["runs"]["fidelity"] = {"attack_srcs_blocked": round(len(ips & srcs) / max(len(srcs), 1), 4),
                                       "n": len(srcs)}
        else:
            res["runs"][name] = round(len(vdays & bhd) / max(len(vdays), 1), 4)  # share of victim host-days BLOCKED
    # newly-framed: victim host-days blocked under a pattern that were NOT blocked clean (the attack's true effect)
    clean_bhd = blocked_host_days(outdir / "clean" / "slips")
    if clean_bhd is not None:
        clean_vd = vdays & clean_bhd
        for name in meta["runs"]:
            if name.startswith("p"):
                b = blocked_host_days(outdir / name / "slips")
                if b is not None:
                    res["newly_framed"][name] = round(len((vdays & b) - clean_vd) / max(len(vdays), 1), 4)
    clean = res["runs"].get("clean")
    pats = {k: v for k, v in res["runs"].items() if k.startswith("p") and isinstance(v, float)}
    if isinstance(clean, float) and pats:
        worst = max(pats.values())
        res["clean"], res["worst_pattern"] = clean, worst
        res["worst_newly_framed"] = max(res["newly_framed"].values()) if res["newly_framed"] else None
        res["framable"] = bool(worst >= 2 * clean and worst - clean >= 0.05)  # SHARED-KEY bar (PREREG SLIPS-FRAME)
    (ROOT / "results" / "cert" / f"slips_frame_{ds}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd, ds, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == "export":
        export(ds, outdir, int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        score(ds, outdir)
