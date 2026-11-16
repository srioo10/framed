"""PREREG LIVE-FAIR (9d8e96b) / LIVE-FAIR-REP / LIVE-SOUND: score the bars per replicate from the Pi logs.
Usage: python fair_report.py DIR   (DIR holds fair_runs.txt, sub.log, gw-*.log copied from Pi #1)
Replicate r covers reps 5(r-1)+1.. (clean 3 of them, attacks 5; run_fair2.sh OFF = 5(r-1))."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from natenf_report import NAT, gw, pubs  # noqa: E402


def runs(d):
    rows, cur = [], {}
    for ln in (Path(d) / "fair_runs.txt").read_text().splitlines():
        p = ln.split()
        if p[:1] != ["RUN"] or len(p) < 6:
            continue
        k = tuple(p[1:4])
        if p[4] == "start":
            cur[k] = (int(p[5]), p[7] if len(p) > 7 else "")
        elif p[4] == "end" and k in cur:
            (t0, att), t1 = cur.pop(k), int(p[5])
            if att:  # INVALID runs (no attacker address) are excluded; their rerun counts
                rows.append({"build": k[0], "variant": k[1], "rep": int(k[2]), "t0": t0, "t1": t1, "att": att})
    R = pd.DataFrame(rows).drop_duplicates(["build", "variant", "rep"], keep="last")
    R["repl"] = (R.rep - 1) // 5 + 1
    return R.reset_index(drop=True)


def score(d):
    F, B, _ = gw(d)
    pt, ps = pubs(d)
    R = runs(d)
    out = []
    for _, r in R.iterrows():
        a, b = r.t0 * 1000, r.t1 * 1000
        m = (pt >= r.t0) & (pt < r.t1)
        s = ps[m]
        nb = B[(B.ms >= a) & (B.ms < b) & (B.ip == NAT)]
        ab = B[(B.ms >= a) & (B.ms < b) & (B.ip == r.att)]
        scans = {"E": [1020], "D": [300, 1020], "H": [1020], "DH": [300, 1020], "P": [1020]}.get(r.variant, [])
        hit = lambda bl: any(((bl.ms >= a + 1000 * x) & (bl.ms <= a + 1000 * (x + 120))).any() for x in scans)  # noqa: E731
        f = F[(F.ms >= a) & (F.ms < b)]
        out.append({**r.to_dict(), "nat_blocks": len(nb), "att_blocks": len(ab),
                    "pubs300": int((m & (pt < r.t0 + 300)).sum()), "missed": int(np.clip(np.diff(s) - 1, 0, None).sum()) if len(s) > 1 else 0,
                    "nat_hit": hit(nb), "att_hit": hit(ab), "flows": len(f),
                    "brk_blocks": int((B[(B.ms >= a) & (B.ms < b)].ip == BROKER).sum()),
                    "att_hit_pos": bool(((ab.ms >= a + 1020000) & (ab.ms < a + 1200000)).any()),
                    "att_mqtt300": int(((f.src == r.att) & (f.dport.astype(str) == "1883") & (f.ms < a + 300000)).sum())})
    return pd.DataFrame(out)


BROKER = "192.168.77.1"


def pos_bars(S):
    """PREREG LIVE-POS (ab3fe55): P1 attacker blocked in [1020, 1200) s in >= 4/5 P runs; P2 no sensor or broker
    block in any P/PC run; P3 no attacker block in PC runs."""
    v = S[S.build == "pos"]
    p, pc = v[v.variant == "P"], v[v.variant == "PC"]
    return {"P1": (int(p.att_hit_pos.sum()), len(p)), "P1_pass": int(p.att_hit_pos.sum()) >= 4,
            "P2_sensor_blocks": int(v.nat_blocks.sum()), "P2_broker_blocks": int(v.brk_blocks.sum()), "P2_runs": len(v),
            "P2_pass": int(v.nat_blocks.sum()) == 0 and int(v.brk_blocks.sum()) == 0,
            "P3": (int(pc.att_blocks.sum()), len(pc)), "P3_pass": int(pc.att_blocks.sum()) == 0}


def bars(S, build_veto="veto"):
    res = {}
    for g, X in S.groupby("repl"):
        o, v = X[X.build == "ord"], X[X.build == build_veto]
        ib = o[o.variant.isin(["A", "B"])]
        oc = o[o.variant == "clean"]
        f1 = (int(((ib.nat_blocks > 0) & (ib.missed > 0)).sum()), len(ib), int((oc.nat_blocks > 0).sum()), len(oc))
        vi = v[v.variant.isin(["clean", "A", "B"])]
        ve, vd = v[v.variant == "E"], v[v.variant == "D"]
        res[g] = {"F1": f1, "F1_pass": f1[0] >= 7 and f1[2] == 0 if len(o) else None,
                  "F2_nat_blocks": int(vi.nat_blocks.sum()), "F2_runs": len(vi), "F2_pass": int(vi.nat_blocks.sum()) == 0,
                  "F3": (int(ve.nat_hit.sum()), len(ve)), "F3_pass": int(ve.nat_hit.sum()) >= 4,
                  "F4": (int(vd.att_hit.sum()), len(vd), int((vd.nat_blocks > 0).sum())),
                  "F4_pass": int(vd.att_hit.sum()) >= 4 and int((vd.nat_blocks > 0).sum()) == 0}
    return res


def scan_bars(S):
    """PREREG LIVE-SCAN: S1 safety, S2 horizontal scan behind the NAT, S3 horizontal scan on its own key."""
    v = S[S.build == "scan"]
    vi, h, dh, e = (v[v.variant.isin(["clean", "A", "B"])], v[v.variant == "H"], v[v.variant == "DH"],
                    v[v.variant == "E"])
    return {"S1_nat_blocks": int(vi.nat_blocks.sum()), "S1_runs": len(vi), "S1_pass": int(vi.nat_blocks.sum()) == 0,
            "S2": (int(h.nat_hit.sum()), len(h)), "S2_pass": int(h.nat_hit.sum()) >= 4,
            "S3": (int(dh.att_hit.sum()), len(dh), int((dh.nat_blocks > 0).sum())),
            "S3_pass": int(dh.att_hit.sum()) >= 4 and int((dh.nat_blocks > 0).sum()) == 0,
            "E_vertical_reported": (int(e.nat_hit.sum()), len(e))}


if __name__ == "__main__":
    assert int(np.clip(np.diff(np.array([1, 2, 5])) - 1, 0, None).sum()) == 2
    d = sys.argv[1]
    S = score(d)
    S.to_csv(Path(d) / "fair_scored.csv", index=False)
    bad = S.pubs300 == 0  # PREREG LIVE-FAIR INVALID rule: the sensor publishes nothing in the first 300 s
    bad |= S.build.isin(["pos", "pos_mon"]) & ((S.att_mqtt300 == 0) | (S.att == NAT))  # LIVE-POS INVALID rules
    if bad.any():
        print("INVALID (no sensor publish in first 300 s):", S[bad][["build", "variant", "rep"]].to_dict("records"))
    S = S[~bad]
    if len(sys.argv) > 2 and sys.argv[2] == "pos":
        print("LIVE-POS", pos_bars(S))
    elif len(sys.argv) > 2 and sys.argv[2] == "scan":
        print("LIVE-SCAN", scan_bars(S))
    else:
        for g, r in bars(S, sys.argv[2] if len(sys.argv) > 2 else "veto").items():
            print("replicate", g, r)
