"""Dataset loaders. Each returns a DataFrame of the original feature columns plus:
  y      fine label (attack class as named by the dataset)
  family attack family used for leave-one-family-out (DoS+DDoS merged into Flood)
  src    provenance (source file / official split) used for grouping and audits
No cleaning happens here; see clean.py. Results are cached as parquet in data/processed/.
"""
import io
import os
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed"
DATA = Path(os.environ.get("NIDS_DATA_DIR", ROOT))  # where the downloaded files live

UNSW_ZIP = DATA / "OneDrive_1_10-23-2026.zip"
BOT_TRAIN = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
BOT_TEST = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
CIC_CSV_ZIP = DATA / "CSV.zip"  # CIC 2024 per-pcap CSVs, 39 features (the version ARLHNIDS used)
CIC_MERGED_ZIP = DATA / "archive(1).zip"  # original MERGED_CSV, 46 features + label
NF_DIR = Path(os.environ.get("NIDS_NF_DIR", ROOT / "data" / "raw" / "netflow_v3"))  # unpacked UQ NetFlow-v3 zips
NF_FILES = {"nf_unsw": "NF-UNSW-NB15-v3.csv", "nf_ton": "NF-ToN-IoT-v3.csv", "nf_bot": "NF-BoT-IoT-v3.csv",
            "nf_cic": "NF-CICIDS2018-v3.csv"}

# ARLHNIDS Table 4: one capture file per class (row counts verified exactly).
CIC_PAPER_FILES = {
    "DDoS": "DDoS-SYN_Flood/DDoS-SYN_Flood15",
    "Normal": "Benign_Final/BenignTraffic3",
    "Spoofing": "MITM-ArpSpoofing/MITM-ArpSpoofing1",
    "DoS": "DoS-HTTP_Flood/DoS-HTTP_Flood",
    "Mirai": "Mirai-udpplain/Mirai-udpplain24",
    "Brute Force": "DictionaryBruteForce/DictionaryBruteForce",
    "Web-Based": "Backdoor_Malware/Backdoor_Malware",
    "Recon": "Recon-PingSweep/Recon-PingSweep",
}


def _cached(name, build):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _read_zip_csv(zf, member, **kw):
    with zf.open(member) as f:
        return pd.read_csv(io.BytesIO(f.read()), engine="pyarrow", **kw)


def _downcast(df):
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    return df


# ---------------------------------------------------------------- families

def cic_family(label: str) -> str:
    s = label.lower()
    if s.startswith(("ddos", "dos")):
        return "Flood"
    if s.startswith("mirai"):
        return "Mirai"
    if s.startswith("recon") or s == "vulnerabilityscan":
        return "Recon"
    if s.startswith("mitm") or s == "dns_spoofing":
        return "Spoofing"
    if s == "dictionarybruteforce":
        return "BruteForce"
    if s.startswith("benign"):
        return "Benign"
    return "Web"  # SqlInjection, CommandInjection, XSS, Backdoor_Malware, Uploading_Attack, BrowserHijacking


UNSW_FAMILY = {
    "Normal": "Benign", "Generic": "Generic", "Exploits": "Exploits", "Fuzzers": "Fuzzers",
    "DoS": "DoS", "Reconnaissance": "Recon", "Analysis": "Analysis+Backdoor",
    "Backdoor": "Analysis+Backdoor", "Shellcode": "Shellcode+Worms", "Worms": "Shellcode+Worms",
}

CIC2018_FAMILY = {  # standard CSE-CIC-IDS2018 grouping
    "Benign": "Benign", "Bot": "Bot", "Infilteration": "Infiltration",
    "DoS_attacks-Hulk": "DoS", "DoS_attacks-GoldenEye": "DoS", "DoS_attacks-Slowloris": "DoS",
    "DoS_attacks-SlowHTTPTest": "DoS", "DDOS_attack-HOIC": "DDoS", "DDOS_attack-LOIC-UDP": "DDoS",
    "DDoS_attacks-LOIC-HTTP": "DDoS", "FTP-BruteForce": "BruteForce", "SSH-Bruteforce": "BruteForce",
    "Brute_Force_-Web": "Web", "Brute_Force_-XSS": "Web", "SQL_Injection": "Web",
}

BOT_FAMILY = {"DDoS": "Flood", "DoS": "Flood", "Reconnaissance": "Recon", "Theft": "Theft", "Normal": "Benign"}


# ---------------------------------------------------------------- UNSW-NB15

def unsw_official():
    """Official 175,341 / 82,332 train/test sets (ARLHNIDS Table 2). src = 'train' | 'test'."""
    def build():
        zf = zipfile.ZipFile(UNSW_ZIP)
        parts = []
        for split, member in [("train", "Training and Testing Sets/UNSW_NB15_training-set.csv"),
                              ("test", "Training and Testing Sets/UNSW_NB15_testing-set.csv")]:
            d = _read_zip_csv(zf, member)
            d.columns = [c.lstrip("﻿") for c in d.columns]
            d["src"] = split
            parts.append(d)
        df = pd.concat(parts, ignore_index=True).drop(columns=["id"])
        df["y"] = df.pop("attack_cat").str.strip()
        df["family"] = df["y"].map(UNSW_FAMILY)
        return _downcast(df)
    return _cached("unsw_official", build)


def unsw_raw():
    """Full 2.54M-record UNSW-NB15_1..4 with IPs and timestamps (for grouped splits)."""
    def build():
        zf = zipfile.ZipFile(UNSW_ZIP)
        names = [n.strip().lower().replace(" ", "")
                 for n in _read_zip_csv(zf, "NUSW-NB15_features.csv", encoding="latin-1")["Name"]]
        parts = []
        for i in range(1, 5):
            d = pd.read_csv(io.BytesIO(zf.open(f"UNSW-NB15_{i}.csv").read()), header=None,
                            names=names, low_memory=False, encoding="latin-1")
            d["src"] = f"UNSW-NB15_{i}"
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        cat = df.pop("attack_cat").fillna("Normal").astype(str).str.strip().replace({"Backdoors": "Backdoor"})
        df["y"] = cat.where(cat != "", "Normal")
        df["family"] = df["y"].map(UNSW_FAMILY)
        # ports have a few hex / '-' values in the raw files
        for c in ("sport", "dsport"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # blank = not applicable (non-HTTP / non-FTP flow); the official sets store these as 0
        for c in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
            blank = df[c].isna() | df[c].astype(str).str.strip().eq("")
            df[c] = pd.to_numeric(df[c].mask(blank, 0), errors="coerce")
        return _downcast(df)
    return _cached("unsw_raw", build)


# ---------------------------------------------------------------- NetFlow-v3 (Luay et al. 2025)

def _netflow_v3(name):
    """UQ NetFlow-v3 CSV (53 nProbe features + Label + Attack). y = Attack string; UNSW families merged as in
    unsw_raw, other datasets keep their Attack names as families."""
    def build():
        import pyarrow as pa
        import pyarrow.compute as pc
        from pyarrow import csv as pacsv
        f = next(NF_DIR.glob(f"*/*/data/{NF_FILES[name]}"))
        # stream the CSV (5 GB for nf_ton): cast each batch to its narrow type so the full float64 table never exists
        reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=1 << 26))
        names = [c.strip() for c in reader.schema.names]
        batches = []
        for b in reader:
            cols = []
            for c, col in zip(names, b.columns):
                if pa.types.is_floating(col.type):
                    col = col.cast(pa.float32())
                elif pa.types.is_string(col.type) and c != "Attack":
                    col = col.dictionary_encode()
                cols.append(col)
            batches.append(pa.record_batch(cols, names=names))
        t = pa.Table.from_batches(batches).unify_dictionaries()
        del batches
        cols = []
        for c, col in zip(t.column_names, t.columns):  # 20-27M rows: narrowest int type that holds the column
            if pa.types.is_integer(col.type) and "MILLISECONDS" not in c:
                lo, hi = pc.min_max(col).values()
                if lo.as_py() is not None and -2**31 < lo.as_py() and hi.as_py() < 2**31:
                    col = col.cast(pa.int32())
            cols.append(col)
        df = pa.table(cols, names=t.column_names).to_pandas(split_blocks=True, self_destruct=True)
        del t, cols
        df["y"] = df.pop("Attack").astype(str).str.strip()
        fam = {"nf_unsw": {**UNSW_FAMILY, "Benign": "Benign"}, "nf_cic": CIC2018_FAMILY}.get(name)
        df["family"] = df["y"].map(fam) if fam else df["y"]
        assert df["family"].notna().all(), set(df["y"][df["family"].isna()])
        df["src"] = name
        return _downcast(df)
    return _cached(name, build)


# ---------------------------------------------------------------- BoT-IoT

def bot_iot():
    """10-best Final subset, official train/test (ARLHNIDS Table 3). src = 'train' | 'test'."""
    def build():
        parts = []
        for split, path in [("train", BOT_TRAIN), ("test", BOT_TEST)]:
            d = pd.read_csv(path, engine="pyarrow")
            d["src"] = split
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["y"] = df.pop("category")
        df["family"] = df["y"].map(BOT_FAMILY)
        return _downcast(df)
    return _cached("bot_iot", build)


# ---------------------------------------------------------------- CICIoT2023

def _cic_csv_member_label(member):
    # CSV/<ClassFolder>/<file>.pcap.csv ; Benign_Final holds BenignTraffic*.pcap.csv
    folder = member.split("/")[1]
    return "BenignTraffic" if folder == "Benign_Final" else folder


def ciciot_paper():
    """Exactly the 8 capture files ARLHNIDS used (Table 4). y = paper class name."""
    def build():
        zf = zipfile.ZipFile(CIC_CSV_ZIP)
        parts = []
        for cls, stem in CIC_PAPER_FILES.items():
            d = _read_zip_csv(zf, f"CSV/{stem}.pcap.csv")
            d["y"] = cls
            d["src"] = stem.split("/")[1]
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["family"] = df["y"].map({"DDoS": "Flood", "DoS": "Flood", "Normal": "Benign",
                                    "Brute Force": "BruteForce", "Web-Based": "Web"}).fillna(df["y"])
        return _downcast(df)
    return _cached("ciciot_paper", build)


def ciciot_csv():
    """All per-pcap CSVs in CSV.zip (34 classes). src = capture file -> natural group for splits."""
    def build():
        zf = zipfile.ZipFile(CIC_CSV_ZIP)
        parts = []
        for m in zf.namelist():
            if not m.endswith(".pcap.csv"):
                continue
            raw = zf.read(m)
            # DoS-UDP_Flood7/8/9 end with one truncated line each; skip and report
            d = _downcast(pd.read_csv(io.BytesIO(raw), engine="pyarrow", on_bad_lines="skip"))
            dropped = raw.rstrip(b"\n").count(b"\n") - len(d)
            if dropped:
                warnings.warn(f"{m}: skipped {dropped} malformed row(s)")
            d["y"] = _cic_csv_member_label(m)
            d["src"] = m.split("/")[-1].removesuffix(".pcap.csv")
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["family"] = df["y"].map(cic_family)
        for c in ("y", "family", "src"):
            df[c] = df[c].astype("category")
        return df
    return _cached("ciciot_csv", build)


def ciciot_merged():
    """Original CICIoT2023 MERGED_CSV (169 parts, 46 features). src = part file (no capture id exists)."""
    def build():
        zf = zipfile.ZipFile(CIC_MERGED_ZIP)
        parts = []
        for m in sorted(zf.namelist()):
            if "/CICIoT2023/part-" not in m:
                continue
            d = _downcast(_read_zip_csv(zf, m))
            d["src"] = m.split("/")[-1][:10]
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["y"] = df.pop("label")
        df["family"] = df["y"].map(cic_family)
        for c in ("y", "family", "src"):
            df[c] = df[c].astype("category")
        return df
    return _cached("ciciot_merged", build)


# ---------------------------------------------------------------- CTU-13 / IoT-23 (NetFlow-v3 column names, PREREG §9 DISC)

RAW = ROOT / "data" / "raw"
PROTO = {"tcp": 6, "udp": 17, "icmp": 1}
CTU_BOTNET = {1: "Neris", 2: "Neris", 9: "Neris", 3: "Rbot", 4: "Rbot", 10: "Rbot", 11: "Rbot", 5: "Virut",
              13: "Virut", 6: "Menti", 7: "Sogou", 8: "Murlo", 12: "NSIS"}
IOT23_MALWARE = {1: "HideAndSeek", 3: "Muhstik", 8: "Hakai", 9: "Hajime", 17: "Kenjiro", 33: "Kenjiro", 20: "Torii",
                 21: "Torii", 36: "Okiru", 39: "IRCBot", 42: "Trojan", 60: "Gagfyt",
                 **{n: "Mirai" for n in (7, 34, 35, 43, 44, 48, 49, 52)}}
IOT23_ROWS = 300_000


def _port(s):
    return pd.to_numeric(s, errors="coerce").fillna(
        s.fillna("").astype(str).map(lambda v: int(v, 16) if v.startswith("0x") else 0)).astype(np.int32)


def _nf_frame(tag, t_ms, src, sport, dst, dport, proto, dur_ms, ib, ob, ip, op, fam, name):
    df = pd.DataFrame({"FLOW_START_MILLISECONDS": t_ms.astype(np.int64),
                       "IPV4_SRC_ADDR": tag + ":" + src.astype(str), "IPV4_DST_ADDR": tag + ":" + dst.astype(str),
                       "L4_SRC_PORT": _port(sport), "L4_DST_PORT": _port(dport),
                       "PROTOCOL": proto.astype(str).str.lower().map(PROTO).fillna(0).astype(np.int16),
                       "FLOW_DURATION_MILLISECONDS": dur_ms.astype(np.float32),
                       "IN_BYTES": ib.astype(np.float32), "OUT_BYTES": ob.astype(np.float32),
                       "IN_PKTS": ip.astype(np.float32), "OUT_PKTS": op.astype(np.float32)})
    df["y"] = df["family"] = fam
    df["src"] = name
    return df


def ctu13():
    """CTU-13 binetflow: From-Botnet = attack (family by scenario botnet), Normal = Benign; To-Botnet and
    Background dropped. Host key = scenario:IP. OUT_PKTS is a reply indicator (Argus gives no packet split)."""
    def build():
        out = []
        for d in sorted((RAW / "ctu13" / "CTU-13-Dataset").iterdir(), key=lambda p: int(p.name)):
            f = next(d.glob("*.binetflow"))
            x = pd.read_csv(f, dtype=str)
            lab = x["Label"].str
            bot, ok = lab.contains("From-Botnet"), lab.contains("Normal")
            x = x[bot | ok]
            fam = np.where(bot[bot | ok], CTU_BOTNET[int(d.name)], "Benign")
            num = lambda c: pd.to_numeric(x[c], errors="coerce").fillna(0)  # noqa: E731
            ob = num("TotBytes") - num("SrcBytes")
            t = pd.to_datetime(x["StartTime"], format="%Y/%m/%d %H:%M:%S.%f").astype("datetime64[ms]").astype("int64")
            out.append(_nf_frame(f"s{d.name}", t, x["SrcAddr"], x["Sport"], x["DstAddr"], x["Dport"], x["Proto"],
                                 num("Dur") * 1000, num("SrcBytes"), ob, num("TotPkts"), (ob > 0).astype(float),
                                 fam, "nf_ctu13"))
        return pd.concat(out, ignore_index=True)
    return _cached("nf_ctu13", build)


def iot23():
    """IoT-23 conn.log.labeled, first IOT23_ROWS rows per capture; Malicious = the capture's malware family."""
    names = ["ts", "uid", "oh", "op", "rh", "rp", "proto", "service", "duration", "ob", "rb", "state", "lo", "lr",
             "missed", "history", "opk", "oipb", "rpk", "ripb", "tunnel", "label", "detail"]

    def build():
        out = []
        for f in sorted((RAW / "iot23").rglob("conn.log.labeled")):
            cap = next(q.name for q in f.parents if "Capture-" in q.name)
            n = int(re.search(r"Capture-(\d+)-", cap).group(1))
            x = pd.read_csv(f, sep=r"\s+", comment="#", header=None, names=names, nrows=IOT23_ROWS, dtype=str,
                            engine="python")
            mal = x["label"].str.lower().eq("malicious")
            fam = np.where(mal, IOT23_MALWARE.get(n, "Unknown") if "Honeypot" not in cap else "Benign", "Benign")
            num = lambda c: pd.to_numeric(x[c], errors="coerce").fillna(0)  # noqa: E731
            out.append(_nf_frame(f"c{'h' if 'Honeypot' in cap else ''}{n}", num("ts") * 1000, x["oh"], x["op"],
                                 x["rh"], x["rp"], x["proto"], num("duration") * 1000, num("ob"), num("rb"),
                                 num("opk"), num("rpk"), fam, "nf_iot23"))
        return pd.concat(out, ignore_index=True)
    return _cached("nf_iot23", build)


UGR_WEEKS = ("june_week3", "june_week4", "july_week5", "august_week1", "august_week2")
_UGR_OCT = {"42.219.150": (246, 247, 243, 242, 241), "42.219.156": (30, 31, 29, 28, 27),
            "42.219.158": (16, 17, 18, 19, 21), "42.219.152": (20, 21, 22, 23, 18), "42.219.154": (69, 68, 70, 71, 66)}
UGR_KEY = np.array([f"{p}.{o}" for p, os_ in _UGR_OCT.items() for o in os_])  # attackers A1-A5 + victims (Table V)
UGR_HASH = "ugr16-fwd-0001xx"  # 16-byte key: the seeded source-IP sample of PREREG §9 FWD


def _ugr_hash(ips):
    return pd.util.hash_array(np.asarray(ips, dtype=object), hash_key=UGR_HASH)


def ugr16():
    """UGR'16 forward-time weeks (PREREG §9 FWD): 0.1% hash sample of source IPs, every flow touching an attacker or
    victim, every flow of a 5% hash sample of sources with a real-anomaly label; blacklist dropped, background =
    Benign. Unidirectional NetFlow: OUT_BYTES = 0, OUT_PKTS = 1 if TCP ACK is set or not TCP. src = week name."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = ["src_ip", "dst_ip", "ts_start", "label_attack", "l4_proto", "src_port", "dst_port", "duration", "packets",
            "bytes", "tcp_flags"]

    def build():
        files = {w: pq.ParquetFile(RAW / "ugr16" / f"{w}.parquet") for w in UGR_WEEKS}
        anom = set()  # pass 1: sources of real-anomaly labels
        for f in files.values():
            for i in range(f.num_row_groups):
                g = f.read_row_group(i, columns=["src_ip", "label_attack"])
                m = pc.starts_with(pc.fill_null(g["label_attack"], ""), "anomaly")
                if pc.any(m).as_py():
                    anom.update(pc.unique(g["src_ip"].filter(m)).to_pylist())
        anom = np.array(sorted(anom), dtype=object)
        anom = set(anom[_ugr_hash(anom) % 20 == 0])
        out = []
        for w, f in files.items():
            for i in range(f.num_row_groups):
                g = f.read_row_group(i, columns=cols)
                src = g["src_ip"].to_numpy(zero_copy_only=False)
                dst = g["dst_ip"].to_numpy(zero_copy_only=False)
                s_, d_ = pd.Series(src), pd.Series(dst)
                keep = (_ugr_hash(src) % 1000 == 0) | s_.isin(UGR_KEY).to_numpy() | d_.isin(UGR_KEY).to_numpy() \
                    | s_.isin(anom).to_numpy()
                x = g.filter(pa.array(keep)).to_pandas()
                lab = x["label_attack"].fillna("Benign").astype(str)
                x = x[lab != "blacklist"]
                lab = lab[lab != "blacklist"].str.removeprefix("anomaly-")
                tcp = x["l4_proto"].to_numpy() == 6
                ack = (x["tcp_flags"].fillna(0).to_numpy(np.int64) & 16) > 0
                df = _nf_frame("u", x["ts_start"] * 1000, x["src_ip"], x["src_port"], x["dst_ip"], x["dst_port"],
                               x["l4_proto"].map({6: "tcp", 17: "udp", 1: "icmp"}).fillna(""),
                               x["duration"].fillna(0) * 1000, x["bytes"], np.zeros(len(x)), x["packets"],
                               (~tcp | ack).astype(np.float32), lab.to_numpy(), w)
                out.append(df)
            print(f"ugr16 {w}: {sum(map(len, out))} rows so far", flush=True)
        df = pd.concat(out, ignore_index=True)
        df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].str.removeprefix("u:")
        df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].str.removeprefix("u:")
        return df.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    return _cached("nf_ugr16", build)


LOADERS = {
    "nf_ctu13": ctu13, "nf_iot23": iot23, "nf_ugr16": ugr16,
    "unsw_official": unsw_official, "unsw_raw": unsw_raw, "bot_iot": bot_iot,
    "ciciot_paper": ciciot_paper, "ciciot_csv": ciciot_csv, "ciciot_merged": ciciot_merged,
    **{n: (lambda n=n: _netflow_v3(n)) for n in NF_FILES},
}
