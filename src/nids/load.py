"""Dataset loaders. Each returns a DataFrame of the original feature columns plus:
  y      fine label (attack class as named by the dataset)
  family attack family used for leave-one-family-out (DoS+DDoS merged into Flood)
  src    provenance (source file / official split) used for grouping and audits
No cleaning happens here; see clean.py. Results are cached as parquet in data/processed/.
"""
import io
import os
import warnings
import zipfile
from pathlib import Path

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
        t = pacsv.read_csv(f)
        cols = []
        for c, col in zip(t.column_names, t.columns):  # 20-27M rows: narrowest type that holds the column
            if pa.types.is_floating(col.type):
                col = col.cast(pa.float32())
            elif pa.types.is_integer(col.type) and "MILLISECONDS" not in c:
                lo, hi = pc.min_max(col).values()
                if lo.as_py() is not None and -2**31 < lo.as_py() and hi.as_py() < 2**31:
                    col = col.cast(pa.int32())
            elif pa.types.is_string(col.type) and c.strip() != "Attack":
                col = col.dictionary_encode()
            cols.append(col)
        df = pa.table(cols, names=[c.strip() for c in t.column_names]).to_pandas(split_blocks=True, self_destruct=True)
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


LOADERS = {
    "unsw_official": unsw_official, "unsw_raw": unsw_raw, "bot_iot": bot_iot,
    "ciciot_paper": ciciot_paper, "ciciot_csv": ciciot_csv, "ciciot_merged": ciciot_merged,
    **{n: (lambda n=n: _netflow_v3(n)) for n in NF_FILES},
}
