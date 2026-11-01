"""CONF on Kaggle (PREREG §9 CONF): per dataset, data/conf/{ds} goes up as a private dataset (bigp-conf-{ds}) and its
kernels start at once: clean (conf_clean.py, A / B / RF / XGB, seeds 2-4), baselines (conf_baselines.py), k5b (k5b_net.py, seeds 2-4), trees
(k5_trees.py RF / XGB plain + box-augmented, one kernel per seed), groot (k5_trees.py --models groot, seeds 2-4).
Code from the bigp-k5code dataset (pushed by `push`). GPU sessions are used as extra CPU slots when --gpu.
Each kernel prints md5s of its pairs and copies its outputs to /kaggle/working.
Usage: .venv/bin/python scripts/kaggle_conf.py push | data nf_ctu13 [--gpu kind ...] | status | pull
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import groot

from kaggle_bowl import KAGGLE, ROOT, STAGE, sh, user

DS = ["nf_ctu13", "nf_ugr16", "nf_ton", "nf_cic"]
CODE = sorted(p.name for p in (ROOT / "scripts").glob("*.py"))  # all scripts: the attack kernel needs many
GROOT = Path(groot.__file__).parent  # no internet on Kaggle: the package ships with the code
SEEDS = (2, 3, 4)
KERNEL = '''import glob, hashlib, os, shutil, subprocess, sys
code = os.path.dirname(glob.glob("/kaggle/input/**/conf_clean.py", recursive=True)[0])
nids = os.path.dirname(glob.glob("/kaggle/input/**/nids/models.py", recursive=True)[0])
conf = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/{ds}/pair_0.npz", recursive=True)[0]))
shutil.copytree(code, "/kaggle/working/w/scripts", dirs_exist_ok=True)
shutil.copytree(os.path.dirname(nids), "/kaggle/working/w/src", dirs_exist_ok=True)  # nids + groot
subprocess.run([sys.executable, "-c", "import torch, sklearn, xgboost, scipy; print('versions', torch.__version__, "
                "sklearn.__version__, xgboost.__version__, scipy.__version__)"])
for f in sorted(glob.glob(f"{conf}/{ds}/pair_*.npz")):
    print("md5", "{ds}/" + os.path.basename(f), hashlib.md5(open(f, "rb").read()).hexdigest(), flush=True)
pairs = ["{ds}/" + os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{conf}/{ds}/pair_*.npz"))]
env = dict(os.environ, PYTHONPATH="/kaggle/working/w/src", OMP_NUM_THREADS="1", BOWL_DATASETS="{ds}")
S = "/kaggle/working/w/scripts/"
for cmd in {cmds}:
    print("RUN", cmd, flush=True)
    r = subprocess.run([sys.executable, S + cmd[0], *cmd[1:], "--arena", conf], env=env)
    print("RC", r.returncode, cmd, flush=True)
for f in glob.glob("/kaggle/working/w/results/**/*.*", recursive=True):
    shutil.copy(f, "/kaggle/working/")
'''


def cmds(kind, ds):
    if kind == "clean":
        return [["conf_clean.py", "--workers", "4", "--out", f"clean_{ds}.jsonl"]]
    if kind == "baselines":
        return [["conf_baselines.py", "--workers", "4", "--out", f"baselines_{ds}.jsonl"]]
    if kind == "k5b":
        return [["k5b_net.py", "--workers", "4", "--seed", str(s), "--out", f"conf_k5b_{ds}_s{s}.csv"] for s in SEEDS]
    if kind.startswith("groot"):  # groot = seeds 2-4 in one kernel; groot2 / groot3 / groot4 = one seed each
        return [["k5_trees.py", "--workers", "4", "--models", "groot", "--seed", str(s), "--out",
                 f"conf_groot_{ds}_s{s}.csv"] for s in (SEEDS if kind == "groot" else (int(kind[-1]),))]
    if kind == "attack":  # ToN: CONF C4 + C3
        return [["spoof_attack.py", ds], ["gate_attack.py", ds, "--tau-day", "--genome", B]]
    if kind == "leank1":  # memory-lean ToN C4, guarded by the bit-exactness test
        return [["test_lean_hostctx.py", "nf_ctu13", "nf_ugr16"], ["spoof_attack.py", ds]]
    if kind == "leanc3":  # memory-lean ToN C3, guarded by the test on CTU-13
        return [["test_lean_hostctx.py", "nf_ctu13"], ["gate_attack.py", ds, "--tau-day", "--genome", B]]
    if kind == "c3":
        return [["gate_attack.py", ds, "--tau-day", "--genome", B]]
    K = ["gate_attack.py", ds, "--tau-day", "--genome", B]  # PREREG §9 K16 / K17 / K18
    if kind == "k16":  # K19 throttle run (carries the K16 rows)
        return [K + ["--two-channel", "--throttle", "1", "2", "5", "10", "20", "50", "--tag", "k19"]]
    if kind == "k17":  # K19 pattern run (carries the K17 rows)
        return [K + ["--two-channel", "--m-attack", "10", "50", "100", "--random", "8", "--no-bytes-grid", "--tag", "k19"]]
    if kind == "k18":
        return [K + ["--random", "8", "--no-bytes-grid", "--at", "--tag", "k18"]]
    if kind == "k14":  # PREREG §9 K14 + K15
        return [["spoof_attack.py", ds, "--baselines"], ["ips_point.py", ds]]
    s = int(kind[-1])  # trees2 / trees3 / trees4
    return [["k5_trees.py", "--workers", "4", "--seed", str(s), "--out", f"conf_trees_{ds}_s{s}.csv"]]


KINDS = ["clean", "baselines", "k5b", "trees2", "trees3", "trees4", "groot", "groot2", "groot3", "groot4", "attack", "c3", "k14", "leank1", "leanc3", "k16", "k17", "k18"]
B = ('{"family": "tabm", "enc": "raw", "e": 4, "w": 32, "depth": 2, "sn": 0, "act": "relu", "members": 8, "head": "nlog", '
     '"kd": 0.9, "syn": 0.5, "clip": 1, "cert": 4, "gate": 1, "rdet": 2, "ls": 0.0, "focal": 0.0, "lr": 0.003}')
# CONF C4 (K1 on ToN) and C3 on ToN (B, K4 at tau_day): need data/processed and the bowl champions, not the arena
ATTACK = '''import glob, os, shutil, subprocess, sys
code = os.path.dirname(glob.glob("/kaggle/input/**/conf_clean.py", recursive=True)[0])
nids = os.path.dirname(glob.glob("/kaggle/input/**/nids/models.py", recursive=True)[0])
procs = sorted({os.path.dirname(f) for f in glob.glob("/kaggle/input/**/*_hostmeta.npz", recursive=True)})
bowl = os.path.dirname(glob.glob("/kaggle/input/**/round*/death.json", recursive=True)[0])
W = "/tmp/w"  # outside /kaggle/working so the kernel output holds only results, not the 1 GB data copy
shutil.copytree(code, W + "/scripts", dirs_exist_ok=True)
shutil.copytree(os.path.dirname(nids), W + "/src", dirs_exist_ok=True)
for proc in procs:  # every mounted bigp-proc-* dataset (the lean kinds mount several)
    shutil.copytree(proc, W + "/data/processed", dirs_exist_ok=True)
shutil.copytree(os.path.dirname(bowl), W + "/results/bowl", dirs_exist_ok=True)
os.makedirs(W + "/results/cert", exist_ok=True)  # gate_attack.py writes there without creating it
env = dict(os.environ, PYTHONPATH=W + "/src", OMP_NUM_THREADS="4", NIDS_LEAN="1")
for cmd in {cmds}:
    print("RUN", cmd, flush=True)
    r = subprocess.run([sys.executable, W + "/scripts/" + cmd[0], *cmd[1:]], env=env, cwd=W)
    print("RC", r.returncode, cmd, flush=True)
    if r.returncode and cmd[0].startswith("test_"):  # never run on a pipeline that failed its exactness test
        break
for f in glob.glob(W + "/results/cert/*.*"):
    shutil.copy(f, "/kaggle/working/")
'''


def push(a):
    """Code as a new version of the bigp-k5code dataset (kaggle_k5.py's layout)."""
    u = user()
    data = STAGE / "k5code"
    shutil.rmtree(data, ignore_errors=True)
    (data / "k5code" / "scripts").mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "scripts" / f, data / "k5code" / "scripts" / f)
    shutil.copytree(ROOT / "src" / "nids", data / "k5code" / "src" / "nids", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(GROOT, data / "k5code" / "src" / "groot", ignore=shutil.ignore_patterns("__pycache__"))
    (data / "dataset-metadata.json").write_text(json.dumps({"title": "bigp-k5code", "id": f"{u}/bigp-k5code",
                                                            "licenses": [{"name": "other"}]}))
    sh("datasets", "version", "-p", str(data), "-m", "conf", "-r", "zip", "-q")
    wait_ready(f"{u}/bigp-k5code")


def wait_ready(ref):
    time.sleep(60)
    for _ in range(90):
        if "ready" in subprocess.run([KAGGLE, "datasets", "status", ref], capture_output=True, text=True).stdout.lower():
            return
        time.sleep(20)
    raise SystemExit(f"{ref} not ready after 30 min")


def data(a):
    """Upload data/conf/{ds} and start its kernels."""
    u, ds = user(), a.ds
    slug = f"bigp-conf-{ds.replace('_', '-')}"
    d = STAGE / slug
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(ROOT / "data" / "conf" / ds, d / "conf" / ds)
    (d / "dataset-metadata.json").write_text(json.dumps({"title": slug, "id": f"{u}/{slug}",
                                                         "licenses": [{"name": "other"}]}))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{u}/{slug}"], capture_output=True).returncode == 0
    if not (exists and a.no_upload):
        sh("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "conf"] if exists else []), "-r", "zip", "-q")
        wait_ready(f"{u}/{slug}")
    for kind in a.kinds:
        k = STAGE / f"conf_{ds}_{kind}"
        shutil.rmtree(k, ignore_errors=True)
        k.mkdir()
        (k / "run.py").write_text((ATTACK if kind in ("attack", "c3", "k14", "leank1", "leanc3", "k16", "k17", "k18") else KERNEL).replace("{ds}", ds)
                                  .replace("{cmds}", repr(cmds(kind, ds))))
        ks = f"bigp-conf-{ds.replace('_', '-')}-{kind}"
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{u}/{ks}", "title": ks, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": kind in a.gpu, "enable_internet": False,
            "dataset_sources": [f"{u}/{slug}", f"{u}/bigp-k5code"] + ([f"{u}/bigp-proc-{ds.replace('_', '-')}"]
                                                                      if kind in ("attack", "c3", "k14", "k16", "k17", "k18") else [])
                               + ([f"{u}/bigp-proc-nf-ton", f"{u}/bigp-proc-nf-ctu13", f"{u}/bigp-proc-nf-ugr16"]
                                  if kind.startswith("lean") else [])}))
        for _ in range(144):  # a full concurrent-session quota refuses the push: retry every 10 min for 24 h
            acc = ["--accelerator", "NvidiaTeslaT4"] if kind in a.gpu else []  # enable_gpu alone runs on CPU
            r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(k), *acc], capture_output=True, text=True)
            print(ks, r.stdout.strip()[-200:], r.stderr.strip()[-200:], flush=True)
            if r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower():
                break
            time.sleep(600)


def proc(a):
    """data/processed/{ds}.parquet + hostmeta and the bowl champions (results/bowl/round*/death.json) for the attack kernel."""
    u, ds = user(), a.ds
    slug = f"bigp-proc-{ds.replace('_', '-')}"
    d = STAGE / slug
    shutil.rmtree(d, ignore_errors=True)
    (d / "processed").mkdir(parents=True)
    for f in (f"{ds}.parquet", f"{ds}_hostmeta.npz"):
        shutil.copy(ROOT / "data" / "processed" / f, d / "processed" / f)
    for f in (ROOT / "results" / "bowl").glob("round*/death.json"):
        (d / "bowl" / f.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, d / "bowl" / f.parent.name / "death.json")
    (d / "dataset-metadata.json").write_text(json.dumps({"title": slug, "id": f"{u}/{slug}",
                                                         "licenses": [{"name": "other"}]}))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{u}/{slug}"], capture_output=True).returncode == 0
    sh("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "proc"] if exists else []), "-r", "zip", "-q")
    wait_ready(f"{u}/{slug}")


def status(a):
    for ds in DS:
        for kind in KINDS:
            subprocess.run([KAGGLE, "kernels", "status", f"{user()}/bigp-conf-{ds.replace('_', '-')}-{kind}"])


def pull(a):
    out_dir = ROOT / "results" / "conf"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DS:
        for kind in KINDS:
            ks = f"bigp-conf-{ds.replace('_', '-')}-{kind}"
            o = STAGE / f"out_{ks}"
            shutil.rmtree(o, ignore_errors=True)
            o.mkdir(parents=True)
            if subprocess.run([KAGGLE, "kernels", "output", f"{user()}/{ks}", "-p", str(o)]).returncode:
                continue
            for f in o.iterdir():
                if f.suffix in (".csv", ".jsonl", ".log"):
                    shutil.copy(f, out_dir / f.name)
                    print("->", f.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "data", "proc", "status", "pull"])
    ap.add_argument("ds", nargs="?")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--gpu", nargs="*", default=[])  # kinds to run on GPU sessions (as extra CPU slots)
    ap.add_argument("--no-upload", action="store_true")  # dataset already up: only (re)start kernels
    a = ap.parse_args()
    {"push": push, "data": data, "proc": proc, "status": status, "pull": pull}[a.cmd](a)
