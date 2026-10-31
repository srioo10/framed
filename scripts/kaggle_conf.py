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

from kaggle_bowl import KAGGLE, ROOT, STAGE, sh, user

DS = ["nf_ctu13", "nf_ugr16", "nf_ton", "nf_cic"]
CODE = ["bowl.py", "arena.py", "fighters.py", "ple.py", "roc.py", "sensitivity.py", "cert_trees.py", "k5_trees.py",
        "k5b_net.py", "conf_clean.py", "conf_baselines.py"]
SEEDS = (2, 3, 4)
KERNEL = '''import glob, hashlib, os, shutil, subprocess, sys
code = os.path.dirname(glob.glob("/kaggle/input/**/conf_clean.py", recursive=True)[0])
nids = os.path.dirname(glob.glob("/kaggle/input/**/nids/models.py", recursive=True)[0])
conf = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/{ds}/pair_0.npz", recursive=True)[0]))
shutil.copytree(code, "/kaggle/working/w/scripts", dirs_exist_ok=True)
shutil.copytree(nids, "/kaggle/working/w/src/nids", dirs_exist_ok=True)
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
    if kind == "groot":
        return [["k5_trees.py", "--workers", "4", "--models", "groot", "--seed", str(s), "--out",
                 f"conf_groot_{ds}_s{s}.csv"] for s in SEEDS]
    s = int(kind[-1])  # trees2 / trees3 / trees4
    return [["k5_trees.py", "--workers", "4", "--seed", str(s), "--out", f"conf_trees_{ds}_s{s}.csv"]]


KINDS = ["clean", "baselines", "k5b", "trees2", "trees3", "trees4", "groot"]


def push(a):
    """Code as a new version of the bigp-k5code dataset (kaggle_k5.py's layout)."""
    u = user()
    data = STAGE / "k5code"
    shutil.rmtree(data, ignore_errors=True)
    (data / "k5code" / "scripts").mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "scripts" / f, data / "k5code" / "scripts" / f)
    shutil.copytree(ROOT / "src" / "nids", data / "k5code" / "src" / "nids", ignore=shutil.ignore_patterns("__pycache__"))
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
        (k / "run.py").write_text(KERNEL.replace("{ds}", ds).replace("{cmds}", repr(cmds(kind, ds))))
        ks = f"bigp-conf-{ds.replace('_', '-')}-{kind}"
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{u}/{ks}", "title": ks, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": kind in a.gpu, "enable_internet": False,
            "dataset_sources": [f"{u}/{slug}", f"{u}/bigp-k5code"]}))
        for _ in range(144):  # a full concurrent-session quota refuses the push: retry every 10 min for 24 h
            acc = ["--accelerator", "NvidiaTeslaT4"] if kind in a.gpu else []  # enable_gpu alone runs on CPU
            r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(k), *acc], capture_output=True, text=True)
            print(ks, r.stdout.strip()[-200:], r.stderr.strip()[-200:], flush=True)
            if r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower():
                break
            time.sleep(600)


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
    ap.add_argument("cmd", choices=["push", "data", "status", "pull"])
    ap.add_argument("ds", nargs="?")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--gpu", nargs="*", default=[])  # kinds to run on GPU sessions (as extra CPU slots)
    ap.add_argument("--no-upload", action="store_true")  # dataset already up: only (re)start kernels
    a = ap.parse_args()
    {"push": push, "data": data, "status": status, "pull": pull}[a.cmd](a)
