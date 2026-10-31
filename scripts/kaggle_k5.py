"""K5 on Kaggle CPU (PREREG §9 CERT K5 row): one private CPU kernel per dataset runs scripts/k5_trees.py on that
dataset's arena pairs. Code (scripts + src/nids) goes up as a small private dataset; the arena comes from the existing
private bigp-arena dataset. Library versions are pinned to the local ones. Each kernel prints an md5 of every arena
pair it reads, to check against the local arena.
k10: the same code dataset, plus three private CPU kernels running scripts/k10_capacity.py (A configs, B configs).
Usage: .venv/bin/python scripts/kaggle_k5.py push | kernels [--ds nf_ctu13] | k10 | status | pull (pull overwrites results/cert/k5_*)
"""
import argparse
import json
import shutil
import subprocess
import time

from kaggle_bowl import KAGGLE, ROOT, STAGE, sh, user

DS = ["nf_ctu13", "nf_ugr16", "nf_cic"]
CODE = ["bowl.py", "arena.py", "fighters.py", "ple.py", "roc.py", "sensitivity.py", "cert_trees.py", "k5_trees.py",
        "k10_capacity.py"]
K10 = {"a": ["A_w128", "A_w128_d3"], "a2": ["A_w192"], "b": ["B_w64", "B_w128"]}
KERNEL = '''import glob, hashlib, os, shutil, subprocess, sys
hits = glob.glob("/kaggle/input/**/k5_trees.py", recursive=True)
if not hits:
    subprocess.run(["find", "/kaggle/input", "-maxdepth", "5"])
    raise SystemExit("k5 code not mounted")
code = os.path.dirname(hits[0])
nids = os.path.dirname(glob.glob("/kaggle/input/**/nids/models.py", recursive=True)[0])
arena = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/arena/{ds}/pair_0.npz", recursive=True)[0]))
shutil.copytree(code, "/kaggle/working/k5/scripts", dirs_exist_ok=True)
shutil.copytree(nids, "/kaggle/working/k5/src/nids", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn=={skl}", "xgboost=={xgb}", "scipy=={sp}"],
               timeout=300)  # best effort: without internet the preinstalled versions run (printed, disclosed)
subprocess.run([sys.executable, "-c", "import sklearn, xgboost, scipy; print('versions', sklearn.__version__, "
                "xgboost.__version__, scipy.__version__)"])
for f in sorted(glob.glob(f"{arena}/{ds}/pair_*.npz")):
    print("md5", "{ds}/" + os.path.basename(f), hashlib.md5(open(f, "rb").read()).hexdigest(), flush=True)
pairs = [f"{ds}/" + os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{arena}/{ds}/pair_*.npz"))]
env = dict(os.environ, PYTHONPATH="/kaggle/working/k5/src", OMP_NUM_THREADS="1")
subprocess.run([sys.executable, "/kaggle/working/k5/scripts/k5_trees.py", "--workers", "4", "--arena", arena,
                "--out", "k5_{ds}.csv", "--pairs", *pairs], check=True, env=env)
'''
KERNEL10 = '''import glob, os, shutil, subprocess, sys
code = os.path.dirname(glob.glob("/kaggle/input/**/k10_capacity.py", recursive=True)[0])
nids = os.path.dirname(glob.glob("/kaggle/input/**/nids/models.py", recursive=True)[0])
arena = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/arena/nf_ctu13/pair_0.npz", recursive=True)[0]))
shutil.copytree(code, "/kaggle/working/k10/scripts", dirs_exist_ok=True)
shutil.copytree(nids, "/kaggle/working/k10/src/nids", dirs_exist_ok=True)
subprocess.run([sys.executable, "-c", "import torch, sklearn, xgboost; print('versions', torch.__version__, "
                "torch.cuda.is_available(), sklearn.__version__, xgboost.__version__)"])
env = dict(os.environ, PYTHONPATH="/kaggle/working/k10/src")
subprocess.run([sys.executable, "/kaggle/working/k10/scripts/k10_capacity.py", "--device", "cpu", "--arena", arena,
                "--out", "k10_{part}.json", "--only", *{only}], check=True, env=env)
shutil.copy("/kaggle/working/k10/results/bowl/k10_{part}.json", "/kaggle/working/k10_{part}.json")
'''


def push(a):
    import numpy, scipy, sklearn, xgboost  # noqa: F401, E401
    u = user()
    data = STAGE / "k5code"
    shutil.rmtree(data, ignore_errors=True)
    (data / "k5code" / "scripts").mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "scripts" / f, data / "k5code" / "scripts" / f)
    shutil.copytree(ROOT / "src" / "nids", data / "k5code" / "src" / "nids", ignore=shutil.ignore_patterns("__pycache__"))
    (data / "dataset-metadata.json").write_text(json.dumps({"title": "bigp-k5code", "id": f"{u}/bigp-k5code",
                                                            "licenses": [{"name": "other"}]}))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{u}/bigp-k5code"], capture_output=True).returncode == 0
    sh("datasets", "version" if exists else "create", "-p", str(data), *(["-m", "k5"] if exists else []), "-r", "zip", "-q")
    time.sleep(60)
    for _ in range(60):
        if "ready" in subprocess.run([KAGGLE, "datasets", "status", f"{u}/bigp-k5code"], capture_output=True,
                                     text=True).stdout.lower():
            break
        time.sleep(20)
    else:
        raise SystemExit("code dataset not ready after 20 min")
    if a.cmd == "k10":
        k10_kernels(u)
    else:
        kernels(u, a.ds)


def k10_kernels(u):
    for part, only in K10.items():
        k = STAGE / f"k10_{part}"
        shutil.rmtree(k, ignore_errors=True)
        k.mkdir()
        (k / "run_k10.py").write_text(KERNEL10.replace("{part}", part).replace("{only}", repr(only)))
        slug = f"bigp-k10-{part}"
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{u}/{slug}", "title": slug, "code_file": "run_k10.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [f"{u}/bigp-arena", f"{u}/bigp-k5code"]}))  # GPU quota gone: CPU image
        sh("kernels", "push", "-p", str(k))


def kernels(u, dss=DS):
    import scipy, sklearn, xgboost  # noqa: E401
    for ds in dss:
        k = STAGE / f"k5_{ds}"
        shutil.rmtree(k, ignore_errors=True)
        k.mkdir()
        (k / "run_k5.py").write_text(KERNEL.replace("{ds}", ds).replace("{skl}", sklearn.__version__)
                                     .replace("{xgb}", xgboost.__version__).replace("{sp}", scipy.__version__))
        slug = f"bigp-k5-{ds.replace('_', '-')}"
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{u}/{slug}", "title": slug, "code_file": "run_k5.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [f"{u}/bigp-arena", f"{u}/bigp-k5code"]}))
        sh("kernels", "push", "-p", str(k))


def status(a):
    for ds in DS:
        sh("kernels", "status", f"{user()}/bigp-k5-{ds.replace('_', '-')}")
    for part in K10:
        subprocess.run([KAGGLE, "kernels", "status", f"{user()}/bigp-k10-{part}"])


def pull(a):
    for ds in DS:
        out = STAGE / f"k5_out_{ds}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        sh("kernels", "output", f"{user()}/bigp-k5-{ds.replace('_', '-')}", "-p", str(out))
        for f in out.rglob("k5_*"):
            shutil.copy(f, ROOT / "results" / "cert" / f.name)
            print("->", f.name)
    for part in K10:
        out = STAGE / f"k10_out_{part}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        subprocess.run([KAGGLE, "kernels", "output", f"{user()}/bigp-k10-{part}", "-p", str(out)])
        for f in out.glob("k10_*.json"):
            shutil.copy(f, ROOT / "results" / "bowl" / f.name)
            print("->", f.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "kernels", "k10", "status", "pull"])
    ap.add_argument("--ds", nargs="+", default=DS)  # push/kernels: only these datasets
    a = ap.parse_args()
    {"push": push, "k10": push, "kernels": lambda a: kernels(user(), a.ds), "status": status, "pull": pull}[a.cmd](a)
