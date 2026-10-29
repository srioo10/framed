"""Run a bowl round on Kaggle GPU (PREREG §9 BOWL compute split). Stages arena arrays + bowl code + the round's
roster (+ the Optuna db, so a round resumes) as a PRIVATE Kaggle dataset, pushes a private GPU kernel that runs
fight then death, and pulls its output back into results/bowl/.
Needs ~/.kaggle/kaggle.json. Usage: .venv/bin/python scripts/kaggle_bowl.py push --round 1 --trials 256
                                   .venv/bin/python scripts/kaggle_bowl.py status | pull --round 1
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "kaggle"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
CODE = ["bowl.py", "arena.py", "fighters.py", "ple.py", "roc.py"]

KERNEL = '''import glob, os, shutil, subprocess, sys
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/code/bowl.py", recursive=True)[0]))
shutil.copytree(f"{src}/code", "/kaggle/working/code", dirs_exist_ok=True)
shutil.copytree(f"{src}/bowl", "/kaggle/working/bowl", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna==5.0.0"], check=True)
for cmd in ("fight", "death"):
    subprocess.run([sys.executable, "/kaggle/working/code/bowl.py", cmd, "--round", "{r}", "--trials", "{t}",
                    "--arena", f"{src}/arena", "--out", "/kaggle/working/bowl", "--jobs", "2", "--threads", "2"],
                   check=True)
'''


def user():
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def sh(*a):
    print("+", " ".join(a), flush=True)
    subprocess.run([KAGGLE, *a], check=True)


def push(a):
    u = user()
    shutil.rmtree(STAGE, ignore_errors=True)
    data = STAGE / "dataset"
    (data / "code").mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "scripts" / f, data / "code" / f)
    shutil.copytree(ROOT / "data" / "arena", data / "arena", ignore=shutil.ignore_patterns("*.json.bak"))
    shutil.copytree(ROOT / "results" / "bowl", data / "bowl")
    (data / "dataset-metadata.json").write_text(json.dumps({"title": "bigp-arena", "id": f"{u}/bigp-arena",
                                                            "licenses": [{"name": "other"}]}))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{u}/bigp-arena"], capture_output=True).returncode == 0
    if exists:
        sh("datasets", "version", "-p", str(data), "-m", f"round {a.round}", "-r", "zip", "-q")
    else:
        sh("datasets", "create", "-p", str(data), "-r", "zip", "-q")  # private by default
    k = STAGE / "kernel"
    k.mkdir()
    (k / "run_bowl.py").write_text(KERNEL.replace("{r}", str(a.round)).replace("{t}", str(a.trials)))
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/bigp-bowl", "title": "bigp-bowl", "code_file": "run_bowl.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{u}/bigp-arena"]}))
    sh("kernels", "push", "-p", str(k))


def status(a):
    sh("kernels", "status", f"{user()}/bigp-bowl")


def pull(a):
    out = STAGE / "output"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    sh("kernels", "output", f"{user()}/bigp-bowl", "-p", str(out))
    src = out / "bowl"
    if src.exists():
        shutil.copytree(src, ROOT / "results" / "bowl", dirs_exist_ok=True)
        print("pulled into results/bowl/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "pull"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--trials", type=int, default=256)
    a = ap.parse_args()
    {"push": push, "status": status, "pull": pull}[a.cmd](a)
