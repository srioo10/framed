"""Run a bowl round on Kaggle GPU (PREREG §9 BOWL compute split). Stages arena arrays + bowl code + the round's
roster (+ the Optuna db, so a round resumes) as a PRIVATE Kaggle dataset, pushes a private GPU kernel that runs
fight then death, and pulls its output back into results/bowl/.
Needs a Kaggle token (~/.kaggle/access_token or kaggle.json). Usage: .venv/bin/python scripts/kaggle_bowl.py push --round 1 --trials 256
                                   .venv/bin/python scripts/kaggle_bowl.py status | pull --round 1
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "kaggle"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
CODE = ["bowl.py", "arena.py", "fighters.py", "ple.py", "roc.py", "sensitivity.py"]

KERNEL = '''import glob, os, shutil, subprocess, sys, time
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/code/bowl.py", recursive=True)[0]))
shutil.copytree(f"{src}/code", "/kaggle/working/code", dirs_exist_ok=True)
shutil.copytree(f"{src}/bowl", "/kaggle/working/bowl", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna==5.0.0"], check=True)
W = {w}  # parallel worker processes sharing the GPU and the sqlite study


def run(cmd, *extra):
    return subprocess.Popen([sys.executable, "/kaggle/working/code/bowl.py", cmd, "--round", "{r}", "--trials", "{t}",
                             "--arena", f"{src}/arena", "--out", "/kaggle/working/bowl", "--jobs", "1",
                             "--threads", str(max(1, 4 // W)), "--parts", str(W), *extra])


def wait(ps):
    assert all(p.wait() == 0 for p in ps), "a bowl worker failed"


if {r} > 1:
    wait([run("ablate")])
ps = [run("fight", "--part", "0")]
time.sleep(60)  # worker 0 enqueues the seed genomes first
wait(ps + [run("fight", "--part", str(i)) for i in range(1, W)])
wait([run("death", "--part", str(i)) for i in range(W)])
wait([run("death", "--merge")])
'''


def user():
    out = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True, check=True).stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.strip().startswith("- username"))


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
    time.sleep(90)  # "ready" can still describe the previous version right after `datasets version`
    for _ in range(60):  # a kernel pushed before the dataset is ready gets no data source
        st = subprocess.run([KAGGLE, "datasets", "status", f"{u}/bigp-arena"], capture_output=True, text=True).stdout
        if "ready" in st.lower():
            break
        time.sleep(20)
    else:
        raise SystemExit("dataset not ready after 20 min")
    k = STAGE / "kernel"
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir()
    (k / "run_bowl.py").write_text(KERNEL.replace("{r}", str(a.round)).replace("{t}", str(a.trials))
                                   .replace("{w}", str(a.workers)))
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/bigp-bowl", "title": "bigp-bowl", "code_file": "run_bowl.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{u}/bigp-arena"]}))
    sh("kernels", "push", "-p", str(k))


def bosses(a):
    """Ultra bosses as a second private kernel on the staged arena dataset (scripts/ultra_bosses.py)."""
    u = user()
    k = STAGE / "bosses"
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "ultra_bosses.py", k / "ultra_bosses.py")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/bigp-bosses", "title": "bigp-bosses", "code_file": "ultra_bosses.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{u}/bigp-arena"]}))
    sh("kernels", "push", "-p", str(k))


def pull_bosses(a):
    out = STAGE / "bosses_output"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    sh("kernels", "output", f"{user()}/bigp-bosses", "-p", str(out))
    for f in sorted(out.glob("ultra/*/ultra_*.npz")):
        shutil.copy(f, ROOT / "data" / "arena" / f.parent.name / f.name)
        print("->", f.parent.name, f.name)


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
    ap.add_argument("cmd", choices=["push", "status", "pull", "bosses", "pull_bosses"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"push": push, "status": status, "pull": pull, "bosses": bosses, "pull_bosses": pull_bosses}[a.cmd](a)
