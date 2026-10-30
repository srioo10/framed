"""Rebuild a bowl round's study from a Kaggle kernel log when the session died before writing its output
(e.g. the 12 h kill during the death match). Every 'Trial N finished with values [...] and parameters {...}' line
becomes a trial of a local study with the logged values; the genome is re-derived through bowl.suggest from the
round's roster. Constraint values are not in the log: size / ratio come from the logged values, the certified FPR
is recomputed locally (CPU, cheap arena, seed 0) for size / speed-feasible trials. Then run the normal death match
on the local study (parts + merge), e.g.
  .venv/bin/python scripts/rebuild_round.py LOG --round 4 --workers 8
  .venv/bin/python scripts/bowl.py death --round 4 --storage sqlite:///results/bowl/round4_rebuilt.db --part i --parts 8
  .venv/bin/python scripts/bowl.py death --round 4 --storage sqlite:///results/bowl/round4_rebuilt.db --merge --parts 8
"""
import argparse
import ast
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bowl  # noqa: E402

LINE = re.compile(r"Trial (\d+) finished with values: (\[[^\]]*\]) and parameters: (\{[^{}]*\})")


def parse(text):
    """{trial number: (values, params)} from raw log text (plain or JSON-escaped)."""
    return {int(n): (ast.literal_eval(v), ast.literal_eval(p)) for n, v, p in LINE.findall(text.replace('\\"', '"'))}


def genome(params, roster):
    import optuna
    return bowl.suggest(optuna.trial.FixedTrial(params), roster)


def cert10(g):
    import torch
    torch.set_num_threads(1)
    pairs = bowl.load_pairs(bowl.ARENA, cheap=True)
    return bowl.evaluate(g, pairs, "cpu")[0]


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rd = bowl.OUT / f"round{a.round}"
    roster = bowl.apply_ablation(json.loads((rd / "roster.json").read_text()), rd)
    T = parse(a.log.read_text(errors="replace"))
    print(f"{len(T)} finished trials in the log")
    G = {n: genome(p, roster) for n, (_, p) in T.items()}
    feas = [n for n, (v, _) in T.items() if v[1] <= bowl.SIZE_MAX and v[2] <= bowl.RATIO_MAX]
    with ProcessPoolExecutor(a.workers) as ex:
        local = dict(zip(feas, ex.map(cert10, [G[n] for n in feas])))
    st = optuna.create_study(study_name=f"round{a.round}", storage=f"sqlite:///{bowl.OUT / f'round{a.round}_rebuilt.db'}",
                             directions=["maximize", "minimize", "minimize"], load_if_exists=False)
    for n in sorted(T):
        v, p = T[n]
        c = [v[1] - bowl.SIZE_MAX, v[2] - bowl.RATIO_MAX]
        res = {"fitness": v[0], "size": v[1], "ratio": v[2], "rebuilt": True}
        if n in local and "cert" in local[n]:
            res["cert"], res["local_fitness"] = local[n]["cert"], local[n]["fitness"]
            c.append(res["cert"][str(bowl.CERT_M)] - bowl.CERT_MAX)
        elif roster.get("cert_mode"):
            c.append(1.0)  # not size / speed feasible: never a finalist
        dist = {k: optuna.distributions.CategoricalDistribution([x]) for k, x in p.items()}
        st.add_trial(optuna.trial.create_trial(params=p, distributions=dist, values=v,
                                               user_attrs={"res": res, "genome": G[n], "c": c, "log_trial": n}))
    ok = sum(max(t.user_attrs["c"]) <= 0 for t in st.trials)
    print(f"rebuilt {len(st.trials)} trials, {ok} within all limits (certified FPR recomputed locally)")


if __name__ == "__main__":
    main()
