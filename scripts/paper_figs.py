"""Paper figures from committed result files (no training).

Fig 1: per-family host-day detection of the held-out family on NF-CICIDS2018 (seeds 1-4), 4 methods.
Fig 2: detection vs model size (log scale), NF-CICIDS2018 host level.
Fig 3: flow-level macro TPR@1%FPR per dataset for MLP / RF / XGB / MCDE.
Usage: .venv/bin/python scripts/paper_figs.py --out reports/figures
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def days(variant, member, seeds=(1, 2, 3, 4)):
    d = [pd.read_csv(ROOT / "results/onepass/nf_cic" / f"{variant}_{member}_s{s}_days.csv") for s in seeds]
    return pd.concat(d, ignore_index=True)


def bars(ax, groups, series, width=0.2):
    """Grouped bars with a 2px surface gap and value labels (palette contrast WARN: labels are required)."""
    for i, (name, vals) in enumerate(series.items()):
        x = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        b = ax.bar(x, vals, width * 0.92, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center",
                    va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(groups)), groups, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, 1.12), labelcolor=INK2)


def fig1(out):
    arms = {"Per-flow MLP": days("none", "mlp_ce").groupby("fold").P.mean(),
            "OnePass (MLP)": days("host", "mlp_ce").groupby("fold").A.mean(),
            "DDU-OnePass": days("host", "mlp_sn_gda").groupby("fold").A.mean(),
            "RF + host evidence": days("host", "rf").groupby("fold").A.mean()}
    fams = list(arms["OnePass (MLP)"].index)
    hosts = days("host", "mlp_ce").groupby("fold").host.nunique()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars(ax, [f"{f}\n({hosts[f]} host{'s' if hosts[f] > 1 else ''})" for f in fams], {k: v.reindex(fams).to_numpy() for k, v in arms.items()})
    ax.set_ylabel("attacker host-days detected")
    ax.set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(out / "fig1_host_level_per_family.png")
    fig.savefig(out / "fig1_host_level_per_family.pdf")


def fig2(out):
    pts = [("MLP int8 (OnePass)", 10.9, days("host", "mlp_ce_int8"), 0, (4, -22)),
           ("MLP fp32", 26.6, days("host", "mlp_ce"), 0, (6, 6)),
           ("DDU-OnePass", 233, days("host", "mlp_sn_gda"), 0, (6, 4)),
           ("XGBoost", 14 * 1024, days("host", "xgb"), 1, (6, -22)),
           ("Random forest", 287 * 1024, days("host", "rf"), 1, (-30, 8))]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for name, kb, d, k, off in pts:
        y = d.groupby("fold").A.mean().mean()
        ax.scatter(kb, y, s=36, color=SERIES[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{name}\n{y:.3f}", (kb, y), textcoords="offset points", xytext=off,
                    fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(5, 3e6)
    ax.set_ylim(0.45, 0.75)
    ax.set_xlabel("serialized model size (KB, log scale)")
    ax.set_ylabel("macro host-day detection")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.scatter([], [], color=SERIES[0], label="one-pass neural"), ax.scatter([], [], color=SERIES[1], label="tree ensemble")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig2_size_vs_detection.png")
    fig.savefig(out / "fig2_size_vs_detection.pdf")


def fig3(out):
    sets = [("NF-CICIDS2018", "nf_cic", "host", [1, 2, 3, 4]), ("NF-ToN-IoT", "nf_ton", "host", [0, 1, 2, 3, 4]),
            ("CICIoT2023", "ciciot_csv", "none", [0, 1, 2, 3, 4]), ("UNSW raw", "unsw_raw", "ct_causal", [0, 1, 2, 3, 4]),
            ("UNSW official", "unsw_official", "none", [0, 1, 2, 3, 4])]
    members = {"Tiny MLP": "mlp_ce", "RF": "rf", "XGBoost": "xgb", "MCDE": "mcde"}
    vals = {k: [] for k in members}
    for _, ds, var, seeds in sets:
        r = pd.read_csv(ROOT / f"results/onepass/flow_{ds}_rows.csv")
        r = r[(r.variant == var) & r.seed.isin(seeds) & ~r.fold.str.contains("__")]
        m = r.groupby(["member", "seed", "fold"]).tpr.mean().groupby(["member", "seed"]).mean().groupby("member").mean()
        for k, col in members.items():
            vals[k].append(m.get(col, float("nan")))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars(ax, [s[0] for s in sets], vals)
    ax.set_ylabel("macro TPR at 1% FPR, unseen family")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out / "fig3_flow_level_datasets.png")
    fig.savefig(out / "fig3_flow_level_datasets.pdf")
    pd.DataFrame(vals, index=[s[0] for s in sets]).round(3).to_csv(out / "fig3_values.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig1(a.out), fig2(a.out), fig3(a.out)


if __name__ == "__main__":
    main()
