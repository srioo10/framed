"""Realised benign-test FPR per method and family from a stage5 JSON (appended to its .md).
Usage: .venv/bin/python scripts/stage5_fpr.py results/fusion/stage5_NAME"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage5  # noqa: E402

out = Path(sys.argv[1])
res = pd.DataFrame(json.loads(out.with_suffix(".json").read_text())["outer"])
L = ["\n## Realised benign-test FPR per method (mean over seeds; target in the heading)\n"]
for f in ("0.001", "0.01", "0.05"):
    cols = [f"{m}|test_fpr@{f}" for m in stage5.METHODS]
    if cols[0] not in res:
        continue
    L.append(f"### target {f}\n\n| fold | " + " | ".join(stage5.METHODS) + " |\n|---|" + "---|" * len(cols))
    for fold, g in res.groupby("fold"):
        L.append(f"| {fold.replace('__', '+')} | " + " | ".join(f"{g[c].mean():.4f}" for c in cols) + " |")
    L.append("")
text = "\n".join(L)
print(text)
md = out.with_suffix(".md")
if "## Realised benign-test FPR per method" not in md.read_text():
    md.write_text(md.read_text() + text + "\n")
