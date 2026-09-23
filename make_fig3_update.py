# -*- coding: utf-8 -*-
"""Regenerate fig3_ablation.png with fonts matching the updated fig1."""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_geo"
FIG = ROOT / "figures_geo"

labels = {"A": "A: 5 single-cycle lags", "B": "B: +5 cumulative lags", "C": "C: +3 dynamic indicators",
          "D": "D: +interval/session (16-D)", "E": "E: +14 geological features (30-D)"}
abl = []
for k in "ABCDE":
    m = json.loads((OUT / f"metrics_ablation{k}.json").read_text(encoding="utf-8"))
    abl.append({"stage": labels[k], "inc_mae": m["inc_mae"]})
abt = pd.DataFrame(abl)
print(abt.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.bar(abt["stage"], abt["inc_mae"], color=["#9e9e9e", "#9e9e9e", "#9e9e9e", "#5b8ff9", "#d9534f"])
ax.set_ylabel("Increment MAE (mm) \u2193", fontsize=16)
ax.set_ylim(0, 1.45)
ax.tick_params(axis="both", labelsize=14)
for i, v in enumerate(abt["inc_mae"]):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=13)
plt.xticks(rotation=12)
fig.tight_layout()
fig.savefig(FIG / "fig3_ablation.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig3_ablation.png")
