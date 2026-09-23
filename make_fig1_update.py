# -*- coding: utf-8 -*-
"""Regenerate fig1_main_results.png: short 'Best single learner' label, larger fonts."""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14          # base font size (was ~10)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_geo"
FIG = ROOT / "figures_geo"

base = pd.read_csv(OUT / "metrics_baselines.csv")
main = json.loads((OUT / "metrics_main.json").read_text(encoding="utf-8"))
geo = json.loads((OUT / "metrics_main_geo.json").read_text(encoding="utf-8"))

tbl = pd.DataFrame([
    {"model": "Persistence", "inc_mae": base.iloc[0]["inc_mae"], "cum_r2": base.iloc[0]["cum_r2"]},
    {"model": "Rate extrapolation", "inc_mae": base.iloc[1]["inc_mae"], "cum_r2": base.iloc[1]["cum_r2"]},
    {"model": "AR(5) ridge", "inc_mae": base.iloc[2]["inc_mae"], "cum_r2": base.iloc[2]["cum_r2"]},
    {"model": "AutoGluon temporal (16-D)", "inc_mae": main["inc_mae"], "cum_r2": main["cum_r2"]},
    {"model": "AutoGluon temporal+geo (30-D, proposed)", "inc_mae": geo["inc_mae"], "cum_r2": geo["cum_r2"]},
    {"model": "Best single learner", "inc_mae": geo["bestsingle_inc_mae"], "cum_r2": geo["bestsingle_cum_r2"]},
])
print(tbl.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
colors = ["#9e9e9e"] * 3 + ["#5b8ff9", "#d9534f", "#bdbdbd"]

ax[0].barh(tbl["model"], tbl["inc_mae"], color=colors)
ax[0].set_xlabel("Increment MAE (mm) \u2193", fontsize=16)
ax[0].set_xlim(0, 1.65)
ax[0].tick_params(axis="both", labelsize=14)
for i, v in enumerate(tbl["inc_mae"]):
    ax[0].text(v - 0.02, i, f"{v:.3f}", va="center", ha="right", fontsize=13, color="white")

ax[1].barh(tbl["model"], tbl["cum_r2"], color=colors)
ax[1].set_xlabel("Cumulative reconstructed R\u00b2 \u2191", fontsize=16)
ax[1].set_xlim(0, 1.02)
ax[1].tick_params(axis="both", labelsize=14)
for i, v in enumerate(tbl["cum_r2"]):
    ax[1].text(v - 0.012, i, f"{v:.3f}", va="center", ha="right", fontsize=13, color="white")

fig.tight_layout()
fig.savefig(FIG / "fig1_main_results.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig1_main_results.png")
