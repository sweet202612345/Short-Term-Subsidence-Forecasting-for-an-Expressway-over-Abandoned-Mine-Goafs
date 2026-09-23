# -*- coding: utf-8 -*-
"""Regenerate fig2_multiseed.png with fonts matching the updated fig1/fig3."""
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

base = pd.read_csv(OUT / "metrics_baselines.csv")
seeds = [json.loads((OUT / "metrics_main_geo.json").read_text(encoding="utf-8"))]
seeds += [json.loads((OUT / f"metrics_seed{i}.json").read_text(encoding="utf-8")) for i in range(1, 11)]
ms = pd.DataFrame([{"seed": s["seed"], "inc_mae": s["inc_mae"]} for s in seeds])
mean, std = ms["inc_mae"].mean(), ms["inc_mae"].std()
print(ms.round(3).to_string(index=False))
print(f"mean {mean:.3f} +- {std:.3f}")

fig, ax = plt.subplots(figsize=(12, 5.8))
ax.bar(ms["seed"].astype(str), ms["inc_mae"], color="#d9534f")
ax.axhline(base.iloc[0]["inc_mae"], ls="--", c="#555", lw=1.8,
           label=f"Persistence {base.iloc[0]['inc_mae']:.3f}")
ax.axhline(base.iloc[1]["inc_mae"], ls="--", c="#2e7d32", lw=1.8,
           label=f"Rate extrapolation {base.iloc[1]['inc_mae']:.3f}")
ax.axhline(mean, ls="-", c="#d9534f", alpha=.5, lw=1.8,
           label=f"Mean {mean:.3f}\u00b1{std:.3f}")
ax.set_xlabel("Random seed", fontsize=16)
ax.set_ylabel("Increment MAE (mm) \u2193", fontsize=16)
ax.set_title("Stability over 11 independent training runs (temporal + geological model)", fontsize=16)
ax.set_ylim(0, 1.6)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=13, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig2_multiseed.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig2_multiseed.png")
