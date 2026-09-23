# -*- coding: utf-8 -*-
"""Regenerate fig6_prediction_intervals.png with fonts matching the updated fig1/2/3."""
import json
from pathlib import Path

import numpy as np
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

pi = pd.read_csv(OUT / "pred_intervals_session8.csv")
summ_pi = json.loads((OUT / "intervals_summary.json").read_text(encoding="utf-8"))
order = np.argsort(pi["y_inc"].values)

fig, ax = plt.subplots(figsize=(12, 5.8))
x = np.arange(len(pi))
ax.fill_between(x, pi["pi_lo"].values[order], pi["pi_hi"].values[order],
                alpha=.3, color="#5b8ff9",
                label=f"90% prediction interval (half-width {summ_pi['half_width_mm']:.2f} mm)")
ax.plot(x, pi["y_inc"].values[order], "o", ms=4, color="#222", label="Measured increment")
ax.plot(x, pi["pred_inc"].values[order], "-", lw=1.4, color="#d9534f", label="Predicted increment")
ax.set_xlabel("Test points (sorted by measured increment)", fontsize=16)
ax.set_ylabel("Increment (mm)", fontsize=16)
ax.set_title(f"Session 8 prediction intervals: nominal 90%, empirical coverage "
             f"{summ_pi['empirical_coverage_session8']*100:.1f}%", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=13, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig6_prediction_intervals.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig6_prediction_intervals.png",
      "| coverage:", summ_pi['empirical_coverage_session8'],
      "| half-width:", summ_pi['half_width_mm'])
