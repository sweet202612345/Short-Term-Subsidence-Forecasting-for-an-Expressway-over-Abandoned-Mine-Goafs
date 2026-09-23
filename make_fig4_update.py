# -*- coding: utf-8 -*-
"""Regenerate fig4_zone_cv.png with fonts matching the updated fig1/2/3/6."""
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

cv = []
for z in ["K21", "K22", "K23", "JPK", "RAMP"]:
    m = json.loads((OUT / f"metrics_cv_{z}.json").read_text(encoding="utf-8"))
    cv.append({"zone": z, "inc_mae": m["inc_mae"], "inc_r2": m["inc_r2"]})
cvt = pd.DataFrame(cv)
print(cvt.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
ax[0].bar(cvt["zone"], cvt["inc_mae"], color="#5b8ff9")
ax[0].set_ylabel("Increment MAE (mm) \u2193", fontsize=16)
ax[0].set_title("Leave-one-zone CV: increment MAE", fontsize=15)
ax[0].tick_params(axis="both", labelsize=14)
ax[0].set_ylim(0, 1.65)
for i, v in enumerate(cvt["inc_mae"]):
    ax[0].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=13)

ax[1].bar(cvt["zone"], cvt["inc_r2"], color="#9e9e9e")
ax[1].set_ylabel("Increment R\u00b2", fontsize=16)
ax[1].set_title("Leave-one-zone CV: increment R\u00b2", fontsize=15)
ax[1].tick_params(axis="both", labelsize=14)
ax[1].set_ylim(-1.0, 0.12)
for i, v in enumerate(cvt["inc_r2"]):
    ax[1].text(i, v - 0.07, f"{v:.3f}", ha="center", fontsize=13)

fig.suptitle("Spatial zone cross-validation (temporal + geological features)", fontsize=16)
fig.tight_layout()
fig.savefig(FIG / "fig4_zone_cv.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig4_zone_cv.png")
