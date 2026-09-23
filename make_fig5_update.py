# -*- coding: utf-8 -*-
"""Regenerate fig5_importance.png: readable manuscript-consistent feature labels + larger fonts."""
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

# code name -> manuscript-readable name (terminology per Sections 2.3.2/3.6 and Tables A1/A2)
NAME = {
    "single_lag_1": "Single-cycle displacement (lag-1)",
    "single_lag_2": "Single-cycle displacement (lag-2)",
    "single_lag_3": "Single-cycle displacement (lag-3)",
    "single_lag_4": "Single-cycle displacement (lag-4)",
    "single_lag_5": "Single-cycle displacement (lag-5)",
    "cum_lag_1": "Cumulative displacement (lag-1)",
    "cum_lag_2": "Cumulative displacement (lag-2)",
    "cum_lag_3": "Cumulative displacement (lag-3)",
    "cum_lag_4": "Cumulative displacement (lag-4)",
    "cum_lag_5": "Cumulative displacement (lag-5)",
    "roll_mean_3": "Three-session rolling mean",
    "roll_std_3": "Three-session rolling standard deviation",
    "diff_1": "Displacement increment difference",
    "timestep": "Session index (timestep)",
    "days_prev": "Previous interval length (days_prev)",
    "days_next": "Predicted interval length (days_next)",
    "has_goaf": "Goaf presence",
    "mining_method": "Mining method",
    "room_width_m": "Room width",
    "pillar_width_m": "Pillar width",
    "mining_depth_m": "Mining depth",
    "seam_thickness_m": "Seam thickness",
    "depth_thickness_ratio": "Depth\u2013thickness ratio",
    "dip_angle_deg": "Dip angle",
    "recovery_rate": "Recovery rate",
    "years_since_abandonment": "Years since abandonment",
    "overburden_lithology": "Overburden lithology",
    "coal_mine": "Coal mine",
    "stability_class": "Stability class",
    "grouting": "Grouting status",
}
GEO = ["has_goaf", "seam_thickness_m", "mining_depth_m", "depth_thickness_ratio",
       "dip_angle_deg", "recovery_rate", "years_since_abandonment", "room_width_m",
       "pillar_width_m", "grouting", "mining_method", "overburden_lithology",
       "coal_mine", "stability_class"]

fi = pd.read_csv(OUT / "table_permutation_importance.csv", index_col=0)
fi = fi.sort_values("importance")
fi["type"] = ["geological" if f in GEO else "temporal" for f in fi.index]
fi["label"] = [NAME[f] for f in fi.index]

fig, ax = plt.subplots(figsize=(12, 9.5))
ax.barh(fi["label"], fi["importance"], xerr=fi["stddev"], error_kw={"lw": 1.6},
        color=["#d9534f" if t == "geological" else "#5b8ff9" for t in fi["type"]])
ax.set_xlabel("Permutation importance (\u0394R\u00b2, 20 shuffles, Session 8 independent test)", fontsize=16)
ax.tick_params(axis="both", labelsize=13.5)
ax.axvline(0, color="#333", lw=0.8)
fig.tight_layout()
fig.savefig(FIG / "fig5_importance.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved:", FIG / "fig5_importance.png", "| features:", len(fi))
