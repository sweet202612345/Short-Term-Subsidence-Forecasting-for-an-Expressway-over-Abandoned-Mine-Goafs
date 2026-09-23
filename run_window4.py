#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Window-length sensitivity experiment (4-session input window) for the major
revision of infrastructures-4531268, answering the reviewers' request for a
shorter input history and additional consecutive prediction dates.

Reuses the documented pipeline (run_pipeline.py / run_pipeline_geo.py) with
lags=4 instead of 5. Feature set (28):
  temporal 14: single_lag_1..4, cum_lag_1..4, diff_1, roll_mean_3, roll_std_3,
               timestep, days_prev, days_next
  static 14  : GEO_NUM (10 numeric) + GEO_CAT (4 categorical) from
               data/geo_features.csv (numeric NaN -> 0, categorical -> "none")

Two configurations, both seed=2024, identical AutoGluon settings as the paper
(presets=good_quality, num_bag_folds=5, num_stack_levels=2, num_bag_sets=1,
time_limit=100 s, eval_metric=r2, tuning_data=calibration session,
use_bag_holdout=True, hyperparameters(2024), ag_args_fit random_state,
ag_args_ensemble fold_fitting_strategy=sequential_local):

  python run_window4.py w4s8   # train S6, calib S7, test S8 (same protocol)
  python run_window4.py w4s7   # train S5, calib S6, test S7 (one earlier date)

With lags=4 the dropna keeps session-5 rows (S5 features use only S1-S4),
which makes the W4S7 configuration possible.

Results -> results_geo/metrics_w4s8.json, metrics_w4s7.json
Models  -> models/ag_w4s8/, models/ag_w4s7/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline as rp  # documented config, AutoGluon wrapper, metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_geo"
OUT.mkdir(exist_ok=True)
DATA_LONG = ROOT / "data" / "monitoring_long_real.csv"
DATA_GEO = ROOT / "data" / "geo_features.csv"

SESSION_DATES = rp.SESSION_DATES
DAYS = rp.DAYS
SEED = rp.CFG["main_seed"]  # 2024

LAGS = 4

GEO_NUM = ["has_goaf", "seam_thickness_m", "mining_depth_m",
           "depth_thickness_ratio", "dip_angle_deg", "recovery_rate",
           "years_since_abandonment", "room_width_m", "pillar_width_m",
           "grouting"]
GEO_CAT = ["mining_method", "overburden_lithology", "coal_mine", "stability_class"]
GEO_FEATURES = GEO_NUM + GEO_CAT

W4_TEMPORAL = ([f"single_lag_{i}" for i in range(1, LAGS + 1)]
               + [f"cum_lag_{i}" for i in range(1, LAGS + 1)]
               + ["diff_1", "roll_mean_3", "roll_std_3",
                  "timestep", "days_prev", "days_next"])
W4_FEATURES = W4_TEMPORAL + GEO_FEATURES   # 28 features
W4_SINGLE_LAGS = [f"single_lag_{i}" for i in range(1, LAGS + 1)]

CONFIGS = {
    "w4s8": dict(train_session=6, calib_session=7, test_session=8),
    "w4s7": dict(train_session=5, calib_session=6, test_session=7),
}


# -------------------------------------------------------------- features ---
def build_features_w4() -> pd.DataFrame:
    """Leakage-safe 4-session-window features: identical logic to
    rp.build_features but with lags=4 and dropna on the 4-window feature set
    (keeps session-5 rows, whose features use only S1-S4)."""
    df = pd.read_csv(DATA_LONG)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["replicate"] <= 2]
    df = df.sort_values(["point_id", "session"]).reset_index(drop=True)
    df["cum_recon"] = df.groupby("point_id")["increment_mm"].cumsum()

    g = df.groupby("point_id", group_keys=False)
    out = df.copy()
    for i in range(1, LAGS + 1):
        out[f"single_lag_{i}"] = g["increment_mm"].shift(i)
        out[f"cum_lag_{i}"] = g["cum_recon"].shift(i)
    out["roll_mean_3"] = g["increment_mm"].transform(lambda s: s.shift(1).rolling(3).mean())
    out["roll_std_3"] = g["increment_mm"].transform(lambda s: s.shift(1).rolling(3).std())
    out["diff_1"] = out["single_lag_1"] - out["single_lag_2"]
    out["days_next"] = out["session"].map({s: DAYS[s - 2] for s in range(2, 9)})
    out["days_prev"] = out["session"].map({s: DAYS[s - 3] for s in range(3, 9)})
    out["timestep"] = out["session"]
    feat = out.dropna(subset=W4_TEMPORAL + ["increment_mm"]).reset_index(drop=True)

    geo = pd.read_csv(DATA_GEO)
    feat = feat.merge(geo, on="point_id", how="left", validate="many_to_one")
    for c in GEO_NUM:
        feat[c] = feat[c].fillna(0.0)
    for c in GEO_CAT:
        feat[c] = feat[c].fillna("none").astype(str)
    return feat


# ------------------------------------------------------------------ run ----
def run_config(tag: str) -> dict:
    cfg = CONFIGS[tag]
    feat = build_features_w4()
    tr = feat[feat["session"] == cfg["train_session"]]
    ca = feat[feat["session"] == cfg["calib_session"]]
    te = feat[feat["session"] == cfg["test_session"]]
    assert len(tr) and len(ca) and len(te), f"empty split for {tag}"
    print(f"[{tag}] train S{cfg['train_session']} n={len(tr)} | "
          f"calib S{cfg['calib_session']} n={len(ca)} | "
          f"test S{cfg['test_session']} n={len(te)} | features={len(W4_FEATURES)}")

    # --- AutoGluon ensemble, identical settings to the paper ---
    t0 = time.time()
    predictor = rp.fit_autogluon(tr, ca, SEED, tag, features=W4_FEATURES)
    fit_s = time.time() - t0

    pred = predictor.predict(te[W4_FEATURES]).values
    ev = rp.eval_frame(te, pred, f"AutoGluon_{tag}")
    ev.to_csv(OUT / f"pred_{tag}.csv", index=False)
    row = rp.aggregate(ev, f"AutoGluon_{tag}", f"session{cfg['test_session']}")
    row["fit_seconds"] = round(fit_s, 1)
    row["seed"] = SEED
    row["lags"] = LAGS
    row["n_features"] = len(W4_FEATURES)
    row.update({f"session_{k}": v for k, v in cfg.items()})

    lb = predictor.leaderboard(te[W4_FEATURES + ["increment_mm"]], silent=True)
    lb.to_csv(OUT / f"leaderboard_{tag}.csv", index=False)
    best = lb[~lb["model"].str.contains("WeightedEnsemble", na=False)].iloc[0]
    ev_best = rp.eval_frame(te, predictor.predict(te[W4_FEATURES], model=best["model"]).values,
                            f"BestSingle_{tag}")
    ev_best.to_csv(OUT / f"pred_bestsingle_{tag}.csv", index=False)
    row_best = rp.aggregate(ev_best, best["model"], f"session{cfg['test_session']}")
    row.update({f"bestsingle_{k}": v for k, v in row_best.items() if k not in ("model", "tag")})
    row["bestsingle_name"] = best["model"]

    # --- three baselines on the SAME test session ---
    from sklearn.linear_model import Ridge
    baselines = {}
    ev_p = rp.eval_frame(te, te["single_lag_1"].values, "Persistence")
    baselines["persistence_mae"] = rp.aggregate(ev_p, "Persistence", "")["inc_mae"]
    rate_pred = te["roll_mean_3"].values * (te["days_next"].values / te["days_prev"].values)
    ev_r = rp.eval_frame(te, rate_pred, "RateExtrapolation")
    baselines["rate_extrapolation_mae"] = rp.aggregate(ev_r, "RateExtrapolation", "")["inc_mae"]
    ar = Ridge(alpha=1.0, random_state=SEED)
    ar.fit(tr[W4_SINGLE_LAGS], tr["increment_mm"])
    ev_a = rp.eval_frame(te, ar.predict(te[W4_SINGLE_LAGS]), f"AR{LAGS}_Ridge")
    baselines[f"ar{LAGS}_ridge_mae"] = rp.aggregate(ev_a, f"AR{LAGS}_Ridge", "")["inc_mae"]
    row["baselines"] = baselines

    with open(OUT / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "w4s8"
    if tag not in CONFIGS:
        raise SystemExit(f"unknown config {tag}; use w4s8 | w4s7")
    run_config(tag)


if __name__ == "__main__":
    main()
