"""TSTR: train on synthetic, test on real (CGM-only tasks both data sources support).

Tasks on 15-minute CGM (the real datasets have no wearables, so only CGM features are used;
time-of-day features are tested separately because they carry the local meal schedule):
    forecast  glucose 30 and 60 minutes ahead from the last 2 hours (MAE, RMSE)
    spike     onset of glucose > 180 mg/dL (at least 15 min) within the next 2 hours, for
              windows that start at or below 180 (AUROC, PR-AUC)

Training sources: synthetic main cohort (train split), synthetic level-matched pool, and the
real Shanghai calibration half (TRTR reference, approved for validation only; it never feeds the
product models). Test sets: Shanghai validation half and CGMacros T2D (external). Persistence
and linear extrapolation give the sanity floor.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from validation.windows import regular_series

HIST = 8          # 2 h of history
H30, H60 = 2, 4   # steps ahead
SPIKE_H = 8       # 2 h
THRESH = 180.0
PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0, "deterministic": True, "num_threads": 4}


def samples(cgm: pd.DataFrame, stride: int = 2) -> pd.DataFrame:
    rows = []
    for pid, s in regular_series(cgm).items():
        v = s.to_numpy(float)
        hours = s.index.hour.to_numpy() + s.index.minute.to_numpy() / 60
        for i in range(HIST - 1, len(v) - SPIKE_H, stride):
            hist = v[i - HIST + 1 : i + 1]
            fut = v[i + 1 : i + 1 + SPIKE_H]
            if np.isnan(hist).any() or np.isnan(fut[[H30 - 1, H60 - 1]]).any() or np.isnan(fut).mean() > 0.25:
                continue
            above = fut > THRESH
            onset = bool(hist[-1] <= THRESH and np.any(above[:-1] & above[1:]))
            r = {f"g_{k}": hist[-1 - k] for k in range(HIST)}
            r.update({"patient_id": pid, "hour_sin": np.sin(2 * np.pi * hours[i] / 24), "hour_cos": np.cos(2 * np.pi * hours[i] / 24),
                      "slope_30": (hist[-1] - hist[-3]) / 30, "slope_60": (hist[-1] - hist[-5]) / 60,
                      "mean_2h": hist.mean(), "sd_2h": hist.std(),
                      "y30": fut[H30 - 1], "y60": fut[H60 - 1], "spike": onset, "eligible_spike": hist[-1] <= THRESH})
            rows.append(r)
    return pd.DataFrame(rows)


CLOCK_FREE = [f"g_{k}" for k in range(HIST)] + ["slope_30", "slope_60", "mean_2h", "sd_2h"]
WITH_CLOCK = CLOCK_FREE + ["hour_sin", "hour_cos"]
FEATURE_SETS = {"CGM history only": CLOCK_FREE, "CGM history + time of day": WITH_CLOCK}
FEATURES = CLOCK_FREE


def fit(train: pd.DataFrame, features: list[str] | None = None) -> dict:
    features = features or FEATURES
    models = {"features": features}
    for target in ("y30", "y60"):
        # predict the change from the last reading; easier to transfer across glucose levels
        models[target] = lgb.train({**PARAMS, "objective": "regression_l1"},
                                   lgb.Dataset(train[features], train[target] - train["g_0"]), num_boost_round=300)
    el = train[train["eligible_spike"]]
    models["spike"] = lgb.train({**PARAMS, "objective": "binary"}, lgb.Dataset(el[features], el["spike"].astype(int)), num_boost_round=300)
    return models


def evaluate(models: dict | None, test: pd.DataFrame) -> dict:
    out = {"n_samples": int(len(test)), "n_patients": int(test["patient_id"].nunique())}
    for target, h in (("y30", 30), ("y60", 60)):
        if models is None:   # persistence
            pred = test["g_0"].to_numpy()
        elif models == "linear":
            pred = test["g_0"].to_numpy() + test["slope_30"].to_numpy() * h
        else:
            pred = test["g_0"].to_numpy() + models[target].predict(test[models["features"]])
        err = pred - test[target].to_numpy()
        out[f"mae_{h}"] = float(np.abs(err).mean())
        out[f"rmse_{h}"] = float(np.sqrt((err**2).mean()))
    el = test[test["eligible_spike"]]
    y = el["spike"].astype(int).to_numpy()
    out["spike_rate"] = float(y.mean())
    if models is None:
        score = el["g_0"].to_numpy()
    elif models == "linear":
        score = (el["g_0"] + el["slope_30"] * 120).to_numpy()
    else:
        score = models["spike"].predict(el[models["features"]])
    if 0 < y.sum() < len(y):
        out["spike_auroc"] = float(roc_auc_score(y, score))
        out["spike_pr_auc"] = float(average_precision_score(y, score))
    return out


def run(train_sets: dict[str, pd.DataFrame], test_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """train_sets/test_sets: name -> CGM frame (patient_id, timestamp, glucose_mgdl) at 15 min.

    Each training source is fitted twice: on CGM history only, and with time-of-day features.
    Clock features encode the local meal schedule (Indian home meals vs a Shanghai hospital
    timetable), so the clock-free variant is the fair transfer test."""
    test_samples = {name: samples(c) for name, c in test_sets.items()}
    rows = []
    for tname, ts in test_samples.items():
        rows.append({"train": "persistence (no training)", "features": "-", "test": tname, **evaluate(None, ts)})
        rows.append({"train": "linear extrapolation (no training)", "features": "-", "test": tname, **evaluate("linear", ts)})
    for name, cgm in train_sets.items():
        tr = samples(cgm)
        for fname, feats in FEATURE_SETS.items():
            models = fit(tr, feats)
            for tname, ts in test_samples.items():
                rows.append({"train": name, "features": fname, "test": tname, "n_train_samples": int(len(tr)), **evaluate(models, ts)})
    return pd.DataFrame(rows)
