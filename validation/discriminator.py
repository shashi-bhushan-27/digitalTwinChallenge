"""Real-vs-synthetic discriminator: can a classifier tell 12-hour CGM windows apart?

Accuracy near 50 % (AUC near 0.5) means the synthetic windows are hard to distinguish from real
ones. Cross-validation is grouped by patient so a patient's windows never sit on both sides.
Windows start at 00:00 and 12:00 in both sets so their start time carries no signal.
Two variants are run: level-matched (each real patient paired with the synthetic patient of the
closest mean CGM, the fair test of dynamics) and unmatched (the main synthetic cohort, which
also measures the deliberate glucose-level difference).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from validation.windows import regular_series, window_features, windows

WINDOW_LEN = 48   # 12 h of 15-minute readings
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0,
          "deterministic": True, "num_threads": 4}


def build(real_cgm: pd.DataFrame, syn_cgm: pd.DataFrame, seed: int, max_per_patient: int = 30) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames, labels, groups = [], [], []
    for label, cgm in ((1, real_cgm), (0, syn_cgm)):
        x, pids, starts = windows(regular_series(cgm), WINDOW_LEN, WINDOW_LEN, align_hours=(0, 12))
        pids = np.array(pids)
        keep = []
        for pid in np.unique(pids):
            idx = np.flatnonzero(pids == pid)
            keep.extend(rng.choice(idx, size=min(len(idx), max_per_patient), replace=False))
        keep = np.sort(np.array(keep, dtype=int))
        frames.append(window_features(x[keep]))
        labels.append(np.full(len(keep), label))
        groups.append(np.array([f"{'R' if label else 'S'}:{p}" for p in pids[keep]]))
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)
    g = np.concatenate(groups)
    # balance the classes by subsampling windows of the larger class
    n = min((y == 1).sum(), (y == 0).sum())
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=n, replace=False) for c in (0, 1)])
    return X.iloc[idx].reset_index(drop=True), y[idx], g[idx]


def run(real_cgm: pd.DataFrame, syn_cgm: pd.DataFrame, seed: int = 0, n_splits: int = 5) -> dict:
    X, y, g = build(real_cgm, syn_cgm, seed)
    pred = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, g):
        model = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=200)
        pred[te] = model.predict(X.iloc[te])
        imp += model.feature_importance("gain")
    importance = pd.Series(imp / imp.sum(), index=X.columns).sort_values(ascending=False)
    top = list(importance.index[:4])
    return {
        "top_feature_medians": {f: {"real": float(X.loc[y == 1, f].median()), "synthetic": float(X.loc[y == 0, f].median())} for f in top},
        "n_windows": int(len(y)), "n_real_patients": int(len({x for x in g if x.startswith("R:")})),
        "n_synthetic_patients": int(len({x for x in g if x.startswith("S:")})),
        "accuracy": float(accuracy_score(y, pred > 0.5)), "auc": float(roc_auc_score(y, pred)),
        "top_features": importance.head(8).round(3).to_dict(),
    }
