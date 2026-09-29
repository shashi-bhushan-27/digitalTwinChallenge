"""Evaluation metrics: forecast error, Clarke error grid, event detection, calibration, alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def forecast_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(pred)
    e = pred[ok] - y[ok]
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())), "n": int(ok.sum())}


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke error grid zone (A-E) for each (reference, prediction) pair in mg/dL.

    Standard zone boundaries (Clarke et al., Diabetes Care 1987); implemented from the usual
    published definitions, to verify against the paper."""
    r = np.asarray(ref, float)
    p = np.asarray(pred, float)
    z = np.full(len(r), "B", dtype=object)
    a = (np.abs(p - r) <= 0.2 * r) | ((r < 70) & (p < 70))
    e = ((r <= 70) & (p >= 180)) | ((r >= 180) & (p <= 70))
    d = ((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 70) & (p >= 70) & (p <= 180))
    c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | ((r >= 130) & (r <= 180) & (p <= (7 / 5) * r - 182))
    z[c] = "C"
    z[d] = "D"
    z[e] = "E"
    z[a] = "A"
    return z


def clarke_summary(ref: np.ndarray, pred: np.ndarray) -> dict:
    ok = ~np.isnan(ref) & ~np.isnan(pred)
    z = clarke_zones(ref[ok], pred[ok])
    return {f"clarke_{k}": float(np.mean(z == k)) for k in "ABCDE"}


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, int)
    out = {"prevalence": float(y.mean()), "n": int(len(y))}
    if 0 < y.sum() < len(y):
        out.update({"pr_auc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
                    "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))})
    return out


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = np.asarray(y, bool)
    a = p >= thr
    tp, fp, fn = (a & y).sum(), (a & ~y).sum(), (~a & y).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"threshold": float(thr), "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9)), "alert_rate": float(a.mean())}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [at_threshold(y, p, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1))])


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(np.mean(y[m])), "n": int(m.sum())})
    return pd.DataFrame(rows)


def alert_events(stream: pd.DataFrame, prob_col: str, thr: float, window_min: int = 120, refractory_min: int = 60) -> dict:
    """Event-level evaluation on a continuous 5-min stream (one patient block after another).

    stream columns: patient_id, timestamp, spike_eligible, spike_onset_event, and `prob_col`.
    An alert fires at eligible times with probability >= thr. Consecutive alerts form one
    episode unless separated by more than `refractory_min`. A true onset is detected if an alert
    fired in the `window_min` before it; lead time is measured from the first such alert. An
    episode is false if no onset follows within `window_min` of its start."""
    detected, lead_times, n_onsets, false_eps, n_eps = 0, [], 0, 0, 0
    patient_days = 0.0
    w = np.timedelta64(window_min, "m")
    rf = np.timedelta64(refractory_min, "m")
    for _, d in stream.groupby("patient_id", sort=False):
        d = d.sort_values("timestamp")
        t = d["timestamp"].to_numpy().astype("datetime64[ns]")
        patient_days += (t[-1] - t[0]) / np.timedelta64(1, "D")
        alert = (d["spike_eligible"].to_numpy(bool)) & (d[prob_col].to_numpy() >= thr)
        at = t[alert]
        onsets = t[d["spike_onset_event"].to_numpy(bool)]
        n_onsets += len(onsets)
        for o in onsets:
            prior = at[(at >= o - w) & (at < o)]
            if len(prior):
                detected += 1
                lead_times.append((o - prior.min()) / np.timedelta64(1, "m"))
        if len(at):
            starts = at[np.concatenate([[True], np.diff(at) > rf])]
            n_eps += len(starts)
            for s0 in starts:
                if not np.any((onsets > s0) & (onsets <= s0 + w)):
                    false_eps += 1
    return {
        "threshold": float(thr), "onsets": int(n_onsets), "onsets_detected": int(detected),
        "event_recall": detected / max(n_onsets, 1), "median_lead_time_min": float(np.median(lead_times)) if lead_times else float("nan"),
        "mean_lead_time_min": float(np.mean(lead_times)) if lead_times else float("nan"),
        "alert_episodes": int(n_eps), "false_alert_episodes": int(false_eps),
        "episode_precision": 1 - false_eps / max(n_eps, 1),
        "false_alerts_per_patient_day": false_eps / max(patient_days, 1e-9), "patient_days": float(patient_days),
    }
