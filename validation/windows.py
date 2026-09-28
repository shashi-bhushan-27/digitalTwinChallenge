"""Fixed-length CGM windows and hand-crafted window features (shared by the discriminator and TSTR)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regular_series(cgm: pd.DataFrame, step_min: int = 15) -> dict[str, pd.Series]:
    """Per-patient series on a regular grid (NaN where missing)."""
    out = {}
    for pid, d in cgm.dropna(subset=["glucose_mgdl"]).groupby("patient_id"):
        s = d.set_index("timestamp")["glucose_mgdl"].sort_index()
        s = s[~s.index.duplicated()]
        s.index = s.index.floor(f"{step_min}min")
        s = s[~s.index.duplicated()]
        grid = pd.date_range(s.index.min(), s.index.max(), freq=f"{step_min}min")
        out[pid] = s.reindex(grid)
    return out


def windows(series: dict[str, pd.Series], length: int, stride: int, max_missing: float = 0.1,
            align_hours: tuple[int, ...] | None = None) -> tuple[np.ndarray, list[str], list[pd.Timestamp]]:
    """Strided windows with at most `max_missing` gaps, linearly filled. With `align_hours`,
    windows only start at those clock hours (so start time carries no information)."""
    xs, pids, starts = [], [], []
    for pid, s in series.items():
        v = s.to_numpy(float)
        if align_hours is not None:
            idx_all = np.flatnonzero(np.isin(s.index.hour, align_hours) & (s.index.minute == 0))
        else:
            idx_all = np.arange(0, len(v) - length + 1, stride)
        for i in idx_all:
            if i + length > len(v):
                continue
            w = v[i : i + length]
            miss = np.isnan(w)
            if miss.mean() > max_missing or miss[0] or miss[-1]:
                continue
            if miss.any():
                idx = np.arange(length)
                w = w.copy()
                w[miss] = np.interp(idx[miss], idx[~miss], w[~miss])
            xs.append(w)
            pids.append(pid)
            starts.append(s.index[i])
    return (np.array(xs) if xs else np.empty((0, length))), pids, starts


def window_features(x: np.ndarray, starts: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Summary features of each window row (level, spread, shape, smoothness, rhythm)."""
    d1 = np.diff(x, axis=1)
    f = {
        "mean": x.mean(1), "sd": x.std(1), "cv": x.std(1) / x.mean(1), "min": x.min(1), "max": x.max(1),
        "p10": np.percentile(x, 10, axis=1), "p90": np.percentile(x, 90, axis=1),
        "tir": ((x >= 70) & (x <= 180)).mean(1), "tar": (x > 180).mean(1), "tbr": (x < 70).mean(1),
        "diff_sd": d1.std(1), "diff_abs_mean": np.abs(d1).mean(1), "diff_max": d1.max(1), "diff_min": d1.min(1),
        "second_diff_sd": np.diff(x, n=2, axis=1).std(1),
        "n_turning": (np.diff(np.sign(d1), axis=1) != 0).sum(1),
    }
    xc = x - x.mean(1, keepdims=True)
    denom = (xc**2).sum(1) + 1e-9
    for lag in (1, 2, 4, 8):
        f[f"acf_{lag}"] = (xc[:, lag:] * xc[:, :-lag]).sum(1) / denom
    spec = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    tot = spec[:, 1:].sum(1) + 1e-9
    k = spec.shape[1]
    f["power_low"] = spec[:, 1 : max(2, k // 8)].sum(1) / tot
    f["power_high"] = spec[:, k // 2 :].sum(1) / tot
    out = pd.DataFrame(f)
    if starts is not None:
        hrs = np.array([t.hour + t.minute / 60 for t in starts])
        out["start_hour_sin"] = np.sin(2 * np.pi * hrs / 24)
        out["start_hour_cos"] = np.cos(2 * np.pi * hrs / 24)
    return out
