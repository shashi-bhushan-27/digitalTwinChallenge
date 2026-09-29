"""Glycaemic metrics used to compare synthetic and real CGM.

All functions take plain arrays or tidy frames (patient_id, timestamp, glucose_mgdl) and ignore
missing readings. Definitions follow the international consensus on CGM metrics (Battelino et al.,
Diabetes Care 2019; to verify): TIR 70-180, TAR >180 / >250, TBR <70 / <54 mg/dL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.units import gmi_pct


def resample(cgm: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    """First reading in each `minutes` bin per patient (brings 5-min synthetic data to 15-min)."""
    df = cgm.dropna(subset=["glucose_mgdl"]).copy()
    df["bin"] = df["timestamp"].dt.floor(f"{minutes}min")
    out = df.sort_values("timestamp").groupby(["patient_id", "bin"], as_index=False).first()
    return out.drop(columns=["timestamp"]).rename(columns={"bin": "timestamp"})


def mage(x: np.ndarray) -> float:
    """Mean amplitude of glycaemic excursions: mean size of peak-to-nadir swings larger than 1 SD.

    Turning points are taken from the sign changes of the first difference; swings smaller than
    1 SD are merged into their neighbours before averaging (both directions)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan
    sd = x.std()
    d = np.sign(np.diff(x))
    for i in range(1, len(d)):  # carry flat segments forward
        if d[i] == 0:
            d[i] = d[i - 1]
    # interior turning points only: the series ends are not peaks or nadirs
    tp = [i + 1 for i in range(len(d) - 1) if d[i] != d[i + 1]]
    if len(tp) < 2:
        return np.nan
    ext = list(x[tp])
    changed = True
    while changed and len(ext) > 2:
        changed = False
        amps = np.abs(np.diff(ext))
        i = int(np.argmin(amps))
        if amps[i] < sd:
            # remove the smaller swing's two points (or one at the ends) and re-merge monotone runs
            if i == 0:
                del ext[1]
            elif i == len(amps) - 1:
                del ext[-2]
            else:
                del ext[i : i + 2]
            merged = [ext[0]]
            for v in ext[1:]:
                if len(merged) >= 2 and (merged[-1] - merged[-2]) * (v - merged[-1]) > 0:
                    merged[-1] = v
                else:
                    merged.append(v)
            ext = merged
            changed = True
    amps = np.abs(np.diff(ext))
    amps = amps[amps >= sd]
    return float(amps.mean()) if len(amps) else np.nan


def summary(g: np.ndarray) -> dict:
    g = np.asarray(g, float)
    g = g[~np.isnan(g)]
    if len(g) == 0:
        return {}
    mean = g.mean()
    return {
        "n_readings": len(g), "mean": mean, "sd": g.std(), "cv": g.std() / mean, "gmi": float(gmi_pct(mean)),
        "tir_70_180": np.mean((g >= 70) & (g <= 180)), "tar_180": np.mean(g > 180), "tar_250": np.mean(g > 250),
        "tbr_70": np.mean(g < 70), "tbr_54": np.mean(g < 54),
        "p10": np.percentile(g, 10), "p50": np.percentile(g, 50), "p90": np.percentile(g, 90),
    }


def patient_summaries(cgm: pd.DataFrame) -> pd.DataFrame:
    """Per-patient CGM summary, MAGE averaged over calendar days, nocturnal (03-06 h) mean."""
    rows = []
    for pid, d in cgm.groupby("patient_id", sort=False):
        d = d.dropna(subset=["glucose_mgdl"])
        s = summary(d["glucose_mgdl"].to_numpy())
        if not s:
            continue
        day_mage = [mage(x["glucose_mgdl"].to_numpy()) for _, x in d.groupby(d["timestamp"].dt.date) if len(x) >= 48]
        s["mage"] = float(np.nanmean(day_mage)) if day_mage and not np.all(np.isnan(day_mage)) else np.nan
        h = d["timestamp"].dt.hour
        s["nocturnal_mean"] = d.loc[(h >= 3) & (h < 6), "glucose_mgdl"].mean()
        s["nocturnal_to_mean"] = s["nocturnal_mean"] / s["mean"]
        daily = d.groupby(d["timestamp"].dt.date)["glucose_mgdl"].mean()
        s["day_to_day_cv"] = daily.std() / daily.mean() if len(daily) > 2 else np.nan
        # lag-1 autocorrelation of first differences (smoothness of the trace)
        diffs = d.set_index("timestamp")["glucose_mgdl"].diff().dropna().to_numpy()
        s["diff_sd"] = diffs.std() if len(diffs) > 2 else np.nan
        # tails of the step changes: how fast glucose falls after peaks and rises after meals
        s["diff_p05"] = np.percentile(diffs, 5) if len(diffs) > 20 else np.nan
        s["diff_p95"] = np.percentile(diffs, 95) if len(diffs) > 20 else np.nan
        # roughness: SD of second differences (sensor noise and smoothing show up here)
        s["diff2_sd"] = np.diff(diffs).std() if len(diffs) > 3 else np.nan
        s["patient_id"] = pid
        rows.append(s)
    return pd.DataFrame(rows).set_index("patient_id")


def postprandial(cgm: pd.DataFrame, meals: pd.DataFrame, window_min: int = 180, isolation_min: int = 120) -> pd.DataFrame:
    """Rise and time-to-peak after each meal with enough CGM coverage.

    Baseline = reading closest to the meal time within 20 min before it. Meals followed by
    another meal within `isolation_min` are skipped so each excursion belongs to one meal."""
    rows = []
    cg = {pid: d.dropna(subset=["glucose_mgdl"]).sort_values("timestamp") for pid, d in cgm.groupby("patient_id")}
    for pid, m in meals.sort_values("timestamp").groupby("patient_id"):
        if pid not in cg:
            continue
        d = cg[pid]
        ts = d["timestamp"].to_numpy()
        gv = d["glucose_mgdl"].to_numpy()
        mt = m["timestamp"].to_numpy()
        for j, t in enumerate(mt):
            if j + 1 < len(mt) and (mt[j + 1] - t) < np.timedelta64(isolation_min, "m"):
                continue
            pre = (ts <= t) & (ts >= t - np.timedelta64(20, "m"))
            post = (ts > t) & (ts <= t + np.timedelta64(window_min, "m"))
            if pre.sum() == 0 or post.sum() < 0.7 * window_min / 15:
                continue
            base = gv[pre][-1]
            k = int(np.argmax(gv[post]))
            rows.append({"patient_id": pid, "timestamp": t, "baseline": base, "peak": gv[post][k],
                         "rise": gv[post][k] - base, "time_to_peak_min": (ts[post][k] - t) / np.timedelta64(1, "m")})
    return pd.DataFrame(rows)


def spike_onsets_per_day(cgm: pd.DataFrame, threshold: float = 180.0, sustain_min: int = 15) -> pd.Series:
    """Upward crossings of `threshold` that stay above it for `sustain_min`, per patient-day."""
    out = {}
    for pid, d in cgm.groupby("patient_id"):
        d = d.dropna(subset=["glucose_mgdl"]).sort_values("timestamp")
        above = (d["glucose_mgdl"] > threshold).to_numpy()
        step = np.median(np.diff(d["timestamp"].to_numpy()) / np.timedelta64(1, "m")) if len(d) > 2 else 15
        need = int(sustain_min // step) + 1   # readings spanning at least `sustain_min`
        count = 0
        i = 1
        while i < len(above):
            if above[i] and not above[i - 1] and above[i : i + need].all():
                count += 1
                while i < len(above) and above[i]:
                    i += 1
            i += 1
        days = (d["timestamp"].max() - d["timestamp"].min()).total_seconds() / 86400
        out[pid] = count / max(days, 1e-9)
    return pd.Series(out, name="spikes_per_day")
