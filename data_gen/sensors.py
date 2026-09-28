"""Sensor realism layer: CGM and wearable recordings from the true signals.

CGM: blood-to-interstitial lag (first-order, 5-15 min per patient), per-session calibration
error with drift, autocorrelated noise (higher on day 1 of a sensor), a 2-hour warm-up at each
new sensor, Bluetooth dropouts and longer gaps, compression lows while asleep, clipping to the
device range, 5-minute sampling at a patient-specific phase.

Wearable: 5-minute heart rate with timestamp jitter, charging and not-worn gaps (HR missing,
steps recorded as zero, the usual device behaviour), phantom steps, sleep stages passed through
a confusion matrix, nightly HRV with measurement noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from common import seeding
from data_gen.behaviour import STAGE_WAKE


def interstitial(glucose: np.ndarray, tau_min: float) -> np.ndarray:
    """First-order lag of a 1-minute plasma glucose trace."""
    a = 1.0 / tau_min
    y, _ = lfilter([a], [1.0, -(1.0 - a)], glucose, zi=[glucose[0] * (1.0 - a)])
    return y


def cgm_for_patient(pid: str, glucose: np.ndarray, sleep_stage: np.ndarray, cfg: dict, seed: int,
                    first_minute: int = 0) -> tuple[pd.DataFrame, dict]:
    """Recorded CGM for one patient. `glucose` is the true 1-min plasma glucose."""
    c = cfg["cgm"]
    r = seeding.rng(seed, "sensors", "cgm", pid)
    tau = float(r.uniform(*c["lag_tau_min"]))
    ig = interstitial(glucose.astype(float), tau)
    step = c["interval_min"]
    phase = int(r.integers(0, step))
    minutes = np.arange(first_minute + phase, len(glucose), step)
    n = len(minutes)
    x = ig[minutes]
    days = minutes / 1440.0

    # sensor sessions
    first_start = first_minute / 1440.0 + r.uniform(*c["first_session_offset_days"])
    session = np.floor((days - first_start) / c["session_days"]).astype(int)
    since_start = days - (first_start + session * c["session_days"])
    gain = np.empty(n)
    offset = np.empty(n)
    for s in np.unique(session):
        m = session == s
        t = since_start[m]
        g0, o0 = r.normal(1.0, c["gain_sd"]), r.normal(0.0, c["offset_sd_mgdl"])
        gslope, oslope = r.normal(0, c["gain_drift_sd_per_day"]), r.normal(0, c["offset_drift_sd_per_day"])
        gain[m] = g0 + gslope * t
        offset[m] = o0 + oslope * t

    # One autocorrelated error process whose size grows with glucose (absolute + relative part),
    # larger on the first day of a sensor. Real CGM error is strongly autocorrelated.
    noise_sd = (c["noise_sd_mgdl"] + c["noise_multiplicative_sd"] * x) * np.where(since_start < 1.0, c["first_day_noise_factor"], 1.0)
    ar = lfilter([np.sqrt(1 - c["noise_rho"] ** 2)], [1, -c["noise_rho"]], r.normal(0, 1, n))
    y = gain * x + offset + noise_sd * ar

    # compression lows while asleep
    asleep = sleep_stage[minutes] >= 1
    night_id = (minutes + 12 * 60) // 1440
    n_comp = 0
    for nid in np.unique(night_id[asleep]):
        if r.random() >= c["compression_low_p_per_night"]:
            continue
        idx = np.flatnonzero(asleep & (night_id == nid))
        start = int(r.choice(idx))
        dur = int(r.uniform(*c["compression_duration_min"]) / step)
        depth = r.uniform(*c["compression_depth_mgdl"])
        seg = np.arange(start, min(start + dur, n))
        if len(seg):
            y[seg] -= depth * np.sin(np.pi * (np.arange(len(seg)) + 0.5) / len(seg))
            n_comp += 1

    flag = np.full(n, "ok", dtype=object)
    missing = np.zeros(n, dtype=bool)
    warm = since_start < c["warmup_h"] / 24.0
    missing |= warm
    flag[warm] = "warmup"
    n_days = max(1.0, n * step / 1440.0)
    for _ in range(r.poisson(c["dropouts_per_day"] * n_days)):
        s0 = int(r.integers(0, n))
        missing[s0 : s0 + int(r.geometric(1 / c["dropout_mean_samples"]))] = True
    for _ in range(r.poisson(c["long_gaps_per_day"] * n_days)):
        s0 = int(r.integers(0, n))
        missing[s0 : s0 + int(r.integers(*c["long_gap_samples"]))] = True
    flag[missing & (flag == "ok")] = "gap"

    lo, hi = c["range_mgdl"]
    y = np.round(y)
    flag[(y < lo) & ~missing] = "low"
    flag[(y > hi) & ~missing] = "high"
    y = np.clip(y, lo, hi)
    y[missing] = np.nan
    df = pd.DataFrame({"patient_id": pid, "minute": minutes, "glucose_mgdl": y, "flag": flag,
                       "sensor_session": session - session.min()})
    return df, {"patient_id": pid, "cgm_lag_tau_min": round(tau, 2), "cgm_phase_min": phase,
                "cgm_compression_events": n_comp}


def wearable_for_patient(pid: str, steps: np.ndarray, heart_rate: np.ndarray, sleep_stage: np.ndarray,
                         hrv_true: pd.DataFrame, cfg: dict, seed: int, first_minute: int = 0) -> dict:
    """Recorded wearable streams for one patient (5-minute resolution)."""
    w = cfg["wearable"]
    r = seeding.rng(seed, "sensors", "wearable", pid)
    T = len(steps)
    worn = np.ones(T, dtype=bool)
    n_days = T // 1440
    for d in range(n_days):
        base = d * 1440
        if r.random() < w["p_charge_per_day"]:
            s0 = base + int(r.choice([r.uniform(7 * 60, 9 * 60), r.uniform(19 * 60, 22 * 60)]))
            worn[s0 : s0 + int(r.uniform(*w["charging_min"]))] = False
        if r.random() < w["p_removed_per_day"]:
            s0 = base + int(r.uniform(9 * 60, 18 * 60))
            worn[s0 : s0 + int(r.uniform(*w["removed_h"]) * 60)] = False
        if r.random() < w["p_not_worn_night"]:
            worn[base + 22 * 60 : base + 31 * 60] = False

    steps_rec = np.where(worn, steps, 0).astype(float)
    for d in range(n_days):
        if r.random() < w["phantom_steps_p_per_day"]:
            s0 = d * 1440 + int(r.uniform(8 * 60, 21 * 60))
            dur = int(r.uniform(*w["phantom_duration_min"]))
            awake_worn = worn[s0 : s0 + dur] & (sleep_stage[s0 : s0 + dur] < 0)
            steps_rec[s0 : s0 + dur] += np.where(awake_worn, r.uniform(*w["phantom_cadence"]), 0).round()

    step = w["interval_min"]
    starts = np.arange(first_minute, T - step + 1, step)
    blocks = np.arange(step)[None, :] + starts[:, None]
    worn_frac = worn[blocks].mean(1)
    hr = heart_rate[blocks].mean(1) + r.normal(0, w["hr_noise_sd"], len(starts))
    hr[(worn_frac < 0.6) | (r.random(len(starts)) < w["random_missing_hr"])] = np.nan
    jitter = r.integers(-w["timestamp_jitter_s"], w["timestamp_jitter_s"] + 1, len(starts))
    hr_df = pd.DataFrame({"patient_id": pid, "minute": starts, "jitter_s": jitter, "heart_rate_bpm": np.round(hr, 1)})
    hr_df = hr_df.dropna(subset=["heart_rate_bpm"])
    steps_df = pd.DataFrame({"patient_id": pid, "minute": starts, "steps": steps_rec[blocks].sum(1).astype(int)})

    # Sleep stages, 5-min epochs, only when the device was worn
    conf = np.asarray(w["sleep_stage_confusion"], dtype=float)
    ep = sleep_stage[blocks]
    in_bed = (ep >= STAGE_WAKE).sum(1) >= 3
    true_stage = np.array([np.bincount(row[row >= 0], minlength=4).argmax() if ok else -1 for row, ok in zip(ep, in_bed, strict=True)])
    recorded = np.full(len(starts), -1)
    ok = in_bed & (worn_frac >= 0.6)
    cum = conf.cumsum(1)
    u = r.random(len(starts))
    for s in range(4):
        m = ok & (true_stage == s)
        recorded[m] = (u[m][:, None] > cum[s][None, :]).sum(1)
    sleep_df = pd.DataFrame({"patient_id": pid, "minute": starts[ok], "stage": recorded[ok].astype(np.int8)})

    hv = hrv_true[(hrv_true["minute"] >= first_minute)].copy()
    worn_at = worn[np.clip(hv["minute"].to_numpy(), 0, T - 1)]
    hv = hv[worn_at]
    hv["rmssd_ms"] = np.round(hv["rmssd_ms"].to_numpy() * np.exp(r.normal(0, w["hrv_noise_sd_log"], len(hv))), 1)
    return {"heart_rate": hr_df, "steps": steps_df, "sleep_stages": sleep_df, "hrv": hv.reset_index(drop=True),
            "worn_fraction": float(worn[first_minute:].mean())}
