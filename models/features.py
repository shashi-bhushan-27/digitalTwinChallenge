"""Feature builder: observed tables only -> one row per (patient, prediction time).

    uv run python -m models.features        # writes data/features/<version>/features.parquet

Prediction times sit on each patient's 5-minute CGM grid wherever a current reading exists and
at least 6 hours of history are available (every 15 min for train/val patients, every 5 min for
test patients so alerts can be evaluated as a continuous stream). Every feature at time t uses
data at or before t only. Feature names carry their group as a prefix:

    cgm_*  CGM history (level, lags, trends, variability, excursion shape, data gaps)
    tod_*  time of day / weekend
    wear_* wearables (steps, heart rate, last night's sleep and HRV)
    meal_* the patient's meal log (what they chose to log, with their own carb estimates)
    ehr_*  static record and prescription schedule

Targets: glucose at each horizon, peak glucose over the next 120 min, and spike onset
(from <= 180 mg/dL, above 180 for >= 15 min within 120 min; decision D-004).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import config
from common.io import write_parquet

GROUP_PREFIXES = {"cgm": ("cgm_", "tod_"), "wear": ("wear_",), "meal": ("meal_",), "ehr": ("ehr_",)}
TARGET_COLS = ["patient_id", "timestamp", "split", "g_now", "spike_label", "spike_eligible", "spike_onset_event", "peak_120"]


def dataset_root(cfgs: dict | None = None):
    from data_gen.generate import load_configs

    cfgs = cfgs or load_configs()
    return config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"] / cfgs["pipeline"]["dataset_version"]


def load_observed(root) -> dict[str, pd.DataFrame]:
    names = ["patients", "medications", "cgm", "wearable_heart_rate", "wearable_steps", "wearable_sleep_stages",
             "wearable_hrv", "meal_log", "splits"]
    return {n: pd.read_parquet(root / "observed" / f"{n}.parquet") for n in names}


def feature_columns(df: pd.DataFrame, groups: tuple[str, ...]) -> list[str]:
    prefixes = tuple(p for g in groups for p in GROUP_PREFIXES[g])
    return [c for c in df.columns if c.startswith(prefixes)]


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _ns(ts) -> np.ndarray:
    """Timestamps as int64 nanoseconds, whatever unit pandas loaded them in."""
    return pd.DatetimeIndex(ts).as_unit("ns").asi8


def _window_sum(ts: np.ndarray, vals: np.ndarray, at: np.ndarray, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of `vals` with timestamps in (at - minutes, at]. ts sorted, int64 ns."""
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(ts, at, side="right")
    lo = np.searchsorted(ts, at - np.int64(minutes) * 60_000_000_000, side="right")
    return cs[hi] - cs[lo], (hi - lo).astype(float)


def _forward_window(x: np.ndarray, n: int, fn: str, min_count: int = 1) -> np.ndarray:
    """fn over x[i+1 .. i+n] for each i (NaN-aware), NaN if fewer than min_count values."""
    pad = np.concatenate([x[1:], np.full(n, np.nan)])
    w = sliding_window_view(pad, n)[: len(x)]
    cnt = np.sum(~np.isnan(w), axis=1)
    with np.errstate(all="ignore"):
        out = np.nanmax(w, axis=1) if fn == "max" else np.nanmean(w, axis=1)
    out[cnt < min_count] = np.nan
    return out


def _back_window(x: np.ndarray, n: int) -> np.ndarray:
    """Windows x[i-n+1 .. i] (left-padded with NaN)."""
    pad = np.concatenate([np.full(n - 1, np.nan), x])
    return sliding_window_view(pad, n)


def _sleep_episodes(stages: pd.DataFrame, hr: pd.DataFrame, hrv: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    if stages.empty:
        return pd.DataFrame(columns=["start", "end", "total", "deep", "rem", "wake", "hrv", "hr"])
    s = stages.sort_values("timestamp")
    t = s["timestamp"].to_numpy()
    new = np.concatenate([[True], np.diff(t) > np.timedelta64(fcfg["sleep_gap_merge_min"], "m")])
    ep_id = np.cumsum(new)
    rows = []
    hr_t = hr["timestamp"].to_numpy().astype("datetime64[ns]")
    hrv_t = hrv["timestamp"].to_numpy().astype("datetime64[ns]")
    for _, g in s.groupby(ep_id):
        start, end = g["timestamp"].iat[0], g["timestamp"].iat[-1] + pd.Timedelta(minutes=5)
        if (end - start) < pd.Timedelta(hours=fcfg["sleep_episode_min_h"]):
            continue
        st = g["stage"].to_numpy()
        s0, e0 = np.datetime64(start, "ns"), np.datetime64(end, "ns")
        m_hr = (hr_t >= s0) & (hr_t < e0)
        m_hv = (hrv_t >= s0) & (hrv_t < e0)
        rows.append({"start": start, "end": end, "total": 5.0 * np.isin(st, [1, 2, 3]).sum(), "deep": 5.0 * (st == 2).sum(),
                     "rem": 5.0 * (st == 3).sum(), "wake": 5.0 * (st == 0).sum(),
                     "hrv": hrv["rmssd_ms"].to_numpy()[m_hv].mean() if m_hv.any() else np.nan,
                     "hr": hr["heart_rate_bpm"].to_numpy()[m_hr].mean() if m_hr.any() else np.nan})
    ep = pd.DataFrame(rows)
    if len(ep):
        # HRV relative to the patient's own previous nights (past data only)
        ep["hrv_ref"] = ep["hrv"].expanding().median().shift(1)
        ep["hr_ref"] = ep["hr"].expanding().median().shift(1)
    return ep


def _minutes_since_clock(ts: pd.DatetimeIndex, clocks: list[str]) -> np.ndarray:
    """Minutes since the most recent scheduled clock time (today or yesterday)."""
    mod = (ts.hour * 60 + ts.minute).to_numpy()
    best = np.full(len(ts), np.inf)
    for c in clocks:
        h, m = map(int, c.split(":"))
        d = (mod - (h * 60 + m)) % 1440
        best = np.minimum(best, d)
    return best


# ----------------------------------------------------------------------------------------------
# per-patient features
# ----------------------------------------------------------------------------------------------


def ehr_static(patients: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    p = patients.set_index("patient_id")
    out = pd.DataFrame(index=p.index)
    for col in ["age_yr", "bmi", "waist_cm", "hba1c_pct", "hba1c_trend_12m", "fpg_mgdl", "duration_yr", "egfr",
                "alt_ul", "tg_mgdl", "hdl_mgdl", "ldl_mgdl", "sbp_mmhg", "adherence_reported", "n_glucose_lowering_drugs"]:
        out[f"ehr_{col}"] = p[col].astype(float)
    out["ehr_male"] = (p["sex"] == "M").astype(float)
    for col in ["fatty_liver", "hypertension", "ckd", "peripheral_neuropathy", "coronary_artery_disease"]:
        out[f"ehr_{col}"] = p[col].astype(float)
    out["ehr_activity_level"] = p["activity_level"].map({"sedentary": 0, "lightly_active": 1, "moderately_active": 2}).astype(float)
    out["ehr_vegetarian"] = p["diet_type"].isin(["vegetarian", "jain"]).astype(float)
    out["ehr_region"] = p["region"].map({"north": 0, "south": 1, "west": 2, "east": 3}).astype(float)
    su_eq = {"glimepiride": 4.0, "gliclazide_mr": 120.0}
    agg = {"ehr_met_g_day": 0.0, "ehr_su_eq_day": 0.0, "ehr_dpp4": 0.0, "ehr_sglt2": 0.0, "ehr_basal_u_per_kg": 0.0}
    for k, v in agg.items():
        out[k] = v
    for m in meds.itertuples(index=False):
        if m.drug_class == "biguanide":
            out.loc[m.patient_id, "ehr_met_g_day"] += m.daily_dose / 1000
        elif m.drug_class == "sulfonylurea":
            out.loc[m.patient_id, "ehr_su_eq_day"] += m.daily_dose / su_eq.get(m.drug, m.daily_dose)
        elif m.drug_class == "dpp4_inhibitor":
            out.loc[m.patient_id, "ehr_dpp4"] = 1.0
        elif m.drug_class == "sglt2_inhibitor":
            out.loc[m.patient_id, "ehr_sglt2"] = 1.0
        elif m.drug_class == "basal_insulin":
            out.loc[m.patient_id, "ehr_basal_u_per_kg"] += m.daily_dose / p.loc[m.patient_id, "weight_kg"]
    return out


def patient_features(pid: str, obs: dict, static: pd.Series, split: str, mcfg: dict) -> pd.DataFrame:
    """Features for one patient; `obs` holds that patient's rows of each observed table."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows are expected at gaps
        return _patient_features(pid, obs, static, split, mcfg)


def _patient_features(pid: str, obs: dict, static: pd.Series, split: str, mcfg: dict) -> pd.DataFrame:
    fcfg, scfg, spk = mcfg["features"], mcfg["sampling"], mcfg["spike"]
    step = scfg["grid_min"]
    c = obs["cgm"].sort_values("timestamp")
    idx = pd.DatetimeIndex(c["timestamp"]).as_unit("ns")
    g = c["glucose_mgdl"].to_numpy(float)
    n = len(g)
    gf = pd.Series(g).ffill(limit=2).to_numpy()
    f: dict[str, np.ndarray] = {"cgm_now": g}

    # ---- CGM history
    for lag in fcfg["cgm_lags_min"]:
        k = lag // step
        f[f"cgm_lag_{lag}"] = np.concatenate([np.full(k, np.nan), gf[:-k]])
    for d in (15, 30, 60):
        f[f"cgm_delta_{d}"] = g - f[f"cgm_lag_{d}"]
    f["cgm_accel_15"] = (g - f["cgm_lag_15"]) - (f["cgm_lag_15"] - f["cgm_lag_30"])
    for w in fcfg["cgm_windows_min"]:
        win = _back_window(g, w // step)
        with np.errstate(all="ignore"):
            f[f"cgm_mean_{w}"] = np.nanmean(win, 1)
            f[f"cgm_sd_{w}"] = np.nanstd(win, 1)
            f[f"cgm_min_{w}"] = np.nanmin(win, 1)
            f[f"cgm_max_{w}"] = np.nanmax(win, 1)
    w24 = _back_window(g, 288)
    with np.errstate(all="ignore"):
        valid = np.sum(~np.isnan(w24), 1)
        f["cgm_mean_24h"] = np.where(valid > 144, np.nanmean(w24, 1), np.nan)
        f["cgm_tir_24h"] = np.where(valid > 144, np.nansum((w24 >= 70) & (w24 <= 180), 1) / np.maximum(valid, 1), np.nan)
        f["cgm_tar_24h"] = np.where(valid > 144, np.nansum(w24 > 180, 1) / np.maximum(valid, 1), np.nan)
    w3 = _back_window(np.where(np.isnan(gf), np.inf, gf), 36)
    amin = np.argmin(w3, 1)
    f["cgm_min_since_3h_low"] = (35 - amin) * float(step)
    f["cgm_rise_from_3h_low"] = g - f["cgm_min_180"]
    f["cgm_missing_2h"] = np.mean(np.isnan(_back_window(g, 24)), 1)
    f["cgm_minutes_above_180_2h"] = np.nansum(_back_window(g, 24) > 180, 1) * float(step)
    tod = (idx.hour * 60 + idx.minute).to_numpy() / 1440.0
    f["tod_sin"] = np.sin(2 * np.pi * tod)
    f["tod_cos"] = np.cos(2 * np.pi * tod)
    f["tod_weekend"] = (idx.dayofweek >= 5).astype(float)

    # ---- wearables
    at = idx.asi8
    s = obs["wearable_steps"].sort_values("timestamp")
    s_t, s_v = _ns(s["timestamp"]), s["steps"].to_numpy(float)
    for w in fcfg["steps_windows_min"]:
        f[f"wear_steps_{w}"] = _window_sum(s_t, s_v, at, w)[0]
    midnight = _ns(idx.normalize())
    cs = np.concatenate([[0.0], np.cumsum(s_v)])
    f["wear_steps_today"] = cs[np.searchsorted(s_t, at, side="right")] - cs[np.searchsorted(s_t, midnight, side="right")]
    hr = obs["wearable_heart_rate"].sort_values("timestamp")
    h_t, h_v = _ns(hr["timestamp"]), hr["heart_rate_bpm"].to_numpy(float)
    ssum, scnt = _window_sum(h_t, h_v, at, 30)
    f["wear_hr_30"] = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
    f["wear_hr_missing_2h"] = 1 - np.minimum(_window_sum(h_t, h_v, at, 120)[1] / 24.0, 1.0)
    st_p = obs["wearable_sleep_stages"]
    ep = _sleep_episodes(st_p, hr, obs["wearable_hrv"], fcfg)
    names = ["wear_sleep_total_min", "wear_sleep_deep_min", "wear_sleep_rem_min", "wear_sleep_wake_min",
             "wear_min_since_wake", "wear_sleep_hrv", "wear_sleep_hrv_rel", "wear_sleep_hr", "wear_hr_30_minus_sleep_hr"]
    for nm in names:
        f[nm] = np.full(n, np.nan)
    if len(ep):
        ends = _ns(ep["end"])
        k = np.searchsorted(ends, at, side="right") - 1
        ok = k >= 0
        kk = np.where(ok, k, 0)
        since = (at - ends[kk]) / 60e9
        ok &= since < 20 * 60          # last night's sleep only (not an older one)
        vals = {"wear_sleep_total_min": ep["total"], "wear_sleep_deep_min": ep["deep"], "wear_sleep_rem_min": ep["rem"],
                "wear_sleep_wake_min": ep["wake"], "wear_sleep_hrv": ep["hrv"], "wear_sleep_hr": ep["hr"]}
        for nm, col in vals.items():
            f[nm] = np.where(ok, col.to_numpy(float)[kk], np.nan)
        f["wear_min_since_wake"] = np.where(ok, since, np.nan)
        f["wear_sleep_hrv_rel"] = np.where(ok, (ep["hrv"] / ep["hrv_ref"]).to_numpy(float)[kk], np.nan)
        f["wear_hr_30_minus_sleep_hr"] = f["wear_hr_30"] - f["wear_sleep_hr"]
    st_t = np.sort(_ns(st_p["timestamp"]))
    f["wear_in_bed_now"] = (_window_sum(st_t, np.ones(len(st_t)), at, 10)[1] > 0).astype(float)

    # ---- meal log
    ml = obs["meal_log"].sort_values("timestamp")
    m_t, m_v = _ns(ml["timestamp"]), ml["carbs_g_est"].to_numpy(float)
    j = np.searchsorted(m_t, at, side="right") - 1
    has = j >= 0
    f["meal_min_since_last"] = np.where(has, np.minimum((at - m_t[np.maximum(j, 0)]) / 60e9, 720.0) if len(m_t) else np.nan, np.nan)
    f["meal_carbs_last"] = np.where(has, m_v[np.maximum(j, 0)] if len(m_v) else np.nan, np.nan)
    for w in fcfg["meal_windows_min"]:
        f[f"meal_carbs_{w}"] = _window_sum(m_t, m_v, at, w)[0]
    f["meal_n_logs_24h"] = _window_sum(m_t, m_v, at, 1440)[1]

    # ---- EHR (static + prescription schedule)
    df = pd.DataFrame(f)
    for col, v in static.items():
        df[col] = v
    meds = obs["medications"]
    for cls, name in (("sulfonylurea", "su"), ("biguanide", "met"), ("basal_insulin", "basal")):
        times = [t for x in meds.loc[meds["drug_class"] == cls, "times"] for t in x.split(";")]
        df[f"ehr_min_since_{name}_dose"] = _minutes_since_clock(idx, times) if times else np.nan

    # ---- targets
    for h in mcfg["horizons_min"]:
        k = h // step
        df[f"y_{h}"] = np.concatenate([g[k:], np.full(k, np.nan)])
    nwin = spk["window_min"] // step
    df["peak_120"] = _forward_window(g, nwin, "max", int(np.ceil(scfg["peak_min_coverage"] * nwin)))
    above = np.nan_to_num(g, nan=-1) > spk["threshold_mgdl"]
    need = spk["sustain_min"] // step + 1
    run = np.ones(n, dtype=bool)
    for q in range(need):
        run &= np.concatenate([above[q:], np.zeros(q, dtype=bool)])
    df["spike_label"] = _forward_window(run.astype(float), nwin, "max") > 0
    df["spike_eligible"] = g <= spk["threshold_mgdl"]
    prev_ok = np.concatenate([[False], ~np.isnan(g[:-1]) & (g[:-1] <= spk["threshold_mgdl"])])
    df["spike_onset_event"] = run & prev_ok
    df["g_now"] = g
    df["patient_id"] = pid
    df["timestamp"] = idx
    df["split"] = split

    # ---- prediction times
    stride = (scfg["eval_stride_min"] if split == "test" else scfg["train_stride_min"]) // step
    first = idx[0] + pd.Timedelta(minutes=scfg["min_history_min"])
    keep = (~np.isnan(g)) & (idx >= first) & (np.arange(n) % stride == 0)
    if split == "test":
        keep = (~np.isnan(g)) & (idx >= first)
    out = df[keep].reset_index(drop=True)
    num = out.select_dtypes("float64").columns
    out[num] = out[num].astype(np.float32)
    return out


def build(root=None, mcfg: dict | None = None, patient_ids: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    root = root or dataset_root()
    mcfg = mcfg or config.load("models")
    obs = load_observed(root)
    static = ehr_static(obs["patients"], obs["medications"])
    split = obs["splits"].set_index("patient_id")["split"]
    ids = patient_ids or list(obs["patients"]["patient_id"])
    per_patient = [k for k in obs if k not in ("patients", "splits")]
    grouped = {k: dict(tuple(obs[k].groupby("patient_id", sort=False))) for k in per_patient}
    empty = {k: obs[k].iloc[:0] for k in per_patient}
    tic = time.time()
    parts = []
    for i, pid in enumerate(ids):
        mine = {k: grouped[k].get(pid, empty[k]) for k in per_patient}
        parts.append(patient_features(pid, mine, static.loc[pid], split[pid], mcfg))
        if verbose and (i + 1) % 100 == 0:
            print(f"[features] {i + 1}/{len(ids)} patients ({time.time() - tic:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    from data_gen.generate import load_configs

    cfgs = load_configs()
    root = dataset_root(cfgs)
    df = build(root)
    out = config.REPO_ROOT / "data" / "features" / cfgs["pipeline"]["dataset_version"] / "features.parquet"
    write_parquet(df, out)
    print(f"[features] {len(df):,} rows x {df.shape[1]} columns -> {out}")
    print(df.groupby("split").agg(rows=("g_now", "size"), patients=("patient_id", "nunique"),
                                  spike_rate=("spike_label", "mean")).to_string())


if __name__ == "__main__":
    main()
