import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def cgm_pair(small_dataset):
    cgm = small_dataset["observed/cgm"]
    true = small_dataset["truth/glucose_true"]
    return cgm, true


def test_cgm_values_and_missingness(cgm_pair):
    cgm, _ = cgm_pair
    g = cgm["glucose_mgdl"]
    assert g.dropna().between(40, 400).all()
    missing = g.isna().mean()
    assert 0.005 < missing < 0.10
    assert set(cgm["flag"]) <= {"ok", "gap", "warmup", "low", "high"}
    step = cgm.groupby("patient_id")["timestamp"].diff().dt.total_seconds().dropna() / 60
    assert (step == 5).all()


def test_cgm_accuracy_in_device_range(cgm_pair):
    cgm, true = cgm_pair
    t = true.set_index(["patient_id", "timestamp"])["glucose_mgdl"]
    # true glucose is stored every 5 min at minute multiples of 5; CGM has a per-patient phase
    c = cgm.dropna(subset=["glucose_mgdl"]).copy()
    c = c[c["flag"] == "ok"]
    c["t5"] = c["timestamp"].dt.floor("5min")
    ref = t.reindex(pd.MultiIndex.from_arrays([c["patient_id"], c["t5"]])).to_numpy()
    ok = ~np.isnan(ref)
    mard = np.mean(np.abs(c["glucose_mgdl"].to_numpy()[ok] - ref[ok]) / ref[ok])
    assert 0.05 < mard < 0.15, mard


def test_cgm_lags_blood_glucose(cgm_pair):
    cgm, true = cgm_pair
    pid = cgm["patient_id"].iat[0]
    c = cgm[cgm["patient_id"] == pid].set_index("timestamp")["glucose_mgdl"].resample("5min").mean().interpolate()
    t = true[true["patient_id"] == pid].set_index("timestamp")["glucose_mgdl"].resample("5min").mean()
    c, t = c.align(t, join="inner")
    lags = range(0, 7)
    corr = [np.corrcoef(t.to_numpy()[: len(t) - k], c.to_numpy()[k:])[0, 1] for k in lags]
    best = lags[int(np.argmax(corr))] * 5
    assert 0 <= best <= 25


def test_wearable_streams(small_dataset):
    hr = small_dataset["observed/wearable_heart_rate"]
    steps = small_dataset["observed/wearable_steps"]
    stages = small_dataset["observed/wearable_sleep_stages"]
    assert hr["heart_rate_bpm"].between(35, 200).all()
    coverage = len(hr) / len(steps)
    assert 0.70 < coverage < 0.98
    assert (steps["steps"] >= 0).all()
    assert set(stages["stage"].unique()) <= {0, 1, 2, 3}
    assert (small_dataset["observed/wearable_hrv"]["rmssd_ms"] > 0).all()
