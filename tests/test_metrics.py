import numpy as np
import pandas as pd
import pytest

from validation import metrics as M


def _trace(values, start="2026-01-01", step=15, pid="P"):
    ts = pd.date_range(start, periods=len(values), freq=f"{step}min")
    return pd.DataFrame({"patient_id": pid, "timestamp": ts, "glucose_mgdl": values})


def test_summary_time_in_ranges():
    g = np.array([50, 60, 100, 150, 200, 260, 300, 120])
    s = M.summary(g)
    assert s["tbr_70"] == pytest.approx(2 / 8)
    assert s["tbr_54"] == pytest.approx(1 / 8)
    assert s["tir_70_180"] == pytest.approx(3 / 8)
    assert s["tar_180"] == pytest.approx(3 / 8)
    assert s["tar_250"] == pytest.approx(2 / 8)


def test_mage_on_regular_swings():
    t = np.arange(0, 96 * 3)
    g = 150 + 60 * np.sin(2 * np.pi * t / 32)       # peak-to-nadir 120
    assert M.mage(g) == pytest.approx(120, rel=0.05)
    assert np.isnan(M.mage(np.full(5, 100.0)))


def test_mage_ignores_small_wiggles():
    t = np.arange(0, 96 * 2)
    g = 150 + 60 * np.sin(2 * np.pi * t / 48) + 3 * np.sin(2 * np.pi * t / 3)
    assert M.mage(g) == pytest.approx(120, rel=0.1)


def test_postprandial_detects_rise():
    base = np.full(40, 110.0)
    base[10:22] = 110 + np.array([10, 25, 45, 60, 70, 65, 55, 40, 30, 20, 10, 5])
    cgm = _trace(base)
    meals = pd.DataFrame({"patient_id": ["P"], "timestamp": [cgm["timestamp"].iat[9]]})
    pp = M.postprandial(cgm, meals)
    assert len(pp) == 1
    assert pp["rise"].iat[0] == pytest.approx(70)
    assert pp["time_to_peak_min"].iat[0] == pytest.approx(75)


def test_spike_onsets_need_sustained_crossing():
    g = np.array([150, 190, 150, 150, 190, 195, 200, 150, 150] * 4, dtype=float)
    s = M.spike_onsets_per_day(_trace(g), threshold=180, sustain_min=15)
    days = (len(g) - 1) * 15 / 1440
    assert s.iat[0] == pytest.approx(4 / days)


def test_resample_keeps_first_reading_per_bin():
    df = _trace(np.arange(12, dtype=float), step=5)
    out = M.resample(df, 15)
    assert list(out["glucose_mgdl"]) == [0, 3, 6, 9]
