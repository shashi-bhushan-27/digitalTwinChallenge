import numpy as np
import pandas as pd

from validation import tstr
from validation.compare import compare_table, match_by_mean
from validation.windows import regular_series, window_features, windows


def _cgm(pid="P", days=2, step=15, level=150.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=days * 1440 // step, freq=f"{step}min")
    t = np.arange(len(ts))
    g = level + 60 * np.sin(2 * np.pi * t / (1440 / step / 3)) + rng.normal(0, 3, len(ts))
    return pd.DataFrame({"patient_id": pid, "timestamp": ts, "glucose_mgdl": g})


def test_windows_align_to_clock_and_skip_gaps():
    df = _cgm(days=3)
    df.loc[100:130, "glucose_mgdl"] = np.nan
    x, pids, starts = windows(regular_series(df), 48, 48, align_hours=(0, 12))
    assert all(s.hour in (0, 12) and s.minute == 0 for s in starts)
    assert len(x) >= 3 and not np.isnan(x).any()
    f = window_features(x)
    assert len(f) == len(x) and f.notna().all().all()


def test_tstr_samples_and_spike_label():
    df = _cgm(days=3, level=170.0)
    s = tstr.samples(df)
    assert {"y30", "y60", "spike", "eligible_spike"} <= set(s.columns)
    assert not s["spike"][~s["eligible_spike"]].any(), "spike onset only from at or below threshold"
    assert s["spike"].any()


def test_match_by_mean_is_one_to_one():
    real = pd.DataFrame({"mean": [120.0, 150.0, 180.0]}, index=["r1", "r2", "r3"])
    syn = pd.DataFrame({"mean": [119.0, 121.0, 151.0, 179.0, 200.0]}, index=list("abcde"))
    m = match_by_mean(real, syn)
    assert list(m.index) == ["a", "c", "d"]
    assert m.index.is_unique
    t = compare_table(pd.DataFrame({"x": [1.0, 2, 3, 4]}), pd.DataFrame({"x": [1.0, 2, 3, 4]}), ["x"])
    assert abs(t["median_gap_in_real_iqr"].iat[0]) < 1e-9
