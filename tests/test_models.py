import numpy as np
import pandas as pd
import pytest

from common import config
from models import evaluate as E
from models import features as F
from models.train import conformal_offset


@pytest.fixture(scope="module")
def obs_one(small_dataset):
    """Observed tables of one patient from the shared small dataset, in the feature builder's format."""
    pid = small_dataset["observed/patients"]["patient_id"].iat[0]
    names = ["medications", "cgm", "wearable_heart_rate", "wearable_steps", "wearable_sleep_stages", "wearable_hrv", "meal_log"]
    obs = {n: small_dataset[f"observed/{n}"].query("patient_id == @pid").reset_index(drop=True) for n in names}
    static = F.ehr_static(small_dataset["observed/patients"], small_dataset["observed/medications"]).loc[pid]
    mcfg = config.load("models")
    mcfg["sampling"]["min_history_min"] = 120
    return pid, obs, static, mcfg


def test_features_use_only_the_past(obs_one):
    pid, obs, static, mcfg = obs_one
    base = F.patient_features(pid, obs, static, "test", mcfg)
    cut = base["timestamp"].iloc[len(base) // 2]
    changed = {k: v.copy() for k, v in obs.items()}
    fut = changed["cgm"]["timestamp"] > cut
    changed["cgm"].loc[fut, "glucose_mgdl"] = 399.0
    for name, col in (("wearable_steps", "steps"), ("wearable_heart_rate", "heart_rate_bpm"), ("meal_log", "carbs_g_est")):
        m = changed[name]["timestamp"] > cut
        changed[name].loc[m, col] = changed[name].loc[m, col] * 3 + 1
    alt = F.patient_features(pid, changed, static, "test", mcfg)
    feats = [c for c in base.columns if c.startswith(("cgm_", "wear_", "meal_", "ehr_", "tod_"))]
    a = base[base["timestamp"] <= cut].reset_index(drop=True)
    b = alt[alt["timestamp"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[feats], b[feats])
    # targets are allowed (and expected) to see the change
    assert not a["y_60"].equals(b["y_60"])


def test_spike_label_definition(obs_one):
    pid, obs, static, mcfg = obs_one
    o = {k: v.copy() for k, v in obs.items()}
    n = len(o["cgm"])
    g = np.full(n, 150.0)
    g[100:103] = 200.0           # only 3 readings (10 min) above 180: not sustained
    g[200:210] = 220.0           # sustained spike
    o["cgm"]["glucose_mgdl"] = g
    df = F.patient_features(pid, o, static, "test", mcfg).set_index("timestamp")
    ts = o["cgm"]["timestamp"]
    assert not df.loc[ts.iat[99], "spike_label"]
    assert df.loc[ts.iat[199], "spike_label"] and df.loc[ts.iat[200 - 24], "spike_label"]
    assert not df.loc[ts.iat[200 - 25], "spike_label"]
    assert df.loc[ts.iat[200], "spike_onset_event"] and not df.loc[ts.iat[201], "spike_onset_event"]
    assert not df.loc[ts.iat[205], "spike_eligible"]


def test_clarke_zones():
    z = E.clarke_zones(np.array([100, 50, 300, 100, 60]), np.array([105, 200, 100, 250, 65]))
    assert list(z) == ["A", "E", "D", "C", "A"]


def test_alert_events_counts():
    t = pd.date_range("2026-01-01", periods=60, freq="5min")
    d = pd.DataFrame({"patient_id": "P", "timestamp": t, "spike_eligible": True, "spike_onset_event": False, "p": 0.0})
    d.loc[30, "spike_onset_event"] = True
    d.loc[10:12, "p"] = 0.9      # alert 100-90 min before the onset: detected, lead 100 min
    d.loc[50, "p"] = 0.9         # alert with no onset afterwards: false
    r = E.alert_events(d, "p", 0.5, window_min=120, refractory_min=60)
    assert r["onsets"] == 1 and r["onsets_detected"] == 1
    assert r["median_lead_time_min"] == 100
    assert r["alert_episodes"] == 2 and r["false_alert_episodes"] == 1


def test_conformal_offset_hits_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    q = conformal_offset(np.full(5000, -0.5), np.full(5000, 0.5), y, 0.8)
    y2 = rng.normal(0, 1, 5000)
    cov = np.mean((y2 >= -0.5 - q) & (y2 <= 0.5 + q))
    assert abs(cov - 0.8) < 0.03
