import copy

import numpy as np
import pandas as pd
import pytest

from data_gen import physiology as ph
from data_gen.cohort import generate_cohort


@pytest.fixture(scope="module")
def setup(cohort_cfg, cfgs):
    pc = cfgs["physiology"]
    c = generate_cohort(cohort_cfg, 60, seed=5)
    lat = ph.derive_latent(c.patients, c.medications, c.truth, pc, seed=5)
    return c, lat, pc


def _flat_inputs(n, t, ra=None, exercise=None):
    one = np.ones((n, t), np.float32)
    return ph.SimInputs(ra=np.zeros((n, t)) if ra is None else ra,
                        exercise=np.zeros((n, t), np.float32) if exercise is None else exercise,
                        si_mult=one, egp_mult=one)


def _no_rhythm(pc):
    q = copy.deepcopy(pc)
    q["liver"]["dawn_amplitude"] = 0.0
    q["sensitivity_rhythm"]["evening_si_drop"] = 0.0
    return q


def _drug_free(lat, pc):
    lat = lat.copy()
    for col in ["rx_metformin_g_day", "rx_su_eq_day", "rx_dpp4_day", "rx_sglt2_day", "rx_glargine_u_day"]:
        lat[col] = 0.0
    ph._solve_basal(lat, pc, adjust_si=False)
    return lat


def test_latent_parameters_are_plausible(setup):
    _, lat, _ = setup
    assert lat["si"].between(4e-5, 1.2e-3).all()
    assert lat["beta"].gt(0).all()
    assert lat["i_basal"].between(2, 60).all()
    assert lat["egp_max"].gt(0).all()
    assert lat.select_dtypes("number").notna().all().all()


def test_fasting_steady_state_holds_without_inputs(setup):
    _, lat, pc = setup
    pc0 = _no_rhythm(pc)
    lat0 = _drug_free(lat, pc0)
    res = ph.simulate(lat0, _flat_inputs(len(lat0), 6 * 60), pc0)
    rel = res.glucose[:, -1] / lat0["g_fasting_true"].to_numpy()
    assert np.allclose(rel, 1.0, atol=0.01)


def test_meal_appearance_integrates_to_carbs(setup):
    _, lat, pc = setup
    meals = pd.DataFrame([{"patient_id": lat["patient_id"].iat[0], "minute": 60, "carbs_g": 60.0, "fat_g": 10.0,
                           "protein_g": 10.0, "fibre_g": 4.0, "gi": "high"}])
    q = copy.deepcopy(pc)
    q["gut"]["meal_carb_jitter_sd"] = 0.0
    q["gut"]["carb_effect_scale"] = 1.0
    ra = ph.meal_appearance(meals, lat.iloc[:1], q, 12 * 60, seed=1)
    expected = q["gut"]["bioavailability"] * 60.0 * 1000 / lat["vg_dl"].iat[0]
    assert ra.sum() == pytest.approx(expected, rel=1e-6)
    assert ra[0, :60].sum() == 0


def _meal_peak(lat, pc, carbs, exercise=None):
    n, t = len(lat), 6 * 60
    meals = pd.DataFrame([{"patient_id": pid, "minute": 30, "carbs_g": carbs, "fat_g": 10.0, "protein_g": 10.0,
                           "fibre_g": 4.0, "gi": "high"} for pid in lat["patient_id"]])
    ra = ph.meal_appearance(meals, lat, pc, t, seed=1)
    return ph.simulate(lat, _flat_inputs(n, t, ra=ra, exercise=exercise), pc).glucose


def test_more_carbs_gives_higher_peak(setup):
    _, lat, pc = setup
    pc0 = _no_rhythm(pc)
    small = _meal_peak(lat, pc0, 30).max(1)
    large = _meal_peak(lat, pc0, 90).max(1)
    assert (large > small).all()


def test_higher_insulin_sensitivity_lowers_peak(setup):
    _, lat, pc = setup
    pc0 = _no_rhythm(pc)
    lat0 = _drug_free(lat, pc0)
    base = _meal_peak(lat0, pc0, 60)
    rise_base = base.max(1) - lat0["g_fasting_true"].to_numpy()
    lat_hi = lat0.copy()
    lat_hi["si"] = lat_hi["si"] * 1.5
    ph._solve_basal(lat_hi, pc0, adjust_si=False)
    hi = _meal_peak(lat_hi, pc0, 60)
    rise_hi = hi.max(1) - lat_hi["g_fasting_true"].to_numpy()
    assert (rise_hi < rise_base).mean() > 0.95


def test_post_meal_walk_lowers_peak(setup):
    _, lat, pc = setup
    pc0 = _no_rhythm(pc)
    n, t = len(lat), 6 * 60
    walk = np.zeros((n, t), np.float32)
    walk[:, 50:80] = 110 * pc0["exercise"]["intensity_per_step_rate"]
    rest = _meal_peak(lat, pc0, 70).max(1)
    walked = _meal_peak(lat, pc0, 70, exercise=walk).max(1)
    assert (walked < rest).all()


def test_sulfonylurea_raises_hypoglycaemia_risk_when_fasting(setup):
    _, lat, pc = setup
    pc0 = _no_rhythm(pc)
    lat0 = _drug_free(lat, pc0)
    n, t = len(lat0), 10 * 60
    inputs = _flat_inputs(n, t)
    inputs.doses = {10: [(i, "su_gut", 1.0) for i in range(n)]}   # max-dose SU, no food
    with_su = ph.simulate(lat0, inputs, pc0).glucose.min(1)
    without = ph.simulate(lat0, _flat_inputs(n, t), pc0).glucose.min(1)
    assert (with_su < without - 5).mean() > 0.9


def test_states_stay_finite_and_bounded(small_dataset):
    g = small_dataset["truth/glucose_true"]["glucose_mgdl"]
    assert g.notna().all()
    assert g.between(20, 600).all()


def test_standard_day_matches_hba1c_anchor(setup):
    c, lat, pc = setup
    g = ph.simulate(lat, ph.standard_day_inputs(lat, c.medications, pc), pc).glucose[:, 1440:].mean(1)
    ratio = g / lat["target_mean_glucose"].to_numpy()
    assert np.mean(np.abs(ratio - 1) < 0.10) > 0.85
