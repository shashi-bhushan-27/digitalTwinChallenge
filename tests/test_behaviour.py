import numpy as np
import pandas as pd
import pytest

from common import config
from data_gen.behaviour import STAGE_OUT, eligible_plates, generate_patient


@pytest.fixture(scope="module")
def library():
    return config.load("meal_library")["plates"]


def _patient(**kw):
    base = {"patient_id": "T-1", "sex": "F", "age_yr": 55, "bmi": 27.0, "duration_yr": 8.0, "diet_type": "vegetarian",
            "food_allergies": "", "intolerances": "", "region": "south", "usual_dinner_time": "20:30",
            "activity_level": "sedentary", "typical_daily_steps": 3800, "morning_walk_habit": False,
            "post_meal_walk_habit": True, "skips_breakfast": False, "peripheral_neuropathy": False}
    base.update(kw)
    return pd.Series(base)


def test_diet_and_allergy_filters(library):
    veg = eligible_plates(library, _patient(diet_type="vegetarian"))
    assert all(p["diet"] == "veg" for p in veg)
    jain = eligible_plates(library, _patient(diet_type="jain"))
    assert all(p["diet"] == "veg" and p["jain_ok"] for p in jain)
    nut = eligible_plates(library, _patient(diet_type="non_vegetarian", food_allergies="peanut;shellfish"))
    assert not any({"peanut", "shellfish"} & set(p["allergens"]) for p in nut)
    lact = eligible_plates(library, _patient(intolerances="lactose"))
    assert not any("dairy" in p["allergens"] for p in lact)


def test_generated_meals_respect_constraints(small_dataset, library):
    by_name = {p["name"]: p for p in library}
    meals = small_dataset["truth/meals_true"].merge(small_dataset["observed/patients"], on="patient_id")
    for m in meals.itertuples():
        plate = by_name[m.plate_name]
        if m.diet_type != "non_vegetarian":
            assert plate["diet"] == "veg"
        if m.diet_type == "jain":
            assert plate["jain_ok"]


def test_single_patient_week(cfgs, library):
    out = generate_patient(_patient(), pd.DataFrame(columns=["drug", "drug_class", "daily_dose", "dose_unit", "times"]),
                           0.8, library, set(), cfgs["behaviour"], 7, pd.Timestamp("2026-03-01"), seed=3)
    per_day = out["meals"].groupby("day").size()
    assert per_day.between(2, 7).all()
    assert out["daily"]["sleep_hours"].between(3, 11).all()
    steps = out["steps"]
    assert steps.min() >= 0
    assert (steps[out["sleep_stage"] != STAGE_OUT] == 0).all(), "steps recorded while in bed"
    assert np.all((out["heart_rate"] >= 38) & (out["heart_rate"] <= 190))
    assert (out["hrv"]["rmssd_ms"] > 0).all()
    assert set(np.unique(out["sleep_stage"])) <= {-1, 0, 1, 2, 3}


def test_hidden_state_is_not_a_function_of_ehr(cfgs, library):
    """Two patients with identical records must still behave differently (hidden variation)."""
    empty = pd.DataFrame(columns=["drug", "drug_class", "daily_dose", "dose_unit", "times"])
    a = generate_patient(_patient(patient_id="T-1"), empty, 0.8, library, set(), cfgs["behaviour"], 5, pd.Timestamp("2026-03-01"), 3)
    b = generate_patient(_patient(patient_id="T-2"), empty, 0.8, library, set(), cfgs["behaviour"], 5, pd.Timestamp("2026-03-01"), 3)
    assert not np.array_equal(a["steps"], b["steps"])
    assert not a["daily"]["sleep_hours"].equals(b["daily"]["sleep_hours"])
