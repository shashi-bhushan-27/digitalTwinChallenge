import json

import numpy as np
import pandas as pd
import pytest

from common import config
from twin import constraints as C
from twin import estimate as E
from twin import whatif as W
from twin.model import Observed

LIB = config.load("meal_library")["plates"]
BY_ID = {p["id"]: p for p in LIB}
TCFG = config.load("twin")
CC = TCFG["constraints"]


def _patient(**kw):
    base = {"patient_id": "T", "age_yr": 55, "egfr": 85, "diet_type": "non_vegetarian", "food_allergies": "", "drug_allergies": "",
            "intolerances": "", "hba1c_pct": 8.2, "bmi": 27.0, "fatty_liver": False, "peripheral_neuropathy": False}
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------- constraints

def test_diet_and_allergy_blocks_have_reasons():
    v = C.check_plate(BY_ID["chicken_biryani"], _patient(diet_type="vegetarian"))
    assert v.status == "blocked" and "vegetarian" in v.reasons[0]
    assert C.check_plate(BY_ID["masala_dosa"], _patient(diet_type="jain")).status == "blocked"
    assert C.check_plate(BY_ID["khichdi_kadhi"], _patient(diet_type="jain")).allowed
    v = C.check_plate(BY_ID["poha"], _patient(food_allergies="peanut"))
    assert not v.allowed and "peanut" in v.reasons[0]
    assert not C.check_plate(BY_ID["curd_rice"], _patient(intolerances="lactose")).allowed
    assert C.check_plate(BY_ID["idli_sambar"], _patient(food_allergies="peanut")).allowed


def test_metformin_egfr_rules():
    assert C.check_drug_change(_patient(egfr=28), "metformin", 500, CC).status == "blocked"
    assert C.check_drug_change(_patient(egfr=40), "metformin", 1500, CC).status == "blocked"
    assert C.check_drug_change(_patient(egfr=40), "metformin", 1000, CC).status == "allowed"
    assert C.check_drug_change(_patient(egfr=80), "metformin", 2500, CC).status == "blocked"
    assert C.check_drug_change(_patient(egfr=80, intolerances="metformin_gi"), "metformin", 1000, CC).status == "warn"


def test_other_drug_rules():
    assert C.check_drug_change(_patient(egfr=20), "dapagliflozin", 10, CC).status == "blocked"
    assert C.check_drug_change(_patient(egfr=40), "dapagliflozin", 10, CC).status == "warn"
    assert C.check_drug_change(_patient(egfr=80), "dapagliflozin", 10, CC).status == "allowed"
    assert C.check_drug_change(_patient(age=70, age_yr=70), "glimepiride", 2, CC).status == "warn"
    assert C.check_drug_change(_patient(), "glimepiride", 10, CC).status == "blocked"
    assert C.check_drug_change(_patient(drug_allergies="sulfonamide_antibiotics"), "gliclazide_mr", 60, CC).status == "warn"
    assert C.check_drug_change(_patient(egfr=40), "sitagliptin", 100, CC).status == "blocked"
    assert C.check_drug_change(_patient(egfr=40), "sitagliptin", 50, CC).status == "allowed"


def test_hypo_risk_factors():
    meds = pd.DataFrame({"drug": ["glimepiride"], "drug_class": ["sulfonylurea"]})
    f = C.hypo_risk_factors(_patient(age_yr=72, egfr=40), meds, CC)
    assert any("age" in x for x in f) and any("eGFR" in x for x in f) and "sulfonylurea" in f


# ---------------------------------------------------------------- what-if on a twin

@pytest.fixture(scope="module")
def twin_setup(small_dataset, cfgs):
    tabs = {k.split("/", 1)[1]: v for k, v in small_dataset.items() if k.startswith("observed/")}
    meds = tabs["medications"]
    su_users = meds.loc[meds["drug_class"] == "sulfonylurea", "patient_id"]
    pid = su_users.iat[0] if len(su_users) else tabs["patients"]["patient_id"].iat[0]
    obs = Observed.from_tables(tabs, pid)
    twin = E.fit(obs, cfgs["physiology"], TCFG, LIB, fit_days=2)
    t_now = pd.Timestamp(obs.cgm["timestamp"].min()).normalize() + pd.Timedelta(days=2, hours=20)
    state = W.state_at(twin, obs, t_now, cfgs["physiology"], TCFG)
    return obs, twin, t_now, state, cfgs["physiology"]


def _run(setup, iv, plate="rice_sambar_meal"):
    obs, twin, t_now, state, pc = setup
    return W.whatif(twin, obs, t_now, W.Plan(meals=[W.PlannedMeal(15, plate)]), iv, pc, TCFG, LIB, state=state)


def test_fit_improves_on_the_prior(twin_setup):
    _, twin, *_ = twin_setup
    assert twin.fit["rmse_fit_mgdl"] <= twin.fit["rmse_prior_mgdl"] + 1e-6
    assert np.isfinite(twin.latent["egp_max"].iat[0])


def test_meal_and_walk_interventions_lower_the_peak(twin_setup):
    half = _run(twin_setup, W.Intervention("scale_meal", factor=0.5))
    skip = _run(twin_setup, W.Intervention("skip_meal"))
    walk = _run(twin_setup, W.Intervention("add_walk", start_after_min=20, duration_min=30))
    swap = _run(twin_setup, W.Intervention("swap_meal", plate_id="millet_bhakri_meal"))
    assert skip.delta_peak < half.delta_peak < 0
    assert walk.delta_peak < 0 and swap.delta_peak < 0
    for r in (half, skip, walk, swap):
        assert r.label.startswith("Simulation for discussion")
        assert any("HbA1c" in f for f in r.factors)


def test_blocked_option_is_not_simulated(twin_setup):
    obs = twin_setup[0]
    banned = "chicken_biryani" if obs.patient["diet_type"] != "non_vegetarian" else None
    if banned is None:
        obs2 = Observed(obs.patient.copy(), obs.meds, obs.cgm, obs.meal_log, obs.steps)
        obs2.patient["diet_type"] = "vegetarian"
        r = W.whatif(twin_setup[1], obs2, twin_setup[2], W.Plan(meals=[W.PlannedMeal(15, "rice_sambar_meal")]),
                     W.Intervention("swap_meal", plate_id="chicken_biryani"), twin_setup[4], TCFG, LIB, state=twin_setup[3])
    else:
        r = _run(twin_setup, W.Intervention("swap_meal", plate_id=banned))
    assert r.alt is None and r.verdict.status == "blocked" and r.verdict.reasons


def test_sulfonylurea_increase_with_skipped_meal_lowers_trough(twin_setup):
    obs, twin, t_now, state, pc = twin_setup
    su = obs.meds[obs.meds["drug_class"] == "sulfonylurea"]
    if su.empty:
        pytest.skip("no sulfonylurea user in the small dataset")
    drug = su["drug"].iat[0]
    lib = {p["id"]: p for p in LIB}
    morning = t_now.normalize() + pd.Timedelta(days=1, hours=7, minutes=30)
    st = W.state_at(twin, obs, morning, pc, TCFG)
    base = W.simulate_plan(twin, obs, st, morning, W.Plan(meals=[W.PlannedMeal(60, "idli_sambar")]), pc, TCFG, lib, hours=8)
    risky = W.simulate_plan(twin, obs, st, morning, W.Plan(meals=[], dose_changes={drug: TCFG["constraints"]["sulfonylurea"]["max_daily_mg"][drug]}),
                            pc, TCFG, lib, hours=8)
    assert risky.trough < base.trough
    s = W.safety([base, risky], obs.patient, obs.meds, TCFG, TCFG["constraints"])
    assert s["level"] in ("ok", "caution", "hypo") and "sulfonylurea" in s["risk_factors"]


def test_typical_day_compares_regimens(twin_setup):
    obs, twin, _, _, pc = twin_setup
    met = obs.meds[obs.meds["drug"] == "metformin"]
    dose = 2000.0 if len(met) and met["daily_dose"].iat[0] < 2000 else 1000.0
    out = W.typical_day(twin, obs, {"metformin": dose}, pc, TCFG)
    if out["verdicts"]["metformin"].allowed:
        assert out["changed"] is not None and np.isfinite(out["changed"]["outcome"].mean)
    assert out["current"]["tir"] >= 0


# ---------------------------------------------------------------- truth replay (full dataset only)

def test_truth_replay_is_exact():
    root = config.REPO_ROOT / "data" / "synthetic" / "v0.1"
    if not (root / "truth" / "glucose_true.parquet").exists():
        pytest.skip("full dataset not generated")
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if man["n_patients"] < 400:
        pytest.skip("replay test expects the full dataset")
    from twin.truth import regenerate

    tp = regenerate("SP-0007")
    g = tp.simulate()
    gt = pd.read_parquet(root / "truth" / "glucose_true.parquet", filters=[("patient_id", "==", "SP-0007")])
    m = ((gt["timestamp"] - tp.t_start).dt.total_seconds() // 60).astype(int).to_numpy()
    assert np.abs(g[m] - gt["glucose_mgdl"].to_numpy()).max() < 0.1
