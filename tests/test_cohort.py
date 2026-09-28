import numpy as np
import pandas as pd
import pytest

from common.io import content_hash
from common.units import egfr_ckd_epi_2021
from data_gen.cohort import GLUCOSE_LOWERING_CLASSES, correlation_matrix, generate_cohort

N = 400


@pytest.fixture(scope="module")
def cohort(cohort_cfg):
    return generate_cohort(cohort_cfg, N, seed=123)


def test_no_genetic_fields(cohort):
    banned = ("gene", "snp", "genotype", "allele", "polygenic", "variant")
    for table in (cohort.patients, cohort.medications, cohort.truth, cohort.hba1c_history):
        for col in table.columns:
            assert not any(b in col.lower() for b in banned), col


def test_value_ranges(cohort):
    p = cohort.patients
    assert p["age_yr"].between(28, 80).all()
    assert p["bmi"].between(17.5, 43).all()
    assert p["hba1c_pct"].between(5.8, 14.0).all()
    assert p["fpg_mgdl"].between(85, 380).all()
    assert p["egfr"].between(20, 125).all()
    assert (p["duration_yr"] <= p["age_yr"] - 22 + 1e-6).all()
    assert p["waist_cm"].between(65, 140).all()
    assert (p["sbp_mmhg"] > p["dbp_mmhg"]).mean() > 0.99
    assert p.isna().sum().sum() == 0


def test_creatinine_consistent_with_egfr(cohort):
    p = cohort.patients
    recomputed = egfr_ckd_epi_2021(p["creatinine_mgdl"], p["age_yr"], p["sex"] == "F")
    assert np.abs(recomputed - p["egfr"]).max() < 2.0   # creatinine rounded to 0.01 mg/dL


def test_correlation_structure(cohort, cohort_cfg):
    names = list(cohort_cfg["continuous"])
    r = correlation_matrix(names, cohort_cfg["correlations"]["pairs"])
    assert np.all(np.linalg.eigvalsh(r) > 0)
    rho = cohort.patients[["hba1c_pct", "fpg_mgdl", "bmi", "waist_cm", "age_yr", "egfr"]].corr(method="spearman")
    assert rho.loc["hba1c_pct", "fpg_mgdl"] > 0.6
    assert rho.loc["bmi", "waist_cm"] > 0.6
    assert rho.loc["age_yr", "egfr"] < -0.3


def test_prevalences_match_config(cohort, cohort_cfg):
    p = cohort.patients
    assert p["fatty_liver"].mean() == pytest.approx(cohort_cfg["fatty_liver"]["prevalence"], abs=0.01)
    assert p["activity_level"].value_counts(normalize=True)["sedentary"] == pytest.approx(0.45, abs=0.01)
    assert p["diet_type"].isin(["vegetarian", "non_vegetarian", "jain"]).all()


def test_medication_rules(cohort):
    p = cohort.patients.set_index("patient_id")
    m = cohort.medications.join(p[["egfr", "hba1c_pct", "duration_yr", "intolerances"]], on="patient_id")
    met = m[m["drug"] == "metformin"]
    assert (met["egfr"] >= 30).all(), "metformin below eGFR 30"
    assert (met.loc[met["egfr"] < 45, "daily_dose"] <= 1000).all(), "metformin dose not reduced at eGFR 30-45"
    assert not met["intolerances"].str.contains("metformin_gi").any()
    lowering = m[m["drug_class"].isin(GLUCOSE_LOWERING_CLASSES)].groupby("patient_id").size()
    needs = p[(p["hba1c_pct"] >= 7.0) | (p["duration_yr"] >= 2.0)].index
    assert set(needs) <= set(lowering.index)
    sita = m[(m["drug"] == "sitagliptin") & (m["egfr"] < 45)]
    assert (sita["daily_dose"] <= 50).all()


def test_hba1c_history_ends_at_current(cohort):
    h = cohort.hba1c_history
    cur = h[h["months_ago"] == 0].set_index("patient_id")["hba1c_pct"]
    assert (cur == cohort.patients.set_index("patient_id")["hba1c_pct"]).all()
    assert h.groupby("patient_id").size().eq(5).all()


def test_deterministic(cohort_cfg):
    a = generate_cohort(cohort_cfg, 50, seed=9)
    b = generate_cohort(cohort_cfg, 50, seed=9)
    c = generate_cohort(cohort_cfg, 50, seed=10)
    assert content_hash(a.patients) == content_hash(b.patients)
    assert content_hash(a.medications) == content_hash(b.medications)
    assert content_hash(a.patients) != content_hash(c.patients)
    pd.testing.assert_frame_equal(a.hba1c_history, b.hba1c_history)
