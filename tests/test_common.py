import numpy as np
import pytest

from common import config, seeding
from common.units import (
    creatinine_from_egfr,
    eag_mgdl_from_hba1c,
    egfr_ckd_epi_2021,
    gmi_pct,
    hba1c_mmolmol_to_pct,
    hba1c_pct_to_mmolmol,
)

CONFIG_FILES = ["pipeline", "cohort", "physiology", "behaviour", "sensors", "meal_library", "validation"]


def test_rng_is_reproducible_and_key_dependent():
    a = seeding.rng(1, "layer", "p1").random(5)
    b = seeding.rng(1, "layer", "p1").random(5)
    c = seeding.rng(1, "layer", "p2").random(5)
    d = seeding.rng(2, "layer", "p1").random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_ckd_epi_inverse_roundtrip():
    rng = np.random.default_rng(0)
    egfr = rng.uniform(15, 130, 500)
    age = rng.uniform(25, 85, 500)
    female = rng.random(500) < 0.5
    scr = creatinine_from_egfr(egfr, age, female)
    assert np.allclose(egfr_ckd_epi_2021(scr, age, female), egfr, rtol=1e-9)
    assert np.all((scr > 0.1) & (scr < 8))


def test_glycaemic_formulas():
    assert eag_mgdl_from_hba1c(7.0) == pytest.approx(154.2, abs=0.1)   # ADAG
    assert gmi_pct(154.0) == pytest.approx(6.99, abs=0.01)             # GMI
    assert hba1c_mmolmol_to_pct(hba1c_pct_to_mmolmol(8.4)) == pytest.approx(8.4)
    assert hba1c_pct_to_mmolmol(6.5) == pytest.approx(47.5, abs=0.1)


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_provenance_records_are_complete(name):
    for rec in config.collect_provenance(config.load_raw(name)):
        assert rec.source.strip(), f"{name}:{rec.path} has an empty source"
        assert isinstance(rec.verified, bool)


def test_resolve_replaces_value_records():
    raw = {"a": {"value": 3, "source": "x", "verified": False}, "b": {"c": 1, "source": "y", "verified": False}}
    assert config.resolve(raw) == {"a": 3, "b": {"c": 1}}
