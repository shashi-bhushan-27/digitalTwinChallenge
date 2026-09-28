"""Unit conversions and standard glycaemic formulas.

Formula constants are from the cited papers; they are tagged "verify against source" in
docs/parameters_to_verify.md like every other literature value (BUILD_PROMPT rule 5).
"""

from __future__ import annotations

import numpy as np

MGDL_PER_MMOLL = 18.0182  # glucose molar mass 180.16 g/mol


def mgdl_to_mmoll(x):
    return np.asarray(x, dtype=float) / MGDL_PER_MMOLL


def mmoll_to_mgdl(x):
    return np.asarray(x, dtype=float) * MGDL_PER_MMOLL


def hba1c_pct_to_mmolmol(pct):
    """IFCC mmol/mol from NGSP %. Master equation: IFCC = (NGSP - 2.15) * 10.929."""
    return (np.asarray(pct, dtype=float) - 2.15) * 10.929


def hba1c_mmolmol_to_pct(mmolmol):
    return np.asarray(mmolmol, dtype=float) / 10.929 + 2.15


def eag_mgdl_from_hba1c(pct):
    """Estimated average glucose (ADAG study, Nathan et al. 2008): eAG = 28.7 * A1c - 46.7."""
    return 28.7 * np.asarray(pct, dtype=float) - 46.7


def hba1c_from_eag_mgdl(eag):
    return (np.asarray(eag, dtype=float) + 46.7) / 28.7


def gmi_pct(mean_glucose_mgdl):
    """Glucose management indicator (Bergenstal et al. 2018): GMI = 3.31 + 0.02392 * mean."""
    return 3.31 + 0.02392 * np.asarray(mean_glucose_mgdl, dtype=float)


def _ckd_epi_2021_a(age, female):
    age = np.asarray(age, dtype=float)
    female = np.asarray(female, dtype=bool)
    return 142.0 * 0.9938**age * np.where(female, 1.012, 1.0)


def egfr_ckd_epi_2021(creatinine_mgdl, age, female):
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al. 2021), mL/min/1.73 m2."""
    scr = np.asarray(creatinine_mgdl, dtype=float)
    female = np.asarray(female, dtype=bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    r = scr / kappa
    return _ckd_epi_2021_a(age, female) * np.minimum(r, 1.0) ** alpha * np.maximum(r, 1.0) ** -1.2


def creatinine_from_egfr(egfr, age, female):
    """Exact inverse of egfr_ckd_epi_2021 (the equation is piecewise power-law in Scr)."""
    egfr = np.asarray(egfr, dtype=float)
    female = np.asarray(female, dtype=bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    a = _ckd_epi_2021_a(age, female)
    ratio = egfr / a
    r_low = ratio ** (1.0 / alpha)  # branch Scr/kappa <= 1
    r_high = ratio ** (-1.0 / 1.2)  # branch Scr/kappa > 1
    r = np.where(ratio >= 1.0, r_low, r_high)
    return r * kappa
