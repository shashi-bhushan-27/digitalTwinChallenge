"""Physiology layer (twin core): hidden per-patient parameters and a glucose-insulin ODE.

Structure (all rates per minute, glucose in mg/dL, insulin in uU/mL):

    dG/dt  = EGP(t) + Ra(t) - (S_G + X + U_ex) * G - E_renal(G)
    dX/dt  = p2 * (SI_eff(t) * I - X)                              remote insulin action
    dI/dt  = -n * I + secretion(t) + exogenous insulin(t)
    dY/dt  = alpha * (beta * sigma_s * (G - h)+ * boosts - Y)      delayed static secretion
    dIh/dt = (I - Ih) / tau_h                                       insulin seen by the liver
    EGP    = EGP_max * dawn(t) * day(t) * (1 - E_metformin) / (1 + Ih / I_half) + counter-reg

Meals enter through Ra(t), a sum of gamma(shape 2) kernels per meal (Dalla Man-style gut).
Secretion adds a first-phase term on rising glucose and an incretin boost tied to Ra, which
DPP-4 inhibitors amplify (glucose-dependent, so little hypoglycaemia risk). Sulfonylureas add
glucose-independent secretion (hypoglycaemia risk, worse with low eGFR because clearance falls).
SGLT2 inhibitors add urinary glucose loss above a lowered threshold. Exercise adds
insulin-independent uptake now and higher insulin sensitivity for hours after.

Hidden parameters come from EHR fields through log-linear links plus random variation the EHR
cannot see (docs/decisions.md D-007). The hepatic output EGP_max is then solved so that the
patient's steady fasting glucose equals a noisy version of the recorded fasting glucose, which
ties the simulated curve to the record without copying it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import seeding

LN2 = float(np.log(2.0))
ACTIVITY_INDEX = {"sedentary": 0, "lightly_active": 1, "moderately_active": 2}


# ----------------------------------------------------------------------------------------------
# Hidden parameters
# ----------------------------------------------------------------------------------------------


def _regimen(medications: pd.DataFrame, patient_ids: pd.Series, cfg: dict) -> pd.DataFrame:
    """Daily drug exposure per patient in the normalised units the ODE uses."""
    su_eq = cfg["drugs"]["sulfonylurea"]["max_dose_equivalent_mg"]
    out = pd.DataFrame(index=pd.Index(patient_ids, name="patient_id"))
    out["rx_metformin_g_day"] = 0.0
    out["rx_su_eq_day"] = 0.0
    out["rx_dpp4_day"] = 0.0
    out["rx_sglt2_day"] = 0.0
    out["rx_glargine_u_day"] = 0.0
    for m in medications.itertuples(index=False):
        if m.drug_class == "biguanide":
            out.loc[m.patient_id, "rx_metformin_g_day"] += m.daily_dose / 1000.0
        elif m.drug_class == "sulfonylurea":
            out.loc[m.patient_id, "rx_su_eq_day"] += m.daily_dose / su_eq[m.drug]
        elif m.drug_class == "dpp4_inhibitor":
            out.loc[m.patient_id, "rx_dpp4_day"] += 1.0
        elif m.drug_class == "sglt2_inhibitor":
            out.loc[m.patient_id, "rx_sglt2_day"] += 1.15 if m.daily_dose >= 25 else 1.0
        elif m.drug_class == "basal_insulin":
            out.loc[m.patient_id, "rx_glargine_u_day"] += m.daily_dose
    return out.reset_index()


def dose_amount(drug: str, drug_class: str, dose: float, n_times: int, cfg: dict) -> tuple[str, float]:
    """(ODE compartment, amount) for one taken dose of a prescribed drug."""
    if drug_class == "biguanide":
        return "met", dose / 1000.0 / n_times
    if drug_class == "sulfonylurea":
        return "su_gut", dose / cfg["drugs"]["sulfonylurea"]["max_dose_equivalent_mg"][drug] / n_times
    if drug_class == "dpp4_inhibitor":
        return "dpp4", 1.0 / n_times
    if drug_class == "sglt2_inhibitor":
        return "sglt2", (1.15 if dose >= 25 else 1.0) / n_times
    if drug_class == "basal_insulin":
        return "gla", dose / n_times
    return "", 0.0


def _rates(cfg: dict) -> dict:
    d = cfg["drugs"]
    return {
        "k_met": LN2 / (d["metformin"]["effect_half_life_h"] * 60),
        "k_su_a": d["sulfonylurea"]["absorption_per_min"],
        "k_su_ref": LN2 / (d["sulfonylurea"]["half_life_h"] * 60),
        "k_dpp4": LN2 / (d["dpp4_inhibitor"]["half_life_h"] * 60),
        "k_sglt2": LN2 / (d["sglt2_inhibitor"]["half_life_h"] * 60),
        "k_gla": LN2 / (d["basal_insulin"]["absorption_half_life_h"] * 60),
    }


def derive_latent(
    patients: pd.DataFrame, medications: pd.DataFrame, truth: pd.DataFrame, cfg: dict, seed: int
) -> pd.DataFrame:
    """Hidden physiology per patient (one row each). Never shown to the models."""
    n = len(patients)
    lat = cfg["latent"]
    ref = lat["reference"]
    rates = _rates(cfg)
    r = seeding.rng(seed, "physiology", "latent")
    z = r.standard_normal((n, 12))

    bmi = patients["bmi"].to_numpy(float)
    waist = patients["waist_cm"].to_numpy(float)
    hba1c = patients["hba1c_pct"].to_numpy(float)
    age = patients["age_yr"].to_numpy(float)
    dur = patients["duration_yr"].to_numpy(float)
    egfr = patients["egfr"].to_numpy(float)
    fatty = patients["fatty_liver"].to_numpy(bool)
    neuro = patients["peripheral_neuropathy"].to_numpy(bool)
    act = patients["activity_level"].map(ACTIVITY_INDEX).to_numpy(float)
    bw = patients["weight_kg"].to_numpy(float)

    s = lat["si"]
    log_si = (
        np.log(s["ref"]) + s["bmi_per_unit"] * (bmi - ref["bmi"]) + s["waist_per_10cm"] * (waist - ref["waist"]) / 10
        + s["fatty_liver"] * fatty + s["hba1c_per_pct"] * (hba1c - ref["hba1c"])
        + s["age_per_10y"] * (age - ref["age"]) / 10 + s["activity_per_level"] * act + s["noise_sd"] * z[:, 0]
    )
    si = np.clip(np.exp(log_si), s["min"], s["max"])
    b = lat["beta"]
    beta = np.exp(
        np.log(b["ref"]) + b["duration_per_yr"] * (dur - ref["duration"]) + b["hba1c_per_pct"] * (hba1c - ref["hba1c"])
        + b["bmi_per_unit"] * (bmi - ref["bmi"]) + b["noise_sd"] * z[:, 1]
    )

    out = pd.DataFrame({"patient_id": patients["patient_id"].to_numpy()})
    out["bw_kg"] = bw
    out["vg_dl"] = cfg["volumes"]["vg_dl_per_kg"] * bw
    out["vi_ml"] = cfg["volumes"]["vi_ml_per_kg"] * bw
    # Target mean glucose: ADAG estimate from HbA1c times unexplained scatter (glycation gap, diet).
    from common.units import eag_mgdl_from_hba1c

    fs = lat["fasting"]
    r_anchor = seeding.rng(seed, "physiology", "anchor")
    target = eag_mgdl_from_hba1c(hba1c) * np.exp(r_anchor.normal(0, cfg["record_anchor"]["mean_glucose_noise_sd"], n))
    # True fasting set point: geometric blend of the (single, noisy) lab FPG and the fasting level
    # implied by HbA1c, plus noise, kept below the target mean.
    log_g = (fs["fpg_weight"] * np.log(patients["fpg_mgdl"].to_numpy(float))
             + (1 - fs["fpg_weight"]) * np.log(fs["fasting_to_mean_ratio"] * target)
             + lat["fasting_glucose_noise_sd"] * z[:, 2])
    out["g_fasting_true"] = np.clip(np.minimum(np.exp(log_g), fs["max_fraction_of_mean"] * target), 75, 330)
    out["target_mean_glucose"] = target
    out["si_prior"] = si
    out["si"] = si
    out["beta_prior"] = beta
    out["beta"] = beta
    out["s_g"] = cfg["glucose_effectiveness"]["s_g_ref"] * np.exp(cfg["glucose_effectiveness"]["s_g_sd"] * z[:, 3])
    lv = cfg["liver"]
    out["i_half"] = lv["i_half_ref"] * np.where(fatty, lv["fatty_liver_i_half_factor"], 1.0) * np.exp(lv["i_half_sd"] * z[:, 4])
    out["gastric_rate"] = np.exp(lat["gastric_rate_sd"] * z[:, 5]) * np.where(neuro, lat["neuropathy_gastric_factor"], 1.0)
    renal_frac = np.minimum(egfr, 90.0) / 90.0
    floor = lat["insulin_clearance_renal"]["floor"]
    out["n_i"] = cfg["insulin"]["n_per_min"] * (floor + (1 - floor) * renal_frac)
    su_floor = cfg["drugs"]["sulfonylurea"]["renal_clearance_floor"]
    out["k_su"] = rates["k_su_ref"] * (su_floor + (1 - su_floor) * renal_frac)
    out["gfr_ml_min"] = egfr
    k_cr = lv["counterreg_gain_per_min"] * (1 - lv["counterreg_age_decline_per_yr"] * (age - ref["age"]))
    out["k_cr"] = np.clip(k_cr, 0.3 * lv["counterreg_gain_per_min"], 1.3 * lv["counterreg_gain_per_min"])
    sd = lat["drug_response_sd"]
    out["resp_metformin"] = np.exp(sd * z[:, 6])
    out["resp_su"] = np.exp(sd * z[:, 7])
    out["resp_dpp4"] = np.exp(sd * z[:, 8])
    out["resp_sglt2"] = np.exp(sd * z[:, 9])
    out["resp_exercise"] = np.exp(lat["exercise_response_sd"] * z[:, 10])
    out["sleep_sensitivity"] = np.exp(lat["sleep_sensitivity_sd"] * z[:, 11])
    out["adherence_true"] = patients["patient_id"].map(truth.set_index("patient_id")["adherence_true"]).to_numpy()

    reg = _regimen(medications, patients["patient_id"], cfg)
    out = out.merge(reg, on="patient_id", how="left")
    _solve_basal(out, cfg)
    if cfg.get("record_anchor", {}).get("enabled", False):
        anchor_to_hba1c(out, patients, medications, cfg, seed)
    return out


def standard_day_inputs(latent: pd.DataFrame, medications: pd.DataFrame, cfg: dict, n_days: int = 2) -> SimInputs:
    """A fixed typical day (meals from config, prescribed doses scaled by true adherence)."""
    ra_cfg = cfg["record_anchor"]
    rows = []
    for pid in latent["patient_id"]:
        for d in range(n_days):
            for _, clock, carbs, fat, protein, fibre, gi in ra_cfg["standard_day"]:
                h, m = map(int, clock.split(":"))
                rows.append({"patient_id": pid, "minute": d * 1440 + h * 60 + m, "carbs_g": carbs, "fat_g": fat,
                             "protein_g": protein, "fibre_g": fibre, "gi": gi})
    no_jitter = {**cfg, "gut": {**cfg["gut"], "meal_rate_jitter_sd": 0.0, "meal_carb_jitter_sd": 0.0}}
    ra = meal_appearance(pd.DataFrame(rows), latent, no_jitter, n_days * 1440, seed=0)
    row_of = {pid: i for i, pid in enumerate(latent["patient_id"])}
    adh = latent.set_index("patient_id")["adherence_true"]
    events: dict[int, list] = {}
    for m in medications[medications["patient_id"].isin(row_of)].itertuples(index=False):
        times = m.times.split(";")
        comp, amt = dose_amount(m.drug, m.drug_class, m.daily_dose, len(times), cfg)
        if not comp:
            continue
        for d in range(n_days):
            for clock in times:
                h, mi = map(int, clock.split(":"))
                events.setdefault(d * 1440 + h * 60 + mi, []).append((row_of[m.patient_id], comp, amt * adh[m.patient_id]))
    n = len(latent)
    shape = (n, n_days * 1440)
    return SimInputs(ra=ra, exercise=np.zeros(shape, np.float32), si_mult=np.ones(shape, np.float32),
                     egp_mult=np.ones(shape, np.float32), doses=events)


def anchor_to_hba1c(latent: pd.DataFrame, patients: pd.DataFrame, medications: pd.DataFrame, cfg: dict, seed: int) -> None:
    """Scale beta-cell function per patient so the mean glucose of a standard day matches the
    target drawn from the recorded HbA1c (ADAG estimate times unexplained scatter).

    SI keeps its EHR-driven value (BMI, waist, liver, activity); HbA1c, duration and this anchor
    inform beta. Beta is bounded relative to its prior and so that basal insulin stays below a
    physiological ceiling; patients who hit a bound keep a glucose-HbA1c discordance, as real
    patients sometimes do. Bisection on log(multiplier); EGP_max is re-solved every step so the
    fasting anchor still holds."""
    ra_cfg = cfg["record_anchor"]
    target = latent["target_mean_glucose"].to_numpy(float)
    inputs = standard_day_inputs(latent, medications, cfg)
    base = latent["beta_prior"].to_numpy(float)
    per_beta, exo, _, _ = _basal_insulin_terms(latent, cfg)
    beta_floor = np.maximum(ra_cfg["min_basal_insulin"] - exo, 0.0) / per_beta
    beta_cap = np.maximum((ra_cfg["max_basal_insulin"] - exo) / per_beta, beta_floor * 1.01)
    lo = np.maximum(np.log(ra_cfg["beta_multiplier_bounds"][0]), np.log(np.maximum(beta_floor, 1e-9) / base))
    hi = np.maximum(np.minimum(np.log(ra_cfg["beta_multiplier_bounds"][1]), np.log(beta_cap / base)), lo)
    for _ in range(ra_cfg["iterations"]):
        mid = (lo + hi) / 2
        latent["beta"] = base * np.exp(mid)
        _solve_basal(latent, cfg, adjust_si=False)
        g = simulate(latent, inputs, cfg).glucose[:, 1440:].mean(1)
        too_high = g > target
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
    mult = np.exp((lo + hi) / 2)
    latent["beta"] = base * mult
    _solve_basal(latent, cfg, adjust_si=False)
    latent["anchor_beta_multiplier"] = mult


def _steady_drug_levels(p: pd.DataFrame | dict, cfg: dict) -> dict:
    """Mean compartment levels at steady state under the patient's true adherence."""
    rates = _rates(cfg)
    adh = np.asarray(p["adherence_true"], float)
    return {
        "met": adh * np.asarray(p["rx_metformin_g_day"], float) / (rates["k_met"] * 1440),
        "su_gut": adh * np.asarray(p["rx_su_eq_day"], float) / (rates["k_su_a"] * 1440),
        "su": adh * np.asarray(p["rx_su_eq_day"], float) / (np.asarray(p["k_su"], float) * 1440),
        "dpp4": adh * np.asarray(p["rx_dpp4_day"], float) / (rates["k_dpp4"] * 1440),
        "sglt2": adh * np.asarray(p["rx_sglt2_day"], float) / (rates["k_sglt2"] * 1440),
        "gla": adh * np.asarray(p["rx_glargine_u_day"], float) / (rates["k_gla"] * 1440),
    }


def drug_effects(levels: dict, p: dict, cfg: dict) -> dict:
    """Pharmacodynamic effects from compartment levels (vectorised)."""
    d = cfg["drugs"]
    rates = _rates(cfg)
    met = levels["met"]
    e_met = d["metformin"]["egp_emax"] * p["resp_metformin"] * met / (met + d["metformin"]["ec50_g"])
    e_su = levels["su"] * rates["k_su_ref"] * 1440 * p["resp_su"]
    e_dpp4 = np.minimum(levels["dpp4"] * rates["k_dpp4"] * 1440, 1.3) * p["resp_dpp4"]
    cn = levels["sglt2"] * rates["k_sglt2"] * 1440
    e_sglt2 = cn / (cn + d["sglt2_inhibitor"]["ec50"]) * p["resp_sglt2"]
    exo = rates["k_gla"] * levels["gla"] * d["basal_insulin"]["bioavailability"] * 1e6 / p["vi_ml"]
    return {"met": np.minimum(e_met, 0.6), "su": e_su, "dpp4": e_dpp4, "sglt2": np.minimum(e_sglt2, 1.2), "exo_insulin": exo}


def static_drive(g, ins: dict):
    """Glucose drive for static secretion: linear above threshold, saturating at high glucose."""
    above = np.maximum(g - ins["secretion_threshold_mgdl"], 0.0)
    return above / (1.0 + above / ins["secretion_saturation_mgdl"])


def renal_excretion(g, p: dict, e_sglt2, cfg: dict):
    theta = cfg["kidney"]["threshold_mgdl"]
    sg = cfg["drugs"]["sglt2_inhibitor"]
    lost = np.maximum(g - theta, 0.0) + e_sglt2 * sg["fraction"] * np.maximum(g - sg["threshold_mgdl"], 0.0)
    return p["gfr_ml_min"] * lost / (100.0 * p["vg_dl"])


def _basal_insulin_terms(out: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Basal insulin = per_beta * beta + exo, at the fasting set point with steady drug levels."""
    ins, d = cfg["insulin"], cfg["drugs"]
    p = {c: out[c].to_numpy(float) for c in out.columns if c != "patient_id" and out[c].dtype != object}
    eff = drug_effects(_steady_drug_levels(out, cfg), p, cfg)
    g = p["g_fasting_true"]
    per_beta = (ins["sigma_static"] * static_drive(g, ins) * (1 + d["sulfonylurea"]["static_boost"] * eff["su"])
                + ins["sigma_basal"] + d["sulfonylurea"]["glucose_independent"] * eff["su"]) / p["n_i"]
    return per_beta, eff["exo_insulin"] / p["n_i"], eff, p


def _solve_basal(out: pd.DataFrame, cfg: dict, adjust_si: bool = True) -> None:
    """Solve EGP_max so the fasting steady state equals g_fasting_true.

    With adjust_si, SI starts from its prior and is moved only if the implied basal hepatic
    output would be implausible; otherwise the current SI is kept."""
    per_beta, exo, eff, p = _basal_insulin_terms(out, cfg)
    g = p["g_fasting_true"]
    i_b = per_beta * p["beta"] + exo
    met_si = 1 + cfg["drugs"]["metformin"]["si_boost_at_emax"] * eff["met"]
    ren_b = renal_excretion(g, p, eff["sglt2"], cfg)
    vg_per_kg = cfg["volumes"]["vg_dl_per_kg"]
    if adjust_si:
        si = p["si_prior"].copy()
        egp_b = (p["s_g"] + si * met_si * i_b) * g + ren_b
        lo, hi = cfg["liver"]["egp_plausible_mg_per_kg_min"]
        egp_c = np.clip(egp_b, lo / vg_per_kg, hi / vg_per_kg)
        si_new = np.clip(((egp_c - ren_b) / g - p["s_g"]) / (met_si * i_b), cfg["latent"]["si"]["min"], cfg["latent"]["si"]["max"])
        adjusted = ~np.isclose(egp_c, egp_b, rtol=1e-9)
        si = np.where(adjusted, si_new, si)
        out["si"] = si
        out["si_adjusted_for_plausibility"] = adjusted
    si = out["si"].to_numpy(float)
    egp_b = (p["s_g"] + si * met_si * i_b) * g + ren_b  # exact anchor with the final SI
    out["i_basal"] = i_b
    out["egp_basal_mg_kg_min"] = egp_b * vg_per_kg
    out["egp_max"] = egp_b / ((1 - eff["met"]) / (1.0 + i_b / p["i_half"]))


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def _gauss_periodic(tod_h: np.ndarray, center: float, width: float) -> np.ndarray:
    dh = (tod_h - center + 12) % 24 - 12
    return np.exp(-0.5 * (dh / width) ** 2)


def meal_appearance(meals: pd.DataFrame, latent: pd.DataFrame, cfg: dict, n_min: int, seed: int) -> np.ndarray:
    """Ra(t) in mg/dL/min, shape (n_patients, n_min). `meals` has patient_id, minute, carbs_g,
    fat_g, protein_g, fibre_g, gi."""
    gut = cfg["gut"]
    k_len = int(gut["kernel_hours"] * 60)
    tau = np.arange(k_len, dtype=float)
    ra = np.zeros((len(latent), n_min), dtype=np.float64)
    row_of = {pid: i for i, pid in enumerate(latent["patient_id"])}
    for pid, grp in meals.groupby("patient_id", sort=False):
        i = row_of.get(pid)
        if i is None:
            continue
        r = seeding.rng(seed, "physiology", "meal_jitter", pid)
        gastric = latent["gastric_rate"].iat[i]
        vg = latent["vg_dl"].iat[i]
        for m in grp.itertuples(index=False):
            jit = np.exp(r.normal(0, gut["meal_rate_jitter_sd"])) * gut.get("absorption_speed_scale", 1.0)
            carbs = m.carbs_g * np.exp(r.normal(0, gut["meal_carb_jitter_sd"])) * gut.get("carb_effect_scale", 1.0)
            slow = 1 + gut["fat_slowing_per_g"] * m.fat_g + gut["protein_slowing_per_g"] * m.protein_g
            kf = gut["k_fast_per_min"] * gastric * jit / slow
            ks = gut["k_slow_per_min"] * gastric * jit / slow
            w = np.clip(gut["fast_fraction_by_gi"][m.gi] - gut["fibre_fast_fraction_drop_per_g"] * m.fibre_g, 0.15, 0.9)
            kern = w * kf * kf * tau * np.exp(-kf * tau) + (1 - w) * ks * ks * tau * np.exp(-ks * tau)
            kern /= kern.sum()
            start = int(m.minute)
            if start >= n_min or carbs <= 0:
                continue
            end = min(start + k_len, n_min)
            ra[i, start:end] += gut["bioavailability"] * carbs * 1000.0 / vg * kern[: end - start]
    return ra


def daily_modulation(daily: pd.DataFrame, latent: pd.DataFrame, cfg: dict, n_days: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute multipliers on SI and EGP from sleep, stress and illness (hidden day state).

    `daily` has patient_id, day (0-based, including warm-up), sleep_hours (night before),
    stress (z-score) and illness (bool)."""
    dd = cfg["day_to_day"]
    n = len(latent)
    n_min = n_days * 1440
    log_si = np.zeros((n, n_days))
    egp = np.ones((n, n_days))
    row_of = {pid: i for i, pid in enumerate(latent["patient_id"])}
    for pid, grp in daily.groupby("patient_id", sort=False):
        i = row_of[pid]
        r = seeding.rng(seed, "physiology", "day_noise", pid)
        eps = np.zeros(n_days)
        for d in range(n_days):
            prev = eps[d - 1] if d else 0.0
            eps[d] = dd["si_ar_rho"] * prev + np.sqrt(1 - dd["si_ar_rho"] ** 2) * r.normal(0, dd["si_ar_sd"])
        g = grp.sort_values("day")
        days = g["day"].to_numpy(int)
        deficit = np.maximum(dd["sleep_reference_h"] - g["sleep_hours"].to_numpy(float), 0.0)
        stress = g["stress"].to_numpy(float)
        ill = g["illness"].to_numpy(bool)
        log_si[i, days] = (
            -dd["sleep_deficit_si_per_h"] * latent["sleep_sensitivity"].iat[i] * deficit
            - dd["stress_si_per_sd"] * stress + np.log1p(-dd["illness_si_drop"]) * ill + eps[days]
        )
        egp[i, days] = np.clip(1 + dd["stress_egp_per_sd"] * stress + dd["illness_egp_rise"] * ill, 0.85, 1.3)
    # Day d applies from its boundary hour; smooth the steps over `transition_h`.
    shift = int(dd["day_boundary_h"] * 60)
    minute_day = np.clip((np.arange(n_min) - shift) // 1440, 0, n_days - 1)
    si_min = np.exp(log_si)[:, minute_day]
    egp_min = egp[:, minute_day]
    width = int(dd["transition_h"] * 60)
    kern = np.ones(width) / width
    for arr in (si_min, egp_min):
        for i in range(n):
            padded = np.concatenate([np.full(width, arr[i, 0]), arr[i], np.full(width, arr[i, -1])])
            arr[i] = np.convolve(padded, kern, mode="same")[width:-width]
    return si_min.astype(np.float32), egp_min.astype(np.float32)


@dataclass
class SimInputs:
    """Exogenous inputs on a 1-minute grid starting at `start_minute_of_day` of day 0."""

    ra: np.ndarray                       # (N, T) mg/dL/min
    exercise: np.ndarray                 # (N, T) intensity (steps/min * factor)
    si_mult: np.ndarray                  # (N, T) hidden daily SI multiplier
    egp_mult: np.ndarray                 # (N, T) hidden daily EGP multiplier
    doses: dict[int, list[tuple[int, str, float]]] = field(default_factory=dict)  # minute -> (row, compartment, amount)
    start_minute_of_day: int = 0

    @property
    def n_min(self) -> int:
        return self.ra.shape[1]


@dataclass
class SimResult:
    glucose: np.ndarray                  # (N, T) plasma glucose, mg/dL, 1-min
    insulin: np.ndarray                  # (N, T // record_every)
    final_state: dict


STATE_KEYS = ("G", "X", "I", "Y", "Ih", "dG", "A_s", "A_l", "met", "su_gut", "su", "dpp4", "sglt2", "gla")


def steady_state(latent: pd.DataFrame, cfg: dict) -> dict:
    p = {c: latent[c].to_numpy(float) for c in latent.columns if c not in ("patient_id",) and latent[c].dtype != object}
    lv = _steady_drug_levels(latent, cfg)
    eff = drug_effects(lv, p, cfg)
    met_si = 1 + cfg["drugs"]["metformin"]["si_boost_at_emax"] * eff["met"]
    ins = cfg["insulin"]
    g = p["g_fasting_true"]
    y_b = p["beta"] * ins["sigma_static"] * static_drive(g, ins) * (
        1 + cfg["drugs"]["sulfonylurea"]["static_boost"] * eff["su"]
    )
    n = len(latent)
    return {
        "G": g.copy(), "X": p["si"] * met_si * p["i_basal"], "I": p["i_basal"].copy(), "Y": y_b,
        "Ih": p["i_basal"].copy(), "dG": np.zeros(n), "A_s": np.zeros(n), "A_l": np.zeros(n),
        "met": lv["met"], "su_gut": lv["su_gut"], "su": lv["su"], "dpp4": lv["dpp4"], "sglt2": lv["sglt2"], "gla": lv["gla"],
    }


def simulate(latent: pd.DataFrame, inputs: SimInputs, cfg: dict, init: dict | None = None, record_every: int = 5) -> SimResult:
    """Explicit 1-minute Euler integration, vectorised over patients."""
    p = {c: latent[c].to_numpy(float) for c in latent.columns if c != "patient_id" and latent[c].dtype != object}
    st = {k: np.array(v, dtype=float, copy=True) for k, v in (init or steady_state(latent, cfg)).items()}
    ins, lv, dr, ex = cfg["insulin"], cfg["liver"], cfg["drugs"], cfg["exercise"]
    rates = _rates(cfg)
    n_min = inputs.n_min
    tod_h = ((inputs.start_minute_of_day + np.arange(n_min)) % 1440) / 60.0
    dawn = 1 + lv["dawn_amplitude"] * _gauss_periodic(tod_h, lv["dawn_center_h"], lv["dawn_width_h"])
    sr = cfg["sensitivity_rhythm"]
    si_rhythm = 1 - sr["evening_si_drop"] * _gauss_periodic(tod_h, sr["evening_center_h"], sr["evening_width_h"])

    beta, si, n_i = p["beta"], p["si"], p["n_i"]
    sig_s, sig_b, alpha, k_d = ins["sigma_static"], ins["sigma_basal"], ins["alpha_per_min"], ins["k_dynamic"]
    inc_gain, inc_half = ins["incretin_gain"], ins["incretin_ra_half"]
    p2, tau_h = ins["p2_per_min"], ins["tau_hepatic_min"]
    su_static, su_gi = dr["sulfonylurea"]["static_boost"], dr["sulfonylurea"]["glucose_independent"]
    dpp4_boost = dr["dpp4_inhibitor"]["incretin_boost"]
    met_si_boost = dr["metformin"]["si_boost_at_emax"]
    g_cr = lv["counterreg_threshold_mgdl"]
    tau_s, tau_l = ex["tau_acute_min"], ex["tau_long_min"]
    ex_up, ex_si = ex["uptake_per_intensity"], ex["si_boost_long"]

    out_g = np.empty((len(latent), n_min), dtype=np.float32)
    out_i = np.empty((len(latent), (n_min + record_every - 1) // record_every), dtype=np.float32)

    for t in range(n_min):
        for row, comp, amt in inputs.doses.get(t, ()):
            st[comp][row] += amt
        G, I, Y, Ih, X = st["G"], st["I"], st["Y"], st["Ih"], st["X"]
        eff = drug_effects(st, p, cfg)
        ra = inputs.ra[:, t]
        e = inputs.exercise[:, t]
        inc = inc_gain * (1 + dpp4_boost * eff["dpp4"]) * ra / (ra + inc_half)

        y_target = beta * sig_s * static_drive(G, ins) * (1 + su_static * eff["su"]) * (1 + inc)
        secretion = Y + beta * sig_b + beta * su_gi * eff["su"] + beta * k_d * np.maximum(st["dG"], 0.0) * (1 + inc)
        si_t = si * inputs.si_mult[:, t] * si_rhythm[t] * (1 + met_si_boost * eff["met"]) * (1 + ex_si * st["A_l"] * p["resp_exercise"])
        egp = (
            p["egp_max"] * dawn[t] * inputs.egp_mult[:, t] * (1 - eff["met"]) / (1 + Ih / p["i_half"])
            + p["k_cr"] * np.maximum(g_cr - G, 0.0)
        )
        uptake = (p["s_g"] + X + ex_up * st["A_s"] * p["resp_exercise"]) * G
        dG = egp + ra - uptake - renal_excretion(G, p, eff["sglt2"], cfg)

        st["G"] = np.maximum(G + dG, 10.0)
        st["dG"] = dG
        st["X"] = X + p2 * (si_t * I - X)
        st["I"] = np.maximum(I + (-n_i * I + secretion + eff["exo_insulin"]), 0.0)
        st["Y"] = Y + alpha * (y_target - Y)
        st["Ih"] = Ih + (I - Ih) / tau_h
        st["A_s"] = st["A_s"] + (e - st["A_s"]) / tau_s
        st["A_l"] = st["A_l"] + (e - st["A_l"]) / tau_l
        st["met"] = st["met"] * (1 - rates["k_met"])
        su_abs = rates["k_su_a"] * st["su_gut"]
        st["su_gut"] = st["su_gut"] - su_abs
        st["su"] = st["su"] + su_abs - p["k_su"] * st["su"]
        st["dpp4"] = st["dpp4"] * (1 - rates["k_dpp4"])
        st["sglt2"] = st["sglt2"] * (1 - rates["k_sglt2"])
        st["gla"] = st["gla"] * (1 - rates["k_gla"])

        out_g[:, t] = st["G"]
        if t % record_every == 0:
            out_i[:, t // record_every] = st["I"]
    return SimResult(glucose=out_g, insulin=out_i, final_state=st)
