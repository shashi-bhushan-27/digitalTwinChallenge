"""Twin state card: the personalised parameters in clinical words, with honest uncertainty.

Percentiles compare the patient's estimates with those of the twin-validation patients
(results/twin/twin_fits.csv). The meal-response line simulates a standard 60 g carbohydrate
meal from a fasting morning on the patient's twin, which is easier to read than raw parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config
from data_gen import physiology as ph
from twin.estimate import TwinState
from twin.model import Observed, Window, build_inputs, twin_physiology_config

REF = config.REPO_ROOT / "results" / "twin" / "twin_fits.csv"


def _band(pct: float) -> str:
    return "low" if pct < 25 else "below average" if pct < 45 else "average" if pct <= 55 else "above average" if pct <= 75 else "high"


def standard_meal_response(twin: TwinState, obs: Observed, pcfg: dict, carbs: float = 60.0) -> dict:
    """Peak rise and time to peak for a standard meal at 08:00 after a fasting night."""
    start = pd.Timestamp("2026-01-05 06:00")
    win = Window(start, 6 * 60)
    meal = pd.DataFrame([{"timestamp": start + pd.Timedelta(hours=2), "carbs_g": carbs, "fat_g": 10.0, "protein_g": 10.0, "fibre_g": 4.0, "gi": "high"}])
    none = Observed(obs.patient, obs.meds.iloc[:0], obs.cgm.iloc[:0], obs.meal_log.iloc[:0], obs.steps.iloc[:0])
    inp = build_inputs(twin.latent, meal, none, win, pcfg, carb_scale=twin.carb_scale, dose_plan=pd.DataFrame(columns=["timestamp", "drug", "drug_class", "daily_dose", "n_times"]))
    g = ph.simulate(twin.latent, inp, twin_physiology_config(pcfg), record_every=360).glucose[0]
    g0 = g[119]
    k = int(np.argmax(g[120:]))
    return {"baseline": float(g0), "rise": float(g[120 + k] - g0), "time_to_peak_min": k}


def twin_card(twin: TwinState, obs: Observed, pcfg: dict) -> dict:
    ref = pd.read_csv(REF) if REF.exists() else None

    def pct(col: str, value: float) -> float | None:
        if ref is None or col not in ref:
            return None
        return float(100 * np.mean(ref[col].to_numpy() <= value))

    si, be = float(twin.latent["si"].iat[0]), float(twin.latent["beta"].iat[0])
    rows = []
    for label, col, value, sd in (("Insulin sensitivity", "si_fit", si, twin.si_log_sd), ("Insulin secretion (beta-cell)", "beta_fit", be, twin.beta_log_sd)):
        p = pct(col, value)
        lo, hi = np.exp(-1.28 * sd), np.exp(1.28 * sd)
        rows.append({"label": label, "value": value, "percentile": p, "band": _band(p) if p is not None else "n/a",
                     "range_80": (value * lo, value * hi), "confidence": "good" if sd < 0.2 else "moderate" if sd < 0.35 else "low"})
    egp = float(twin.latent["egp_basal_mg_kg_min"].iat[0])
    resp = standard_meal_response(twin, obs, pcfg)
    return {
        "rows": rows,
        "fasting_set_point_mgdl": float(twin.latent["g_fasting_true"].iat[0]),
        "hepatic_output_mg_kg_min": egp,
        "standard_meal": resp,
        "fit_note": (f"Estimated from {twin.fit['fit_days']:.0f} days of CGM, {twin.fit['n_logged_meals']} logged meals"
                     f" ({twin.fit['n_imputed_meals']} unlogged rises filled in); CGM fit error {twin.fit['rmse_fit_mgdl']:.0f} mg/dL"),
    }
