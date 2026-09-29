"""The twin's model of one patient: record-based prior, inputs from observed data, simulation.

The twin uses the same ODE structure as the simulator (data_gen/physiology.py) but knows only
what a clinic would know: the EHR record, prescriptions and reported adherence, the CGM, the
wearable steps and the patient's own meal log. Hidden simulator values (true insulin
sensitivity, true meals, doses actually taken, day-to-day stress) are never read here.

Note on validity: the twin's ODE is the model that generated the synthetic data, so structural
model error is zero by construction. On real patients it would not be; the parameter-recovery
and counterfactual checks in twin/validate.py therefore validate against the simulated world only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_gen import physiology as ph

NOISE_KEYS = [("latent", "si", "noise_sd"), ("latent", "beta", "noise_sd"), ("latent", "fasting_glucose_noise_sd"),
              ("latent", "gastric_rate_sd"), ("latent", "drug_response_sd"), ("latent", "exercise_response_sd"),
              ("latent", "sleep_sensitivity_sd"), ("liver", "i_half_sd"), ("glucose_effectiveness", "s_g_sd"),
              ("record_anchor", "mean_glucose_noise_sd"), ("gut", "meal_rate_jitter_sd"), ("gut", "meal_carb_jitter_sd")]


def twin_physiology_config(pcfg: dict) -> dict:
    """Physiology config with every hidden random term switched off (population-mean links only)."""
    q = copy.deepcopy(pcfg)
    for path in NOISE_KEYS:
        node = q
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = 0.0
    return q


def prior_latent(patient: pd.Series, meds: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """One-row parameter table implied by the record alone (reported adherence, no hidden noise)."""
    q = twin_physiology_config(pcfg)
    pts = pd.DataFrame([patient.to_dict()])
    truth = pd.DataFrame({"patient_id": [patient["patient_id"]], "adherence_true": [float(patient["adherence_reported"])]})
    return ph.derive_latent(pts, meds, truth, q, seed=0)


@dataclass
class Observed:
    """One patient's observed streams (the twin's only view of the patient)."""

    patient: pd.Series
    meds: pd.DataFrame
    cgm: pd.DataFrame        # timestamp, glucose_mgdl
    meal_log: pd.DataFrame   # timestamp, meal_type, plate_name, carbs_g_est
    steps: pd.DataFrame      # timestamp, steps (per 5 min)

    @classmethod
    def from_tables(cls, tables: dict, pid: str) -> Observed:
        def one(name):
            d = tables[name]
            return d[d["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True) if "timestamp" in d else d[d["patient_id"] == pid].reset_index(drop=True)
        pts = tables["patients"]
        return cls(patient=pts[pts["patient_id"] == pid].iloc[0], meds=one("medications"), cgm=one("cgm"),
                   meal_log=one("meal_log"), steps=one("wearable_steps"))


def _slot(hour: float) -> str:
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 19:
        return "snack"
    if 19 <= hour < 24:
        return "dinner"
    return "night"


def logged_meals(obs: Observed, library: list[dict], tcfg: dict) -> pd.DataFrame:
    """Logged meals with macros from the dish the patient named, sized by their carb estimate
    corrected for typical under-reporting (logged = logging_bias x eaten)."""
    by_name = {p["name"]: p for p in library}
    dm = tcfg["estimation"]["default_meal_macros"]
    bias = float(tcfg["estimation"]["logging_bias"])
    rows = []
    for m in obs.meal_log.itertuples(index=False):
        c = float(m.carbs_g_est) / bias
        p = by_name.get(m.plate_name)
        if p and p["carbs"] > 0:
            f = c / p["carbs"]
            rows.append({"timestamp": m.timestamp, "carbs_g": c, "fat_g": p["fat"] * f, "protein_g": p["protein"] * f,
                         "fibre_g": p["fibre"] * f, "gi": p["gi"], "source": "logged", "name": m.plate_name})
        else:
            rows.append({"timestamp": m.timestamp, "carbs_g": c, "fat_g": c * dm["fat_per_g_carb"], "protein_g": c * dm["protein_per_g_carb"],
                         "fibre_g": c * dm["fibre_per_g_carb"], "gi": dm["gi"], "source": "logged", "name": m.plate_name})
    return pd.DataFrame(rows, columns=["timestamp", "carbs_g", "fat_g", "protein_g", "fibre_g", "gi", "source", "name"])


def imputed_meals(obs: Observed, logged: pd.DataFrame, tcfg: dict) -> pd.DataFrame:
    """Typical meals placed where CGM rises steeply with no logged meal (the twin's guess)."""
    u = tcfg["estimation"]["unlogged_meals"]
    dm = tcfg["estimation"]["default_meal_macros"]
    c = obs.cgm.dropna(subset=["glucose_mgdl"])
    if c.empty:
        return logged.iloc[:0]
    s = c.set_index("timestamp")["glucose_mgdl"]
    k = u["over_min"] // 5
    rise = s.shift(-k) - s
    cand = rise[rise >= u["rise_mgdl"]].index
    lt = logged["timestamp"].to_numpy().astype("datetime64[ns]")
    rows, last = [], None
    for t in cand:
        t64 = np.datetime64(t, "ns")
        if len(lt) and np.any((lt >= t64 - np.timedelta64(u["no_log_before_min"], "m")) & (lt <= t64 + np.timedelta64(15, "m"))):
            continue
        if last is not None and (t - last) < pd.Timedelta(minutes=u["merge_min"]):
            continue
        start = t - pd.Timedelta(minutes=10)
        carbs = float(u["default_carbs_by_slot"][_slot(start.hour + start.minute / 60)])
        rows.append({"timestamp": start, "carbs_g": carbs, "fat_g": carbs * dm["fat_per_g_carb"], "protein_g": carbs * dm["protein_per_g_carb"],
                     "fibre_g": carbs * dm["fibre_per_g_carb"], "gi": dm["gi"], "source": "imputed", "name": "unlogged meal (imputed)"})
        last = t
    return pd.DataFrame(rows, columns=logged.columns)


@dataclass
class Window:
    t0: pd.Timestamp          # minute 0 of the simulation
    n_min: int

    def minute(self, ts) -> np.ndarray:
        return ((pd.DatetimeIndex(pd.to_datetime(ts)).as_unit("ns").asi8 - self.t0.as_unit("ns").value) // 60_000_000_000).astype(int)


def build_inputs(lat: pd.DataFrame, meals: pd.DataFrame, obs: Observed, win: Window, pcfg: dict, carb_scale=1.0,
                 adherence: float | None = None, extra_steps: np.ndarray | None = None, dose_plan: pd.DataFrame | None = None) -> ph.SimInputs:
    """SimInputs for `lat` (N candidate rows of the same patient) over the window.

    meals: timestamp + macros (logged, imputed, or planned). Doses: prescriptions at their clock
    times scaled by `adherence` (reported adherence by default), or an explicit `dose_plan`
    (timestamp, drug, drug_class, daily_dose, n_times). Steps: the observed wearable steps."""
    q = twin_physiology_config(pcfg)
    n = len(lat)
    m = meals.copy()
    m["minute"] = win.minute(m["timestamp"]) if len(m) else []
    m = m[(m["minute"] >= 0) & (m["minute"] < win.n_min)]
    m["patient_id"] = lat["patient_id"].iat[0]
    base = ph.meal_appearance(m, lat.iloc[:1], q, win.n_min, seed=0)[0]
    scale = np.broadcast_to(np.asarray(carb_scale, float), (n,)).copy()

    ex = np.zeros(win.n_min, np.float32)
    st = obs.steps
    if len(st):
        mins = win.minute(st["timestamp"])
        ok = (mins >= 0) & (mins < win.n_min - 5)
        per_min = st["steps"].to_numpy(float)[ok] / 5.0
        for off in range(5):
            ex[mins[ok] + off] = per_min
    if extra_steps is not None:
        ex = np.maximum(ex, extra_steps[: win.n_min])
    ex *= pcfg["exercise"]["intensity_per_step_rate"]

    events: dict[int, list] = {}
    if dose_plan is None:
        adh = float(obs.patient["adherence_reported"]) if adherence is None else adherence
        rows = []
        days = pd.date_range(win.t0.normalize() - pd.Timedelta(days=1), win.t0 + pd.Timedelta(minutes=win.n_min), freq="D")
        for d in obs.meds.itertuples(index=False):
            times = d.times.split(";")
            for day in days:
                for clock in times:
                    h, mi = map(int, clock.split(":"))
                    rows.append({"timestamp": day + pd.Timedelta(hours=h, minutes=mi), "drug": d.drug, "drug_class": d.drug_class,
                                 "daily_dose": d.daily_dose, "n_times": len(times), "scale": adh})
        dose_plan = pd.DataFrame(rows, columns=["timestamp", "drug", "drug_class", "daily_dose", "n_times", "scale"])
    if len(dose_plan):
        mins = win.minute(dose_plan["timestamp"])
        for t, d in zip(mins, dose_plan.itertuples(index=False), strict=True):
            if 0 <= t < win.n_min:
                comp, amt = ph.dose_amount(d.drug, d.drug_class, d.daily_dose, d.n_times, pcfg)
                if comp:
                    amt *= getattr(d, "scale", 1.0)
                    events.setdefault(int(t), []).append((slice(None), comp, amt))   # same dose for every candidate row
    one = np.ones((1, win.n_min), np.float32)   # (1, T) rows broadcast over the N candidates
    return ph.SimInputs(ra=base[None, :], exercise=ex[None, :], si_mult=one, egp_mult=one,
                        doses=events, start_minute_of_day=int(win.t0.hour * 60 + win.t0.minute), ra_scale=scale)


def with_params(prior: pd.DataFrame, si_log: np.ndarray, beta_log: np.ndarray, gf_log: np.ndarray, pcfg: dict) -> pd.DataFrame:
    """Candidate parameter rows: multipliers on the prior's SI, beta and fasting set point, with
    hepatic output re-solved so each candidate's fasting steady state is exact."""
    n = len(si_log)
    lat = pd.concat([prior] * n, ignore_index=True)
    lat["si_prior"] = prior["si"].iat[0] * np.exp(si_log)
    lat["si"] = lat["si_prior"]
    lat["beta"] = prior["beta"].iat[0] * np.exp(beta_log)
    lat["g_fasting_true"] = prior["g_fasting_true"].iat[0] * np.exp(gf_log)
    ph._solve_basal(lat, twin_physiology_config(pcfg), adjust_si=False)
    return lat


def lagged(g: np.ndarray, tau_min: float) -> np.ndarray:
    """Plasma glucose passed through the CGM's interstitial lag (rows = candidates)."""
    from scipy.signal import lfilter

    a = 1.0 / tau_min
    zi = (g[:, :1] * (1 - a))
    y, _ = lfilter([a], [1.0, -(1.0 - a)], g, axis=1, zi=zi)
    return y
