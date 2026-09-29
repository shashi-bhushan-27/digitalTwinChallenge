"""What-if simulator on the personalised ODE twin (never on the ML model).

Two modes:
* next hours: from the twin's current state, simulate a plan for the next few hours (the
  planned meal, walks, prescribed doses) with and without one intervention - swap or resize a
  meal, add a walk, skip a meal, change a dose.
* typical day: compare a typical day at steady state under the current vs a changed drug
  regimen (drug effects build up over days, so a one-off 5-hour window would understate them).

Every intervention is checked against the patient's constraints first (allergies, diet, eGFR
drug limits); blocked options are not simulated and carry their reasons. Every result carries a
hypoglycaemia check (trough below 70 mg/dL) with the patient's risk factors, and the label
"Simulation for discussion with the patient, not a prescription."
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.units import hba1c_from_eag_mgdl
from data_gen import physiology as ph
from twin import constraints as C
from twin.estimate import TwinState
from twin.model import Observed, Window, build_inputs, twin_physiology_config

DRUG_TIMES = {"biguanide": ["08:30", "20:30"], "sulfonylurea": ["08:00"], "dpp4_inhibitor": ["08:00"], "sglt2_inhibitor": ["08:00"],
              "basal_insulin": ["22:00"]}


@dataclass
class PlannedMeal:
    offset_min: int
    plate_id: str
    portion: float = 1.0


@dataclass
class Walk:
    offset_min: int
    duration_min: int = 20
    cadence_spm: float = 100.0


@dataclass
class Plan:
    meals: list[PlannedMeal] = field(default_factory=list)
    walks: list[Walk] = field(default_factory=list)
    dose_changes: dict[str, float] = field(default_factory=dict)   # drug -> new total daily dose (0 = stop)


@dataclass
class Intervention:
    kind: str                  # swap_meal | scale_meal | add_walk | skip_meal | change_dose
    meal_index: int = 0
    plate_id: str | None = None
    portion: float = 1.0
    factor: float = 1.0
    start_after_min: int = 20
    duration_min: int = 20
    cadence_spm: float = 100.0
    drug: str | None = None
    new_daily_dose: float | None = None

    def describe(self, library_by_id: dict) -> str:
        if self.kind == "swap_meal":
            return f"Swap the meal for {library_by_id[self.plate_id]['name']}" + (f" (x{self.portion:g} portion)" if self.portion != 1 else "")
        if self.kind == "scale_meal":
            return f"Change the portion to {self.factor:g}x"
        if self.kind == "add_walk":
            return f"Walk {self.duration_min} min, starting {self.start_after_min} min after the meal"
        if self.kind == "skip_meal":
            return "Skip the meal"
        if self.kind == "change_dose":
            return f"{self.drug.replace('_', ' ')}: {self.new_daily_dose:g} per day" if self.new_daily_dose else f"Stop {self.drug}"
        return self.kind


@dataclass
class Outcome:
    trajectory: pd.DataFrame          # timestamp, glucose_mgdl (plasma, 5-min)
    peak: float
    peak_time: pd.Timestamp
    minutes_above_180: float
    trough: float
    minutes_below_70: float
    mean: float


@dataclass
class WhatIfResult:
    description: str
    verdict: C.Verdict
    base: Outcome | None
    alt: Outcome | None
    safety: dict
    factors: list[str]
    label: str

    @property
    def delta_peak(self) -> float | None:
        return None if self.alt is None else self.alt.peak - self.base.peak


def patient_factors(p: pd.Series, meds: pd.DataFrame) -> list[str]:
    """The 'patient factors considered' chips shown above every what-if."""
    chips = [f"HbA1c {p['hba1c_pct']:.1f} %", f"eGFR {int(p['egfr'])}", f"BMI {p['bmi']:.1f}", p["diet_type"].replace("_", "-")]
    chips += [f"{a.replace('_', ' ')} allergy" for a in C._split(p["food_allergies"]) + C._split(p["drug_allergies"])]
    chips += [f"{t.replace('_', ' ')} intolerance" for t in C._split(p["intolerances"])]
    if p.get("fatty_liver"):
        chips.append("fatty liver")
    chips += sorted({m.replace("_", " ") for m in meds.loc[meds["drug_class"].isin(DRUG_TIMES), "drug"]})
    return chips


# ----------------------------------------------------------------------------------------------
# state and simulation
# ----------------------------------------------------------------------------------------------


def state_at(twin: TwinState, obs: Observed, t_now: pd.Timestamp, pcfg: dict, tcfg: dict) -> dict:
    """The twin's ODE state at t_now: replay the record so far, then nudge glucose to the CGM."""
    t0 = pd.Timestamp(obs.cgm["timestamp"].min()).floor("min")
    n = int((t_now - t0) / pd.Timedelta(minutes=1))
    win = Window(t0, max(n, 1))
    past = Observed(obs.patient, obs.meds, obs.cgm[obs.cgm["timestamp"] <= t_now], obs.meal_log[obs.meal_log["timestamp"] <= t_now],
                    obs.steps[obs.steps["timestamp"] <= t_now])
    meals = twin.meals[twin.meals["timestamp"] <= t_now]
    inp = build_inputs(twin.latent, meals, past, win, pcfg, carb_scale=twin.carb_scale)
    res = ph.simulate(twin.latent, inp, twin_physiology_config(pcfg), record_every=1440)
    st = {k: np.array(v, dtype=float, copy=True) for k, v in res.final_state.items()}
    c = past.cgm.dropna(subset=["glucose_mgdl"]).tail(4)
    if len(c) >= 2:
        slope = (c["glucose_mgdl"].iat[-1] - c["glucose_mgdl"].iat[0]) / max((c["timestamp"].iat[-1] - c["timestamp"].iat[0]).total_seconds() / 60, 1)
        st["G"] = np.array([max(40.0, c["glucose_mgdl"].iat[-1] + slope * tcfg["estimation"]["cgm_lag_min"])])
    return st


def _plate_meals(plan: Plan, t_now: pd.Timestamp, library_by_id: dict) -> pd.DataFrame:
    rows = []
    for m in plan.meals:
        p = library_by_id[m.plate_id]
        rows.append({"timestamp": t_now + pd.Timedelta(minutes=m.offset_min), "carbs_g": p["carbs"] * m.portion, "fat_g": p["fat"] * m.portion,
                     "protein_g": p["protein"] * m.portion, "fibre_g": p["fibre"] * m.portion, "gi": p["gi"], "source": "planned", "name": p["name"]})
    return pd.DataFrame(rows, columns=["timestamp", "carbs_g", "fat_g", "protein_g", "fibre_g", "gi", "source", "name"])


def _dose_plan(meds: pd.DataFrame, changes: dict, start: pd.Timestamp, n_min: int, scale: float) -> pd.DataFrame:
    regimen = {r.drug: (r.drug_class, r.daily_dose, r.times.split(";")) for r in meds.itertuples(index=False) if r.drug_class in DRUG_TIMES}
    for drug, dose in changes.items():
        cls = C.DRUG_CLASS[drug]
        times = regimen[drug][2] if drug in regimen else DRUG_TIMES[cls]
        if drug == "metformin" and dose <= 500:
            times = times[:1]
        regimen[drug] = (cls, dose, times)
    rows = []
    for day in pd.date_range(start.normalize() - pd.Timedelta(days=1), start + pd.Timedelta(minutes=n_min), freq="D"):
        for drug, (cls, dose, times) in regimen.items():
            if dose <= 0:
                continue
            for clock in times:
                h, mi = map(int, clock.split(":"))
                rows.append({"timestamp": day + pd.Timedelta(hours=h, minutes=mi), "drug": drug, "drug_class": cls, "daily_dose": dose,
                             "n_times": len(times), "scale": scale})
    return pd.DataFrame(rows, columns=["timestamp", "drug", "drug_class", "daily_dose", "n_times", "scale"])


def _outcome(g: np.ndarray, start: pd.Timestamp, tcfg: dict) -> Outcome:
    w = tcfg["whatif"]
    g5 = g[::5]
    ts = start + pd.to_timedelta(np.arange(len(g5)) * 5, unit="m")
    k = int(np.argmax(g5))
    return Outcome(pd.DataFrame({"timestamp": ts, "glucose_mgdl": g5}), float(g5.max()), ts[k], float((g > w["spike_threshold_mgdl"]).sum()),
                   float(g5.min()), float((g < w["hypo_threshold_mgdl"]).sum()), float(g.mean()))


def simulate_plan(twin: TwinState, obs: Observed, state: dict, t_now: pd.Timestamp, plan: Plan, pcfg: dict, tcfg: dict,
                  library_by_id: dict, hours: float | None = None) -> Outcome:
    n_min = int((hours or tcfg["whatif"]["horizon_h"]) * 60)
    win = Window(t_now, n_min)
    future = Observed(obs.patient, obs.meds, obs.cgm.iloc[:0], obs.meal_log.iloc[:0], obs.steps.iloc[:0])
    extra = np.zeros(n_min, np.float32)
    for w in plan.walks:
        a, b = max(w.offset_min, 0), min(w.offset_min + w.duration_min, n_min)
        extra[a:b] = w.cadence_spm
    doses = _dose_plan(obs.meds, plan.dose_changes, t_now, n_min, scale=1.0)
    # meals already eaten are still being absorbed: carry their remaining tail into the forecast
    k_len = pd.Timedelta(hours=pcfg["gut"]["kernel_hours"])
    eaten = twin.meals[(twin.meals["timestamp"] <= t_now) & (twin.meals["timestamp"] > t_now - k_len)]
    meals = pd.concat([eaten, _plate_meals(plan, t_now, library_by_id)], ignore_index=True)
    inp = build_inputs(twin.latent, meals, future, win, pcfg, carb_scale=twin.carb_scale, extra_steps=extra, dose_plan=doses)
    g = ph.simulate(twin.latent, inp, twin_physiology_config(pcfg), init=state, record_every=n_min).glucose[0]
    return _outcome(g, t_now, tcfg)


def apply(plan: Plan, iv: Intervention, patient: pd.Series, meds: pd.DataFrame, library_by_id: dict, ccfg: dict) -> tuple[Plan, C.Verdict]:
    new = copy.deepcopy(plan)
    if iv.kind == "swap_meal":
        v = C.check_plate(library_by_id[iv.plate_id], patient)
        new.meals[iv.meal_index] = PlannedMeal(new.meals[iv.meal_index].offset_min, iv.plate_id, iv.portion)
        return new, v
    if iv.kind == "scale_meal":
        new.meals[iv.meal_index].portion *= iv.factor
        return new, C.Verdict("allowed")
    if iv.kind == "skip_meal":
        del new.meals[iv.meal_index]
        return new, C.Verdict("allowed")
    if iv.kind == "add_walk":
        start = (new.meals[iv.meal_index].offset_min if new.meals else 0) + iv.start_after_min
        new.walks.append(Walk(start, iv.duration_min, iv.cadence_spm))
        v = C.Verdict("allowed")
        if patient.get("peripheral_neuropathy"):
            v = C.Verdict("warn", ["peripheral neuropathy recorded: check footwear and feet before advising walks"])
        return new, v
    if iv.kind == "change_dose":
        new.dose_changes[iv.drug] = float(iv.new_daily_dose)
        return new, C.check_drug_change(patient, iv.drug, float(iv.new_daily_dose), ccfg, meds)
    raise ValueError(iv.kind)


def safety(outcomes: list[Outcome | None], patient: pd.Series, meds: pd.DataFrame, tcfg: dict, ccfg: dict) -> dict:
    w = tcfg["whatif"]
    troughs = [o.trough for o in outcomes if o is not None]
    low = min(troughs) if troughs else float("nan")
    level = "hypo" if low < w["hypo_threshold_mgdl"] else "caution" if low < w["caution_threshold_mgdl"] else "ok"
    msg = {"hypo": f"Predicted low: trough {low:.0f} mg/dL (below {w['hypo_threshold_mgdl']})",
           "caution": f"Trough {low:.0f} mg/dL is close to the {w['hypo_threshold_mgdl']} mg/dL hypoglycaemia threshold",
           "ok": f"No predicted low (trough {low:.0f} mg/dL)"}[level]
    return {"level": level, "message": msg, "risk_factors": C.hypo_risk_factors(patient, meds, ccfg)}


def whatif(twin: TwinState, obs: Observed, t_now: pd.Timestamp, base_plan: Plan, iv: Intervention, pcfg: dict, tcfg: dict,
           library: list[dict], ccfg: dict | None = None, state: dict | None = None) -> WhatIfResult:
    ccfg = ccfg or tcfg["constraints"]
    lib = {p["id"]: p for p in library}
    state = state or state_at(twin, obs, t_now, pcfg, tcfg)
    new_plan, verdict = apply(base_plan, iv, obs.patient, obs.meds, lib, ccfg)
    base = simulate_plan(twin, obs, state, t_now, base_plan, pcfg, tcfg, lib)
    alt = simulate_plan(twin, obs, state, t_now, new_plan, pcfg, tcfg, lib) if verdict.allowed else None
    return WhatIfResult(iv.describe(lib), verdict, base, alt, safety([base, alt], obs.patient, obs.meds, tcfg, ccfg),
                        patient_factors(obs.patient, obs.meds), tcfg["whatif"]["label"])


# ----------------------------------------------------------------------------------------------
# typical day at steady state (drug regimen changes)
# ----------------------------------------------------------------------------------------------


def typical_meals(twin: TwinState, obs: Observed) -> list[tuple[str, float, float]]:
    """(clock, carbs, portion-free macros ratio) per usual meal slot from the patient's own log."""
    lg = twin.meals[twin.meals["source"] == "logged"].copy()
    defaults = {"breakfast": ("08:30", 50.0), "lunch": ("13:30", 75.0), "snack": ("17:30", 25.0), "dinner": ("21:00", 70.0)}
    out = []
    if len(lg):
        lg["slot"] = [("breakfast" if 5 <= t.hour < 11 else "lunch" if 11 <= t.hour < 16 else "snack" if 16 <= t.hour < 19 else "dinner")
                      for t in lg["timestamp"]]
        lg["mod"] = lg["timestamp"].dt.hour * 60 + lg["timestamp"].dt.minute
    for slot, (clock, carbs) in defaults.items():
        d = lg[lg["slot"] == slot] if len(lg) else lg
        if len(d) >= 2:
            mod = int(d["mod"].median())
            out.append((f"{mod // 60:02d}:{mod % 60:02d}", float(d["carbs_g"].median()), slot))
        elif slot != "snack":
            out.append((clock, carbs, slot))
    return out


def typical_day(twin: TwinState, obs: Observed, changes: dict, pcfg: dict, tcfg: dict, ccfg: dict | None = None) -> dict:
    """Day 3 of a steady-state simulation: current regimen vs `changes` (drug -> new daily dose)."""
    ccfg = ccfg or tcfg["constraints"]
    verdicts = {d: C.check_drug_change(obs.patient, d, dose, ccfg, obs.meds) for d, dose in changes.items()}
    days = tcfg["whatif"]["typical_day_days"]
    start = pd.Timestamp("2026-01-05")          # a notional Monday; only the clock times matter
    n_min = days * 1440
    dm = tcfg["estimation"]["default_meal_macros"]
    rows = []
    for d in range(days):
        for clock, carbs, _slot in typical_meals(twin, obs):
            h, mi = map(int, clock.split(":"))
            rows.append({"timestamp": start + pd.Timedelta(days=d, hours=h, minutes=mi), "carbs_g": carbs, "fat_g": carbs * dm["fat_per_g_carb"],
                         "protein_g": carbs * dm["protein_per_g_carb"], "fibre_g": carbs * dm["fibre_per_g_carb"], "gi": dm["gi"]})
    meals = pd.DataFrame(rows)
    none = Observed(obs.patient, obs.meds, obs.cgm.iloc[:0], obs.meal_log.iloc[:0], obs.steps.iloc[:0])
    adh = float(obs.patient["adherence_reported"])
    out = {}
    blocked = any(not v.allowed for v in verdicts.values())
    for name, ch in (("current", {}), ("changed", changes)):
        if name == "changed" and blocked:
            out[name] = None
            continue
        inp = build_inputs(twin.latent, meals, none, Window(start, n_min), pcfg, carb_scale=twin.carb_scale,
                           dose_plan=_dose_plan(obs.meds, ch, start, n_min, scale=adh))
        g = ph.simulate(twin.latent, inp, twin_physiology_config(pcfg), record_every=n_min).glucose[0][-1440:]
        o = _outcome(g, start + pd.Timedelta(days=days - 1), tcfg)
        out[name] = {"outcome": o, "tir": float(np.mean((g >= 70) & (g <= 180))), "est_hba1c": float(hba1c_from_eag_mgdl(g.mean()))}
    return {"verdicts": verdicts, **out, "safety": safety([out["current"]["outcome"], out["changed"]["outcome"] if out.get("changed") else None],
                                                            obs.patient, obs.meds, tcfg, ccfg),
            "factors": patient_factors(obs.patient, obs.meds), "label": tcfg["whatif"]["label"]}
