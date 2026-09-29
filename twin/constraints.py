"""Constraint checks run before any what-if simulation.

Every check returns a Verdict: allowed / warn / blocked, with the reason in plain language. Meal
options are filtered by diet type (vegetarian, Jain), food allergies and intolerances; drug
changes by eGFR limits, maximum doses and allergy cautions. The rules are recalled from drug
labels and guidelines and are listed for clinical review (docs/clinical_rules_to_review.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_gen.behaviour import ALLERGY_TO_TAG, INTOLERANCE_TO_TAG

ALLERGEN_WORDS = {"peanut": "peanut", "tree_nut": "tree nuts", "dairy": "milk/dairy", "gluten": "gluten", "egg": "egg",
                  "fish": "fish", "shellfish": "shellfish"}
DRUG_CLASS = {"metformin": "biguanide", "glimepiride": "sulfonylurea", "gliclazide_mr": "sulfonylurea",
              "teneligliptin": "dpp4_inhibitor", "sitagliptin": "dpp4_inhibitor", "vildagliptin": "dpp4_inhibitor",
              "dapagliflozin": "sglt2_inhibitor", "empagliflozin": "sglt2_inhibitor", "insulin_glargine": "basal_insulin"}


@dataclass
class Verdict:
    status: str                       # "allowed" | "warn" | "blocked"
    reasons: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.status != "blocked"


def _split(s) -> list[str]:
    return [x for x in str(s or "").split(";") if x]


def check_plate(plate: dict, patient: pd.Series | dict) -> Verdict:
    """Can this patient eat this plate? Diet type, allergies and intolerances."""
    reasons = []
    diet = patient["diet_type"]
    if diet in ("vegetarian", "jain") and plate["diet"] != "veg":
        reasons.append(f"not vegetarian ({'contains egg' if plate['diet'] == 'egg' else 'contains meat or fish'}); patient is {diet.replace('_', '-')}")
    if diet == "jain" and plate["diet"] == "veg" and not plate["jain_ok"]:
        reasons.append("usually cooked with onion, garlic or root vegetables; patient follows a Jain diet")
    for a in _split(patient["food_allergies"]):
        tag = ALLERGY_TO_TAG.get(a)
        if tag and tag in plate["allergens"]:
            reasons.append(f"contains {ALLERGEN_WORDS[tag]}; patient has a {a.replace('_', ' ')} allergy")
    for t in _split(patient["intolerances"]):
        tag = INTOLERANCE_TO_TAG.get(t)
        if tag and tag in plate["allergens"]:
            reasons.append(f"contains {ALLERGEN_WORDS[tag]}; patient has {t} intolerance")
    return Verdict("blocked" if reasons else "allowed", reasons)


def meal_options(library: list[dict], patient, meal: str) -> list[tuple[dict, Verdict]]:
    """All plates for a meal slot, each with its verdict (blocked ones keep their reasons)."""
    slots = {"breakfast": ("breakfast",), "lunch": ("lunch",), "dinner": ("dinner", "lunch"), "snack": ("snack",)}[meal]
    return [(p, check_plate(p, patient)) for p in library if any(s in p["meal"] for s in slots)]


def check_drug_change(patient: pd.Series | dict, drug: str, new_daily_dose: float, ccfg: dict, current_meds: pd.DataFrame | None = None) -> Verdict:
    """Is a new total daily dose of `drug` acceptable for this patient? (0 = stop the drug)."""
    reasons, status = [], "allowed"
    egfr, age = float(patient["egfr"]), float(patient["age_yr"])
    cls = DRUG_CLASS.get(drug)
    allergies = _split(patient["drug_allergies"])
    intol = _split(patient["intolerances"])

    def block(msg):
        nonlocal status
        status = "blocked"
        reasons.append(msg)

    def warn(msg):
        nonlocal status
        if status != "blocked":
            status = "warn"
        reasons.append(msg)

    if new_daily_dose <= 0:
        return Verdict("warn", ["stopping a glucose-lowering drug usually raises glucose; simulate and review"])
    if cls is None:
        return Verdict("blocked", [f"{drug} is not in the twin's drug model"])
    if cls == "biguanide":
        m = ccfg["metformin"]
        if egfr < m["contraindicated_below_egfr"]:
            block(f"eGFR {egfr:.0f} is below {m['contraindicated_below_egfr']}: metformin is contraindicated")
        elif egfr < 45 and new_daily_dose > m["max_daily_mg_egfr_30_45"]:
            block(f"eGFR {egfr:.0f} (30-44): metformin limited to {m['max_daily_mg_egfr_30_45']} mg/day")
        if new_daily_dose > m["max_daily_mg"]:
            block(f"above the {m['max_daily_mg']} mg/day maximum used in this twin")
        if "metformin_gi" in intol:
            warn("patient has recorded metformin GI intolerance")
    elif cls == "sglt2_inhibitor":
        s = ccfg["sglt2_inhibitor"]
        if egfr < s["do_not_start_below_egfr"]:
            block(f"eGFR {egfr:.0f} is below {s['do_not_start_below_egfr']}: do not start an SGLT2 inhibitor")
        elif egfr < s["limited_glucose_effect_below_egfr"]:
            warn(f"eGFR {egfr:.0f} is below {s['limited_glucose_effect_below_egfr']}: little glucose-lowering effect expected")
    elif cls == "sulfonylurea":
        s = ccfg["sulfonylurea"]
        mx = s["max_daily_mg"].get(drug)
        if mx and new_daily_dose > mx:
            block(f"above the {mx} mg/day maximum for {drug}")
        if age >= s["caution_age_ge"]:
            warn(f"age {age:.0f}: higher risk of hypoglycaemia with sulfonylureas")
        if egfr < s["caution_egfr_lt"]:
            warn(f"eGFR {egfr:.0f}: sulfonylurea clearance is reduced, hypoglycaemia risk rises")
        if "sulfonamide_antibiotics" in allergies:
            warn("sulfonamide-antibiotic allergy recorded: cross-reactivity with sulfonylureas is debated, review before use")
    elif cls == "dpp4_inhibitor" and drug == "sitagliptin":
        r = ccfg["sitagliptin_renal_mg"]
        cap = r["egfr_below_30"] if egfr < 30 else r["egfr_below_45"] if egfr < 45 else None
        if cap is not None and new_daily_dose > cap:
            block(f"eGFR {egfr:.0f}: sitagliptin dose limited to {cap} mg/day")
    elif cls == "basal_insulin" and current_meds is not None:
        cur = current_meds.loc[current_meds["drug"] == drug, "daily_dose"]
        if len(cur) and cur.iat[0] > 0 and abs(new_daily_dose / cur.iat[0] - 1) > ccfg["basal_insulin"]["max_step_fraction"]:
            warn(f"change of more than {int(100 * ccfg['basal_insulin']['max_step_fraction'])} % in one step; usual titration is gradual")
    return Verdict(status, reasons)


def hypo_risk_factors(patient: pd.Series | dict, meds: pd.DataFrame, ccfg: dict) -> list[str]:
    """Clinical factors that raise hypoglycaemia risk (shown with any simulated low)."""
    h = ccfg["hypo_risk_factors"]
    out = []
    if float(patient["age_yr"]) >= h["age_ge"]:
        out.append(f"age {int(patient['age_yr'])}")
    if float(patient["egfr"]) < h["egfr_lt"]:
        out.append(f"eGFR {int(patient['egfr'])}")
    for cls in h["drug_classes"]:
        if (meds["drug_class"] == cls).any():
            out.append(cls.replace("_", " "))
    return out
