"""Load the real open datasets into one common format (validation and calibration only).

Common format:
    patients: dataset, patient_id, age_yr, sex, bmi, hba1c_pct, fpg_mgdl, duration_yr,
              egfr, on_insulin, glucose_lowering_agents
    cgm:      dataset, patient_id, recording_id, timestamp, glucose_mgdl
    meals:    dataset, patient_id, recording_id, timestamp, carbs_g (NaN when unknown), description

Shanghai T2DM: FreeStyle Libre-type CGM every 15 min, meals as free text on CGM rows.
CGMacros: minute-level files where Libre readings (15 min) were linearly interpolated; only the
original 15-minute points are kept. T2D participants are those with lab HbA1c >= 6.5 %.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT
from common.units import hba1c_mmolmol_to_pct

REAL = REPO_ROOT / "data" / "real"
INSULIN_WORDS = ("insulin", "novolin", "humulin", "gansulin", "glargine", "degludec", "detemir", "aspart", "lispro", "70/30", "30r", "40r")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python -m validation.real_data.download")
    return path


@lru_cache(maxsize=1)
def load_shanghai() -> dict[str, pd.DataFrame]:
    root = _require(REAL / "shanghai")
    s = pd.read_excel(root / "Shanghai_T2DM_Summary.xlsx")
    s.columns = [c.strip() for c in s.columns]
    agents = s["Hypoglycemic Agents"].fillna("none").astype(str).str.strip()
    rec = pd.DataFrame({
        "dataset": "shanghai_t2dm",
        "recording_id": s["Patient Number"].astype(str),
        "patient_id": "SH-" + s["Patient Number"].astype(str).str[:4],
        "age_yr": pd.to_numeric(s["Age (years)"], errors="coerce"),
        "sex": s["Gender (Female=1, Male=2)"].map({1: "F", 2: "M"}),
        "bmi": pd.to_numeric(s["BMI (kg/m2)"], errors="coerce"),
        "hba1c_pct": hba1c_mmolmol_to_pct(pd.to_numeric(s["HbA1c (mmol/mol)"], errors="coerce")),
        "fpg_mgdl": pd.to_numeric(s["Fasting Plasma Glucose (mg/dl)"], errors="coerce"),
        "duration_yr": pd.to_numeric(s["Duration of diabetes (years)"], errors="coerce"),
        "egfr": pd.to_numeric(s["Estimated Glomerular Filtration Rate  (ml/min/1.73m2)"], errors="coerce"),
        "glucose_lowering_agents": agents,
        "on_insulin": agents.str.lower().apply(lambda a: any(w in a for w in INSULIN_WORDS)),
    })
    patients = rec.groupby("patient_id", as_index=False).first().drop(columns=["recording_id"])

    cgm_parts, meal_parts = [], []
    for f in sorted((root / "Shanghai_T2DM").glob("*.xls*")):
        d = pd.read_excel(f)
        d.columns = [str(c).strip() for c in d.columns]
        rid = f.stem
        pid = "SH-" + rid[:4]
        ts = pd.to_datetime(d["Date"], errors="coerce")
        cgm_col = next(c for c in d.columns if c.startswith("CGM"))
        g = pd.to_numeric(d[cgm_col], errors="coerce")
        ok = ts.notna() & g.notna()
        cgm_parts.append(pd.DataFrame({"dataset": "shanghai_t2dm", "patient_id": pid, "recording_id": rid,
                                       "timestamp": ts[ok].to_numpy(), "glucose_mgdl": g[ok].to_numpy()}))
        diet_col = "Dietary intake" if "Dietary intake" in d.columns else None
        if diet_col:
            m = d[diet_col].notna() & ts.notna()
            meal_parts.append(pd.DataFrame({"dataset": "shanghai_t2dm", "patient_id": pid, "recording_id": rid,
                                            "timestamp": ts[m].to_numpy(), "carbs_g": np.nan,
                                            "description": d.loc[m, diet_col].astype(str).str.replace("\n", "; ").to_numpy()}))
    cgm = pd.concat(cgm_parts, ignore_index=True).sort_values(["recording_id", "timestamp"]).drop_duplicates(["recording_id", "timestamp"])
    meals = pd.concat(meal_parts, ignore_index=True)
    return {"patients": patients, "recordings": rec, "cgm": cgm.reset_index(drop=True), "meals": meals}


@lru_cache(maxsize=1)
def load_cgmacros() -> dict[str, pd.DataFrame]:
    root = _require(REAL / "cgmacros" / "CGMacros")
    bio = pd.read_csv(root / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    patients = pd.DataFrame({
        "dataset": "cgmacros",
        "patient_id": "CM-" + bio["subject"].astype(int).astype(str).str.zfill(3),
        "age_yr": bio["Age"], "sex": bio["Gender"], "bmi": bio["BMI"],
        "hba1c_pct": bio["A1c PDL (Lab)"],   # stored in % despite the dictionary saying mmol/mol
        "fpg_mgdl": bio["Fasting GLU - PDL (Lab)"], "duration_yr": np.nan, "egfr": np.nan,
        "glucose_lowering_agents": "unknown", "on_insulin": False,
    })
    patients["group"] = np.select([patients["hba1c_pct"] >= 6.5, patients["hba1c_pct"] >= 5.7], ["t2d", "prediabetes"], "healthy")

    cgm_parts, meal_parts = [], []
    for f in sorted(root.glob("CGMacros-*/CGMacros-*.csv")):
        pid = "CM-" + f.stem.split("-")[1]
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        ts = pd.to_datetime(d["Timestamp"], errors="coerce")
        lib = pd.to_numeric(d["Libre GL"], errors="coerce")
        # original Libre points sit on a 15-minute grid anchored at the first reading
        t0 = ts[lib.notna()].min()
        on_grid = ((ts - t0).dt.total_seconds() / 60 % 15 == 0) & lib.notna()
        cgm_parts.append(pd.DataFrame({"dataset": "cgmacros", "patient_id": pid, "recording_id": pid,
                                       "timestamp": ts[on_grid].to_numpy(), "glucose_mgdl": lib[on_grid].round(1).to_numpy()}))
        m = d["Meal Type"].notna()
        meal_parts.append(pd.DataFrame({"dataset": "cgmacros", "patient_id": pid, "recording_id": pid,
                                        "timestamp": ts[m].to_numpy(), "carbs_g": pd.to_numeric(d.loc[m, "Carbs"], errors="coerce").to_numpy(),
                                        "description": d.loc[m, "Meal Type"].astype(str).to_numpy()}))
    cgm = pd.concat(cgm_parts, ignore_index=True)
    meals = pd.concat(meal_parts, ignore_index=True)
    return {"patients": patients, "cgm": cgm, "meals": meals}


def real_t2d() -> dict[str, pd.DataFrame]:
    """Both datasets, T2D participants only, in the common format."""
    sh, cm = load_shanghai(), load_cgmacros()
    cm_t2d = cm["patients"][cm["patients"]["group"] == "t2d"]
    ids = set(cm_t2d["patient_id"])
    return {
        "patients": pd.concat([sh["patients"], cm_t2d.drop(columns=["group"])], ignore_index=True),
        "cgm": pd.concat([sh["cgm"], cm["cgm"][cm["cgm"]["patient_id"].isin(ids)]], ignore_index=True),
        "meals": pd.concat([sh["meals"], cm["meals"][cm["meals"]["patient_id"].isin(ids)]], ignore_index=True),
    }
