"""Replay a synthetic patient's TRUE simulation (validation only).

Because every random stream is keyed by master seed + patient id, one patient's behaviour and
physiology can be regenerated exactly, then re-simulated with an intervention applied to the
true inputs. That gives the ground-truth counterfactual the twin's what-if answers are
checked against. Nothing in the dashboard or the twin itself imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from common import config
from data_gen import behaviour as bh
from data_gen import physiology as ph
from data_gen.cohort import generate_cohort
from data_gen.generate import _doses_to_events, load_configs


@dataclass
class TruePatient:
    pid: str
    latent: pd.DataFrame           # one row, true hidden parameters
    meals: pd.DataFrame            # true meals (minute index from simulation start)
    steps: np.ndarray              # true steps per minute
    med_doses: pd.DataFrame
    daily: pd.DataFrame
    t_start: pd.Timestamp          # minute 0 (start of the warm-up day)
    n_min: int
    cfgs: dict

    def inputs(self, meals: pd.DataFrame | None = None, extra_steps: np.ndarray | None = None,
               med_doses: pd.DataFrame | None = None) -> ph.SimInputs:
        pc = self.cfgs["physiology"]
        m = self.meals if meals is None else meals
        ra = ph.meal_appearance(m, self.latent, pc, self.n_min, int(self.cfgs["pipeline"]["seed"]))
        si, egp = ph.daily_modulation(self.daily, self.latent, pc, self.n_min // 1440, int(self.cfgs["pipeline"]["seed"]))
        steps = self.steps.astype(np.float32)
        if extra_steps is not None:
            steps = np.maximum(steps, extra_steps.astype(np.float32))
        ex = steps[None, :] * pc["exercise"]["intensity_per_step_rate"]
        doses = _doses_to_events(self.med_doses if med_doses is None else med_doses, {self.pid: 0}, pc)
        return ph.SimInputs(ra=ra, exercise=ex, si_mult=si, egp_mult=egp, doses=doses)

    def simulate(self, **kw) -> np.ndarray:
        return ph.simulate(self.latent, self.inputs(**kw), self.cfgs["physiology"]).glucose[0]

    def minute(self, ts) -> int:
        return int((pd.Timestamp(ts) - self.t_start) / pd.Timedelta(minutes=1))


@lru_cache(maxsize=1)
def _cohort_and_latent():
    cfgs = load_configs()
    pc = cfgs["pipeline"]
    root = config.REPO_ROOT / pc["output"]["dir"] / pc["dataset_version"]
    import json

    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cohort = generate_cohort(cfgs["cohort"], int(man["n_patients"]), int(pc["seed"]), pc["cohort"]["id_prefix"])
    latent = pd.read_parquet(root / "truth" / "latent.parquet")
    return cfgs, cohort, latent, int(man["n_days"])


def regenerate(pid: str) -> TruePatient:
    cfgs, cohort, latent, n_days = _cohort_and_latent()
    pc = cfgs["pipeline"]
    warm = int(pc["simulation"]["warmup_days"])
    days_total = n_days + warm
    t_start = pd.Timestamp(pc["simulation"]["start_date"]) - pd.Timedelta(days=warm)
    p = cohort.patients.set_index("patient_id").loc[pid]
    p = pd.Series({"patient_id": pid, **p.to_dict()})
    meds = cohort.medications[cohort.medications["patient_id"] == pid]
    adh = float(cohort.truth.set_index("patient_id").loc[pid, "adherence_true"])
    lib = cfgs["meal_library"]
    beh = bh.generate_patient(p, meds, adh, lib["plates"], set(lib.get("heavy_meal_ids", [])), cfgs["behaviour"], days_total, t_start,
                              int(pc["seed"]))
    lat = latent[latent["patient_id"] == pid].reset_index(drop=True)
    lat = lat[[c for c in lat.columns if lat[c].dtype != object or c == "patient_id"]]
    return TruePatient(pid=pid, latent=lat, meals=beh["meals"], steps=beh["steps"], med_doses=beh["med_doses"], daily=beh["daily"],
                       t_start=t_start, n_min=days_total * 1440, cfgs=cfgs)
