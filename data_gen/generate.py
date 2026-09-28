"""Build and freeze the synthetic dataset: cohort -> behaviour -> physiology -> sensors.

    uv run python -m data_gen.generate                          # full dataset from configs/
    uv run python -m data_gen.generate --n-patients 20 --days 3 --out data/synthetic/smoke

Outputs go to <out>/<dataset_version>/:
    observed/   what a clinic system would hold (EHR tables, CGM, wearables, meal log)
    truth/      hidden simulator state (latent physiology, true glucose, true meals, doses,
                daily state). Used only for validation and explanation checks, never as features.
    manifest.json   seed, config hashes, library versions, row counts and content hashes
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from common import config, seeding
from common.io import content_hash, write_parquet
from data_gen import behaviour as bh
from data_gen import physiology as ph
from data_gen import sensors as sn
from data_gen.cohort import generate_cohort

CHUNK = 100


def load_configs(overrides: dict | None = None, use_calibrated: bool = True) -> dict:
    cfgs = {
        "pipeline": config.load("pipeline"),
        "cohort": config.load("cohort"),
        "behaviour": config.load("behaviour"),
        "meal_library": config.load("meal_library"),
        "sensors": config.load("sensors"),
    }
    cfgs["physiology"] = config.load("physiology")
    # Population scalars fitted by validation/calibrate.py, one overlay per config name.
    calibrated = config.CONFIG_DIR / "calibrated.yaml"
    if calibrated.exists() and use_calibrated:
        for key, val in config.resolve(config.load_raw(calibrated.name)).get("overlays", {}).items():
            cfgs[key] = config.deep_merge(cfgs[key], val)
    for key, val in (overrides or {}).items():
        cfgs[key] = config.deep_merge(cfgs[key], val)
    return cfgs


def assign_splits(patient_ids: pd.Series, split_cfg: dict, seed: int) -> pd.DataFrame:
    r = seeding.rng(seed, "split")
    order = r.permutation(len(patient_ids))
    n = len(patient_ids)
    n_train = int(round(split_cfg["train"] * n))
    n_val = int(round(split_cfg["val"] * n))
    split = np.empty(n, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_val]] = "val"
    split[order[n_train + n_val :]] = "test"
    return pd.DataFrame({"patient_id": patient_ids.to_numpy(), "split": split})


def _doses_to_events(med_doses: pd.DataFrame, row_of: dict, pcfg: dict) -> dict[int, list]:
    events: dict[int, list] = {}
    for d in med_doses.itertuples(index=False):
        comp, amt = ph.dose_amount(d.drug, d.drug_class, d.daily_dose, d.n_times, pcfg)
        if comp:
            events.setdefault(int(d.minute), []).append((row_of[d.patient_id], comp, amt))
    return events


def generate(cfgs: dict, n_patients: int | None = None, n_days: int | None = None, verbose: bool = True) -> dict:
    pc = cfgs["pipeline"]
    seed = int(pc["seed"])
    n = int(n_patients or pc["cohort"]["n_patients"])
    days_obs = int(n_days or pc["simulation"]["n_days"])
    warm = int(pc["simulation"]["warmup_days"])
    days_total = days_obs + warm
    t_start = pd.Timestamp(pc["simulation"]["start_date"]) - pd.Timedelta(days=warm)
    first_minute = warm * 1440
    true_every = int(pc["sampling"]["store_true_glucose_interval_min"])
    tic = time.time()

    cohort = generate_cohort(cfgs["cohort"], n, seed, pc["cohort"]["id_prefix"])
    latent = ph.derive_latent(cohort.patients, cohort.medications, cohort.truth, cfgs["physiology"], seed)
    if verbose:
        print(f"[generate] cohort + latent: {n} patients ({time.time() - tic:.1f}s)")

    out: dict[str, list] = {k: [] for k in (
        "cgm", "hr", "steps", "sleep", "hrv", "meal_log", "glucose_true", "meals", "doses", "daily", "nights", "habits", "sensor_params")}
    for c0 in range(0, n, CHUNK):
        pts = cohort.patients.iloc[c0 : c0 + CHUNK].reset_index(drop=True)
        lat = latent.iloc[c0 : c0 + CHUNK].reset_index(drop=True)
        ids = set(pts["patient_id"])
        beh = bh.generate_behaviour(
            pts, cohort.medications[cohort.medications["patient_id"].isin(ids)], cohort.truth[cohort.truth["patient_id"].isin(ids)],
            cfgs["meal_library"], cfgs["behaviour"], days_total, t_start, seed,
        )
        row_of = {pid: i for i, pid in enumerate(lat["patient_id"])}
        ra = ph.meal_appearance(beh.meals, lat, cfgs["physiology"], days_total * 1440, seed)
        si_mult, egp_mult = ph.daily_modulation(beh.daily, lat, cfgs["physiology"], days_total, seed)
        exercise = (beh.steps.astype(np.float32) * cfgs["physiology"]["exercise"]["intensity_per_step_rate"])
        inputs = ph.SimInputs(ra=ra, exercise=exercise, si_mult=si_mult, egp_mult=egp_mult,
                              doses=_doses_to_events(beh.med_doses, row_of, cfgs["physiology"]))
        sim = ph.simulate(lat, inputs, cfgs["physiology"])
        for i, pid in enumerate(lat["patient_id"]):
            cgm, sp = sn.cgm_for_patient(pid, sim.glucose[i], beh.sleep_stage[i], cfgs["sensors"], seed, first_minute)
            out["cgm"].append(cgm)
            hv = beh.hrv[beh.hrv["patient_id"] == pid]
            wr = sn.wearable_for_patient(pid, beh.steps[i], beh.heart_rate[i], beh.sleep_stage[i], hv, cfgs["sensors"], seed, first_minute)
            sp["wearable_worn_fraction"] = round(wr["worn_fraction"], 3)
            out["sensor_params"].append(sp)
            out["hr"].append(wr["heart_rate"])
            out["steps"].append(wr["steps"])
            out["sleep"].append(wr["sleep_stages"])
            out["hrv"].append(wr["hrv"])
            g = sim.glucose[i, first_minute::true_every]
            out["glucose_true"].append(pd.DataFrame({"patient_id": pid, "minute": np.arange(first_minute, days_total * 1440, true_every),
                                                     "glucose_mgdl": np.round(g.astype(float), 1)}))
        out["meal_log"].append(beh.meal_log[beh.meal_log["minute"] >= first_minute])
        out["meals"].append(beh.meals[beh.meals["minute"] >= first_minute])
        out["doses"].append(beh.med_doses[beh.med_doses["minute"] >= first_minute])
        out["daily"].append(beh.daily[beh.daily["day"] >= warm])
        out["nights"].append(beh.nights[beh.nights["bedtime_min"] >= first_minute - 1440])
        out["habits"].append(beh.habits)
        if verbose:
            print(f"[generate] patients {c0 + 1}-{c0 + len(pts)} simulated ({time.time() - tic:.1f}s)")

    def cat(key: str) -> pd.DataFrame:
        return pd.concat(out[key], ignore_index=True)

    def stamp(df: pd.DataFrame, col: str = "minute", jitter: str | None = None) -> pd.DataFrame:
        df = df.copy()
        ts = t_start + pd.to_timedelta(df[col].to_numpy(), unit="m")
        if jitter:
            ts = ts + pd.to_timedelta(df[jitter].to_numpy(), unit="s")
            df = df.drop(columns=[jitter])
        df.insert(1, "timestamp", ts)
        return df.drop(columns=[col])

    truth_latent = latent.merge(pd.DataFrame(out["sensor_params"]), on="patient_id").merge(cat("habits"), on="patient_id")
    nights = cat("nights")
    for col in ("bedtime_min", "wake_min"):
        nights[col.replace("_min", "")] = t_start + pd.to_timedelta(nights[col], unit="m")
    nights = nights.drop(columns=["bedtime_min", "wake_min"])
    daily = cat("daily")
    daily["day"] = daily["day"] - warm

    tables = {
        "observed/patients": cohort.patients,
        "observed/hba1c_history": cohort.hba1c_history,
        "observed/medications": cohort.medications,
        "observed/cgm": stamp(cat("cgm")),
        "observed/wearable_heart_rate": stamp(cat("hr"), jitter="jitter_s"),
        "observed/wearable_steps": stamp(cat("steps")),
        "observed/wearable_sleep_stages": stamp(cat("sleep")),
        "observed/wearable_hrv": stamp(cat("hrv")),
        "observed/meal_log": stamp(cat("meal_log")),
        "observed/splits": assign_splits(cohort.patients["patient_id"], pc["split"], seed),
        "truth/latent": truth_latent,
        "truth/glucose_true": stamp(cat("glucose_true")),
        "truth/meals_true": stamp(cat("meals").drop(columns=["day"])),
        "truth/med_doses": stamp(cat("doses")),
        "truth/daily_state": daily,
        "truth/nights": nights,
    }
    if verbose:
        print(f"[generate] done in {time.time() - tic:.1f}s")
    return tables


def write_dataset(tables: dict, cfgs: dict, out_dir: Path, n_patients: int, n_days: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_version": cfgs["pipeline"]["dataset_version"],
        "seed": cfgs["pipeline"]["seed"],
        "n_patients": n_patients,
        "n_days": n_days,
        "config_hashes": {k: config.config_hash(v) for k, v in cfgs.items()},
        "libraries": {lib: version(lib) for lib in ("numpy", "pandas", "scipy", "pyarrow")},
        "python": platform.python_version(),
        "tables": {},
    }
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        write_parquet(df, path)
        manifest["tables"][name] = {"rows": int(len(df)), "columns": list(map(str, df.columns)), "content_sha256": content_hash(df)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="output root (default: pipeline.output.dir)")
    args = ap.parse_args()
    cfgs = load_configs()
    n = args.n_patients or cfgs["pipeline"]["cohort"]["n_patients"]
    days = args.days or cfgs["pipeline"]["simulation"]["n_days"]
    tables = generate(cfgs, n, days)
    root = Path(args.out) if args.out else config.REPO_ROOT / cfgs["pipeline"]["output"]["dir"]
    path = write_dataset(tables, cfgs, root / cfgs["pipeline"]["dataset_version"], n, days)
    print(f"[generate] wrote {path}")


if __name__ == "__main__":
    main()
