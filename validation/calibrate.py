"""Tune a few population-level generator scalars against the real CALIBRATION split only.

    uv run python -m validation.calibrate

Random search (log-uniform, fixed seed) over the ranges in configs/validation.yaml. Each
candidate generates a small level-matched pool, computes per-patient dynamics and scores the
median gaps against the Shanghai calibration patients (scaled by their IQR). The best
candidate is written to configs/calibrated.yaml, which data_gen.generate applies as an overlay.
Held-out real patients (Shanghai validation half, CGMacros) are never looked at here.
"""

from __future__ import annotations

import json
import time

import numpy as np
import yaml

from common import config, seeding
from data_gen.generate import generate, load_configs
from validation.compare import (
    compare_table,
    loss,
    match_by_mean,
    pool_configs,
    real_sets,
    summarise,
    synthetic_observed,
)


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def evaluate(overlays: dict, vcfg: dict, real_summary, n: int, days: int) -> tuple[float, object]:
    cfgs = pool_configs(load_configs(overrides=overlays, use_calibrated=False), vcfg)
    tables = generate(cfgs, n, days, verbose=False)
    cgm, meals = synthetic_observed(tables)
    syn = summarise(cgm, meals, vcfg["resample_min"])
    matched = match_by_mean(real_summary, syn)
    table = compare_table(real_summary, matched, vcfg["compare_metrics"])
    return loss(table, vcfg["loss_metrics"]), table


def main() -> None:
    vcfg = config.load("validation")
    cal = vcfg["calibration"]
    sets = real_sets(vcfg)
    real = summarise(sets["calibration"]["cgm"], sets["calibration"]["meals"])
    print(f"[calibrate] {len(real)} calibration patients (Shanghai), searching {cal['n_candidates']} candidates")
    r = seeding.rng(vcfg["seed"], "calibration_search")
    names = list(cal["search"])
    candidates = [{k: 1.0 for k in names}]  # defaults first, as the reference
    defaults = load_configs(use_calibrated=False)
    for k in names:
        cfg_name, *path = k.split(".")
        node = defaults[cfg_name]
        for p in path:
            node = node[p]
        candidates[0][k] = float(node)
    for _ in range(cal["n_candidates"] - 1):
        candidates.append({k: float(np.exp(r.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in cal["search"].items()})

    results = []
    tic = time.time()

    def run_candidate(cand: dict, label: str) -> None:
        overlays: dict = {}
        for k, v in cand.items():
            cfg_name, rest = k.split(".", 1)
            _set(overlays.setdefault(cfg_name, {}), rest, v)
        score, table = evaluate(overlays, vcfg, real, cal["n_patients"], cal["n_days"])
        results.append((score, cand, overlays, table))
        print(f"[calibrate] {label} loss={score:7.3f} " + " ".join(f"{k.split('.')[-1]}={v:.3g}" for k, v in cand.items())
              + f" ({time.time() - tic:.0f}s)")

    for i, cand in enumerate(candidates):
        run_candidate(cand, f"{i + 1:2d}/{len(candidates)}")
    # local refinement: log-normal steps around the current best, clipped to the ranges
    for j in range(cal.get("n_refine", 0)):
        best = min(results, key=lambda x: x[0])[1]
        cand = {}
        for k, (lo, hi) in cal["search"].items():
            step = cal["refine_sd"] * (np.log(hi) - np.log(lo))
            cand[k] = float(np.clip(np.exp(np.log(best[k]) + r.normal(0, step)), lo, hi))
        run_candidate(cand, f"refine {j + 1:2d}/{cal['n_refine']}")

    results.sort(key=lambda x: x[0])
    best_score, best, overlays, table = results[0]
    default_score = next(s for s, c, _, _ in results if c is candidates[0])
    out = {
        "note": "Written by validation/calibrate.py; tuned on the Shanghai calibration split only.",
        "loss_default": round(default_score, 4),
        "loss_best": round(best_score, 4),
        "overlays": json.loads(json.dumps(overlays)),
    }
    path = config.CONFIG_DIR / "calibrated.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    res_dir = config.REPO_ROOT / "results" / "validation"
    res_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(res_dir / "calibration_best_vs_calibration_split.csv", index=False)
    with open(res_dir / "calibration_search.json", "w", encoding="utf-8") as fh:
        json.dump([{"loss": s, "params": c} for s, c, _, _ in results], fh, indent=2)
    print(f"[calibrate] default loss {default_score:.3f} -> best {best_score:.3f}; wrote {path}")
    print(table[["metric", "real_median", "synthetic_median", "median_gap_in_real_iqr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
