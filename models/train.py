"""Train baselines, the ablation ladder and the fusion models; evaluate on held-out test patients.

    uv run python -m models.features      # once per dataset version
    uv run python -m models.train

Splits are by patient (observed/splits.parquet): train fits the models, validation drives early
stopping, isotonic calibration, the conformal band width and the alert threshold, and every
reported number comes from the untouched test patients.

Ablation ladder (as CLAUDE.md asks): persistence -> CGM only -> + wearables -> + EHR, then
+ meal log (the with/without meal-information comparison). The last rung is the product model
("fusion"): point and quantile forecasts at 15-120 min with a conformal 80 % band, a peak-glucose
model for the explanation panel, and a calibrated spike-onset classifier.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from common import config
from models import evaluate as E
from models.features import feature_columns

RES = config.REPO_ROOT / "results" / "models"
ART = config.REPO_ROOT / "models" / "artifacts"
LADDER = [
    ("CGM only", ("cgm",)),
    ("CGM + wearables", ("cgm", "wear")),
    ("CGM + wearables + EHR", ("cgm", "wear", "ehr")),
    ("CGM + wearables + EHR + meal log", ("cgm", "wear", "ehr", "meal")),
]
FUSION = LADDER[-1]


def load_features() -> pd.DataFrame:
    from data_gen.generate import load_configs

    cfgs = load_configs()
    path = config.REPO_ROOT / "data" / "features" / cfgs["pipeline"]["dataset_version"] / "features.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `uv run python -m models.features` first")
    return pd.read_parquet(path)


def fit_lgb(mcfg: dict, Xtr, ytr, Xva, yva, objective: str, alpha: float | None = None) -> lgb.Booster:
    p = {k: v for k, v in mcfg["lgb"].items() if k not in ("num_boost_round", "early_stopping_rounds")}
    p.update({"objective": objective, "verbose": -1, "seed": mcfg["seed"], "deterministic": True})
    if alpha is not None:
        p["alpha"] = alpha
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=mcfg["lgb"]["num_boost_round"], valid_sets=[dva],
                     callbacks=[lgb.early_stopping(mcfg["lgb"]["early_stopping_rounds"], verbose=False)])


def _rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[df[target].notna()]


def train_forecasts(mcfg, tr, va, feats, horizons, quantiles: bool) -> dict:
    """Models of the change from the current reading at each horizon (point, and q10/q90)."""
    models = {}
    for h in horizons:
        a, b = _rows(tr, f"y_{h}"), _rows(va, f"y_{h}")
        ya, yb = a[f"y_{h}"] - a["g_now"], b[f"y_{h}"] - b["g_now"]
        models[(h, "point")] = fit_lgb(mcfg, a[feats], ya, b[feats], yb, "regression")
        if quantiles:
            for q in (mcfg["quantiles"][0], mcfg["quantiles"][-1]):
                models[(h, f"q{int(q * 100)}")] = fit_lgb(mcfg, a[feats], ya, b[feats], yb, "quantile", alpha=q)
    return models


def train_classifier(mcfg, tr, va, feats) -> tuple[lgb.Booster, IsotonicRegression]:
    a, b = tr[tr["spike_eligible"]], va[va["spike_eligible"]]
    model = fit_lgb(mcfg, a[feats], a["spike_label"].astype(int), b[feats], b["spike_label"].astype(int), "binary")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(model.predict(b[feats]), b["spike_label"].astype(int))
    return model, iso


def conformal_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Split-conformal widening (CQR) so [lo - Q, hi + Q] covers `coverage` of validation cases."""
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * coverage) / n)))


def evaluate_ladder(mcfg, tr, va, te, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics for baselines and ladder rungs; per-row test predictions are added to `preds`."""
    rows, cls_rows = [], []
    hz = mcfg["ablation_horizons_min"]
    el = te[te["spike_eligible"]]
    # baselines without training
    for h in hz:
        y = te[f"y_{h}"].to_numpy(float)
        rows.append({"model": "persistence", "horizon_min": h, **E.forecast_metrics(y, te["g_now"].to_numpy(float))})
        lin = (te["g_now"] + te["cgm_delta_30"] / 30 * h).to_numpy(float)
        rows.append({"model": "linear extrapolation (30-min trend)", "horizon_min": h, **E.forecast_metrics(y, lin)})
    cls_rows.append({"model": "persistence (current glucose as score)", **E.classification_metrics(el["spike_label"], el["g_now"].to_numpy() / 400)})
    for name, groups in LADDER[:-1]:
        feats = feature_columns(tr, groups)
        t0 = time.time()
        fm = train_forecasts(mcfg, tr, va, feats, hz, quantiles=False)
        for h in hz:
            pred = te["g_now"].to_numpy(float) + fm[(h, "point")].predict(te[feats])
            rows.append({"model": name, "horizon_min": h, **E.forecast_metrics(te[f"y_{h}"].to_numpy(float), pred)})
            if h == 60:
                preds[f"f60 | {name}"] = pred
        clf, iso = train_classifier(mcfg, tr, va, feats)
        p = iso.predict(clf.predict(el[feats]))
        preds[f"p | {name}"] = np.nan
        preds.loc[te["spike_eligible"].to_numpy(), f"p | {name}"] = p
        cls_rows.append({"model": name, **E.classification_metrics(el["spike_label"], p), "n_features": len(feats)})
        print(f"[train] ladder '{name}' ({len(feats)} features) done in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(cls_rows)


def train_fusion(mcfg, tr, va, te, name: str, groups: tuple) -> dict:
    feats = feature_columns(tr, groups)
    t0 = time.time()
    hz = mcfg["horizons_min"]
    fm = train_forecasts(mcfg, tr, va, feats, hz, quantiles=True)
    cov = mcfg["band_coverage"]
    qlo, qhi = f"q{int(mcfg['quantiles'][0] * 100)}", f"q{int(mcfg['quantiles'][-1] * 100)}"
    meta = {"name": name, "features": feats, "horizons_min": hz, "conformal": {}, "band_coverage": cov}
    rows = []
    for h in hz:
        b = _rows(va, f"y_{h}")
        lo = b["g_now"].to_numpy() + fm[(h, qlo)].predict(b[feats])
        hi = b["g_now"].to_numpy() + fm[(h, qhi)].predict(b[feats])
        Q = conformal_offset(lo, hi, b[f"y_{h}"].to_numpy(float), cov)
        meta["conformal"][str(h)] = Q
        g0 = te["g_now"].to_numpy(float)
        point = g0 + fm[(h, "point")].predict(te[feats])
        lo_t = np.minimum(g0 + fm[(h, qlo)].predict(te[feats]) - Q, point)
        hi_t = np.maximum(g0 + fm[(h, qhi)].predict(te[feats]) + Q, point)
        y = te[f"y_{h}"].to_numpy(float)
        ok = ~np.isnan(y)
        if h == 60:
            meta_pred60 = point
        rows.append({"model": name, "horizon_min": h, **E.forecast_metrics(y, point), **E.clarke_summary(y, point),
                     "band_coverage": float(np.mean((y[ok] >= lo_t[ok]) & (y[ok] <= hi_t[ok]))),
                     "band_width_median": float(np.median(hi_t - lo_t))})
    # peak glucose over the next 2 h (explained in the dashboard)
    a, b = _rows(tr, "peak_120"), _rows(va, "peak_120")
    peak = {k: fit_lgb(mcfg, a[feats], a["peak_120"] - a["g_now"], b[feats], b["peak_120"] - b["g_now"], obj, alpha=al)
            for k, obj, al in (("point", "regression", None), (qlo, "quantile", mcfg["quantiles"][0]), (qhi, "quantile", mcfg["quantiles"][-1]))}
    plo = b["g_now"].to_numpy() + peak[qlo].predict(b[feats])
    phi = b["g_now"].to_numpy() + peak[qhi].predict(b[feats])
    meta["conformal"]["peak_120"] = conformal_offset(plo, phi, b["peak_120"].to_numpy(float), cov)
    t = _rows(te, "peak_120")
    pk = t["g_now"].to_numpy() + peak["point"].predict(t[feats])
    rows.append({"model": name, "horizon_min": "peak in next 120", **E.forecast_metrics(t["peak_120"].to_numpy(float), pk)})

    clf, iso = train_classifier(mcfg, tr, va, feats)
    vb = va[va["spike_eligible"]]
    p_val = iso.predict(clf.predict(vb[feats]))
    thr = E.best_f1_threshold(vb["spike_label"].to_numpy(), p_val)
    el = te[te["spike_eligible"]]
    p_te = iso.predict(clf.predict(el[feats]))
    cls = {"model": name, **E.classification_metrics(el["spike_label"], p_te), **E.at_threshold(el["spike_label"].to_numpy(), p_te, thr)}
    calib = E.calibration_curve(el["spike_label"].to_numpy(), p_te)
    stream = te[["patient_id", "timestamp", "spike_eligible", "spike_onset_event"]].copy()
    stream["p"] = np.nan
    stream.loc[te["spike_eligible"].to_numpy(), "p"] = p_te
    events = E.alert_events(stream, "p", thr, mcfg["spike"]["window_min"], mcfg["alerts"]["refractory_min"])
    grid = []
    for g in mcfg["alerts"]["sensitivity_grid"]:
        grid.append({**E.at_threshold(el["spike_label"].to_numpy(), p_te, g),
                     **{k: v for k, v in E.alert_events(stream, "p", g, mcfg["spike"]["window_min"], mcfg["alerts"]["refractory_min"]).items() if k != "threshold"}})
    meta.update({"threshold": thr, "isotonic": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
                 "sensitivity_grid": grid, "train_patients": int(tr["patient_id"].nunique()), "val_patients": int(va["patient_id"].nunique()),
                 "test_patients": int(te["patient_id"].nunique()), "train_rows": int(len(tr))})
    print(f"[train] fusion '{name}' done in {time.time() - t0:.0f}s")
    return {"forecast_models": fm, "peak_models": peak, "classifier": clf, "isotonic": iso, "meta": meta,
            "forecast_rows": rows, "classification": cls, "calibration": calib, "events": events, "sensitivity": pd.DataFrame(grid),
            "test_pred": {"p": p_te, "y": el["spike_label"].to_numpy(), "f60": meta_pred60}}


def comparisons(preds: pd.DataFrame, mcfg: dict, n_boot: int = 200) -> pd.DataFrame:
    """Paired differences between ladder rungs with patient-level bootstrap 95 % intervals,
    overall and in strata where extra streams could matter most: the first 24 h of a patient's
    data (little CGM history), the first 3 h after waking (the day's state not yet visible in
    glucose), and right after a walk (exercise effect)."""
    from sklearn.metrics import average_precision_score

    names = [n for n, _ in LADDER]
    pairs = [(names[0], names[1]), (names[1], names[2]), (names[2], names[3]), (names[0], names[3])]
    strata = {
        "all test times": np.ones(len(preds), bool),
        "first 24 h of data": (preds["hours_since_data_start"] <= 24).to_numpy(),
        "within 3 h of waking": (preds["wear_min_since_wake"] <= 180).to_numpy(),
        "after a walk (>= 1500 steps in 30 min)": (preds["wear_steps_30"] >= 1500).to_numpy(),
    }
    rng = np.random.default_rng(mcfg["seed"])
    pids = preds["patient_id"].unique()
    rows = []
    for sname, smask in strata.items():
        d = preds[smask]
        per = {}
        for pid, g in d.groupby("patient_id"):
            ok = g["y_60"].notna().to_numpy()
            el = g["spike_eligible"].to_numpy()
            per[pid] = {"err": {n: np.abs(g[f"f60 | {n}"].to_numpy()[ok] - g["y_60"].to_numpy()[ok]) for n in names},
                        "y": g["spike_label"].to_numpy()[el].astype(int), "p": {n: g[f"p | {n}"].to_numpy()[el] for n in names}}
        for a, b in pairs:
            def stat(ids, a=a, b=b, per=per):
                ids = [i for i in ids if i in per]
                ea = np.concatenate([per[i]["err"][a] for i in ids])
                eb = np.concatenate([per[i]["err"][b] for i in ids])
                y = np.concatenate([per[i]["y"] for i in ids])
                if not 0 < y.sum() < len(y):
                    return eb.mean() - ea.mean(), np.nan
                pa = np.concatenate([per[i]["p"][a] for i in ids])
                pb = np.concatenate([per[i]["p"][b] for i in ids])
                return eb.mean() - ea.mean(), average_precision_score(y, pb) - average_precision_score(y, pa)
            point = stat(pids)
            boot = np.array([stat(rng.choice(pids, len(pids), replace=True)) for _ in range(n_boot)])
            rows.append({"stratum": sname, "from": a, "to": b, "n_rows": int(len(d)),
                         "delta_mae60": point[0], "mae60_ci_low": np.nanpercentile(boot[:, 0], 2.5), "mae60_ci_high": np.nanpercentile(boot[:, 0], 97.5),
                         "delta_pr_auc": point[1], "pr_auc_ci_low": np.nanpercentile(boot[:, 1], 2.5), "pr_auc_ci_high": np.nanpercentile(boot[:, 1], 97.5)})
    return pd.DataFrame(rows)


def save_fusion(res: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (h, kind), m in res["forecast_models"].items():
        m.save_model(str(out_dir / f"forecast_{h}_{kind}.txt"))
    for kind, m in res["peak_models"].items():
        m.save_model(str(out_dir / f"peak_{kind}.txt"))
    res["classifier"].save_model(str(out_dir / "spike.txt"))
    (out_dir / "meta.json").write_text(json.dumps(res["meta"], indent=2), encoding="utf-8")


def figures(ladder_fc: pd.DataFrame, ladder_cls: pd.DataFrame, fusion: dict, fusion_fc: pd.DataFrame) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    order = ["persistence", "linear extrapolation (30-min trend)"] + [n for n, _ in LADDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    d = ladder_fc[ladder_fc["horizon_min"] == 60].set_index("model").reindex(order).dropna(subset=["mae"])
    axes[0].barh(range(len(d)), d["mae"], color=["#bbb", "#bbb"] + ["#3b6ea8"] * (len(d) - 2))
    axes[0].set_yticks(range(len(d)), d.index, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE at 60 min (mg/dL), test patients")
    for i, v in enumerate(d["mae"]):
        axes[0].text(v, i, f" {v:.1f}", va="center", fontsize=8)
    c = ladder_cls.set_index("model")
    axes[1].barh(range(len(c)), c["pr_auc"], color=["#bbb"] + ["#d9822b"] * (len(c) - 1))
    axes[1].set_yticks(range(len(c)), c.index, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("PR-AUC, spike onset within 2 h (higher is better)")
    for i, v in enumerate(c["pr_auc"]):
        axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.suptitle("Ablation ladder: does fusing more streams help?", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "ablation.png", dpi=130)
    plt.close(fig)

    cal = fusion["calibration"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-", color="#d9822b")
    for r in cal.itertuples():
        ax.annotate(f"{r.n}", (r.mean_predicted, r.observed_rate), fontsize=6, xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("predicted spike probability")
    ax.set_ylabel("observed spike rate")
    ax.set_title(f"Calibration (test), Brier {fusion['classification']['brier']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "calibration.png", dpi=130)
    plt.close(fig)

    s = fusion["sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s["threshold"], s["event_recall"], "o-", label="spikes caught (event recall)", color="#3b6ea8")
    ax.plot(s["threshold"], s["episode_precision"], "o-", label="alerts followed by a spike", color="#d9822b")
    ax2 = ax.twinx()
    ax2.plot(s["threshold"], s["false_alerts_per_patient_day"], "s--", color="#c0392b", label="false alerts / patient-day")
    ax2.set_ylabel("false alerts per patient-day", color="#c0392b")
    ax.axvline(fusion["meta"]["threshold"], color="#555", lw=0.8, ls=":")
    ax.set_xlabel("alert threshold (calibrated probability)")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=7, loc="center left", frameon=False)
    ax.set_title("Sensitivity slider: recall vs alert fatigue (test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "alert_tradeoff.png", dpi=130)
    plt.close(fig)

    f = fusion_fc[fusion_fc["horizon_min"] != "peak in next 120"].copy()
    p = ladder_fc[ladder_fc["model"] == "persistence"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(f["horizon_min"].astype(int), f["mae"], "o-", color="#3b6ea8", label="fusion (point)")
    ax.plot(p["horizon_min"], p["mae"], "s--", color="#999", label="persistence")
    ax.set_xlabel("horizon (min)")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Forecast error by horizon (test patients)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "horizon_mae.png", dpi=130)
    plt.close(fig)


def main() -> None:
    mcfg = config.load("models")
    df = load_features()
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"[train] rows train {len(tr):,} / val {len(va):,} / test {len(te):,}; patients "
          f"{tr['patient_id'].nunique()} / {va['patient_id'].nunique()} / {te['patient_id'].nunique()}")
    tic = time.time()
    preds = te[["patient_id", "timestamp", "g_now", "y_60", "spike_label", "spike_eligible", "wear_min_since_wake", "wear_steps_30"]].copy()
    start = te.groupby("patient_id")["timestamp"].transform("min")
    preds["hours_since_data_start"] = (te["timestamp"] - start).dt.total_seconds() / 3600 + mcfg["sampling"]["min_history_min"] / 60
    ladder_fc, ladder_cls = evaluate_ladder(mcfg, tr, va, te, preds)
    fusion = train_fusion(mcfg, tr, va, te, FUSION[0], FUSION[1])
    preds[f"f60 | {FUSION[0]}"] = fusion["test_pred"]["f60"]
    preds[f"p | {FUSION[0]}"] = np.nan
    preds.loc[te["spike_eligible"].to_numpy(), f"p | {FUSION[0]}"] = fusion["test_pred"]["p"]
    RES.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(RES / "test_predictions.parquet", index=False)
    comparisons(preds, mcfg).to_csv(RES / "ablation_bootstrap.csv", index=False)
    save_fusion(fusion, ART / "fusion")
    fusion_fc = pd.DataFrame(fusion["forecast_rows"])
    ladder_fc = pd.concat([ladder_fc, fusion_fc[fusion_fc["horizon_min"].isin(mcfg["ablation_horizons_min"])][["model", "horizon_min", "mae", "rmse", "n"]]],
                          ignore_index=True)
    ladder_cls = pd.concat([ladder_cls, pd.DataFrame([{k: v for k, v in fusion["classification"].items()
                                                       if k in ("model", "prevalence", "n", "pr_auc", "auroc", "brier")}])], ignore_index=True)
    RES.mkdir(parents=True, exist_ok=True)
    ladder_fc.to_csv(RES / "ablation_forecast.csv", index=False)
    ladder_cls.to_csv(RES / "ablation_spike.csv", index=False)
    fusion_fc.to_csv(RES / "fusion_forecast.csv", index=False)
    fusion["calibration"].to_csv(RES / "fusion_calibration.csv", index=False)
    fusion["sensitivity"].to_csv(RES / "fusion_alert_sensitivity.csv", index=False)
    summary = {"classification_at_threshold": fusion["classification"], "events_at_threshold": fusion["events"],
               "threshold": fusion["meta"]["threshold"], "conformal_offsets": fusion["meta"]["conformal"],
               "rows": {"train": len(tr), "val": len(va), "test": len(te)}, "runtime_s": round(time.time() - tic)}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures(ladder_fc, ladder_cls, fusion, fusion_fc)
    pd.set_option("display.width", 200)
    print(ladder_fc.pivot_table(index="model", columns="horizon_min", values="mae").round(2).to_string())
    print(ladder_cls[["model", "pr_auc", "auroc", "brier"]].round(3).to_string(index=False))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fusion["events"].items()}, indent=1))
    print(f"[train] done in {time.time() - tic:.0f}s")


if __name__ == "__main__":
    main()
