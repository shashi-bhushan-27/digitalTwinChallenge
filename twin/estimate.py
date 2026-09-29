"""Personalise the twin: estimate a patient's physiology from their own observed data.

Four quantities are fitted to the first days of CGM, starting from the record-based prior:
insulin sensitivity and beta-cell function (log multipliers on the prior), the fasting set
point, and how much the patient under-reports carbohydrate in the meal log. The fit uses the
windows the twin can explain: 3 hours after each logged meal and the night-time hours (2-6 h).

Search: a coarse grid, then local grids around the best point; every candidate is simulated in
one vectorised run. Loss = squared CGM residuals (thinned for autocorrelation) + Gaussian prior
penalties. A rough posterior over the last grid gives the uncertainty shown on the twin card.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_gen import physiology as ph
from twin.model import (
    Observed,
    Window,
    build_inputs,
    imputed_meals,
    lagged,
    logged_meals,
    prior_latent,
    twin_physiology_config,
    with_params,
)


@dataclass
class TwinState:
    patient_id: str
    latent: pd.DataFrame          # one row: personalised parameters (hepatic output re-solved)
    prior: pd.DataFrame           # one row: record-only parameters
    carb_scale: float
    si_log: float
    beta_log: float
    gf_log: float
    si_log_sd: float
    beta_log_sd: float
    fit: dict
    meals: pd.DataFrame           # logged + imputed meals the twin uses over the whole record


def _fit_mask(obs: Observed, meals: pd.DataFrame, win: Window, ecfg: dict) -> dict:
    """CGM points inside the windows the twin can explain.

    Post-meal points (0-180 min after a logged meal) are compared as rises from the reading at
    meal time, so hidden day-to-day shifts in level do not distort the meal response; night points
    (2-6 h) are compared as absolute levels and pin the fasting set point. Returns minute
    indices, values, and for each point the anchor minute/value (-1 for night points)."""
    c = obs.cgm.dropna(subset=["glucose_mgdl"])
    mins = win.minute(c["timestamp"])
    vals = c["glucose_mgdl"].to_numpy(float)
    ok = (mins >= ecfg["start_skip_h"] * 60) & (mins < win.n_min)
    hours = pd.DatetimeIndex(c["timestamp"]).hour
    night = (hours >= ecfg["night_window_h"][0]) & (hours < ecfg["night_window_h"][1])
    anchor_min = np.full(len(c), -1)
    anchor_val = np.full(len(c), np.nan)
    ts = c["timestamp"].to_numpy().astype("datetime64[ns]")
    for t in meals.loc[meals["source"] == "logged", "timestamp"].sort_values():
        t64 = np.datetime64(t, "ns")
        before = np.flatnonzero((ts <= t64) & (ts >= t64 - np.timedelta64(20, "m")))
        if not len(before):
            continue
        a = before[-1]
        dt = (ts - t64) / np.timedelta64(1, "m")
        w = (dt > 0) & (dt <= ecfg["post_meal_window_min"])
        anchor_min[w] = mins[a]          # later meals overwrite earlier ones: the most recent meal anchors
        anchor_val[w] = vals[a]
    post = anchor_min >= 0
    keep = ok & (night | post) & ((~post) | (anchor_min >= ecfg["start_skip_h"] * 60))
    return {"mins": mins[keep], "y": vals[keep], "anchor_min": anchor_min[keep], "anchor_val": anchor_val[keep]}


def fit(obs: Observed, pcfg: dict, tcfg: dict, library: list[dict], fit_days: float | None = None) -> TwinState:
    ecfg = tcfg["estimation"]
    q = twin_physiology_config(pcfg)
    prior = prior_latent(obs.patient, obs.meds, pcfg)
    logged = logged_meals(obs, library, tcfg)
    meals = pd.concat([logged, imputed_meals(obs, logged, tcfg)], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    t0 = pd.Timestamp(obs.cgm["timestamp"].min()).floor("min")
    win = Window(t0, int((fit_days or ecfg["fit_days"]) * 1440))
    fm = _fit_mask(obs, meals, win, ecfg)
    mins, y, amin, aval = fm["mins"], fm["y"], fm["anchor_min"], fm["anchor_val"]
    post = amin >= 0
    target = np.where(post, y - aval, y)
    lag = ecfg["cgm_lag_min"]
    sd = ecfg["prior_sd"]
    sig2 = ecfg["residual_sd_mgdl"] ** 2
    thin = ecfg["autocorr_thinning"]

    def evaluate(si, be, cs, gf):
        lat = with_params(prior, si, be, gf, pcfg)
        inp = build_inputs(lat, meals, obs, win, pcfg, carb_scale=cs)
        g = ph.simulate(lat, inp, q, record_every=720).glucose
        full = lagged(g.astype(float), lag)
        pred = full[:, mins]
        pred = np.where(post[None, :], pred - full[:, np.maximum(amin, 0)], pred)
        sse = ((target[None, :] - pred) ** 2).sum(1)
        pen = (si / sd["si_log"]) ** 2 + (be / sd["beta_log"]) ** 2 + (np.log(cs / ecfg["carb_scale_prior_centre"]) / sd["carb_scale_log"]) ** 2 + (gf / 0.15) ** 2
        return sse / sig2 / thin + pen, sse

    gr = ecfg["grid"]
    axes = [np.linspace(*gr["si_log"][:2], int(gr["si_log"][2])), np.linspace(*gr["beta_log"][:2], int(gr["beta_log"][2])),
            np.asarray(gr["carb_scale"], float), np.linspace(-0.2, 0.2, 5)]
    # the full grid is large; search SI x beta first at central carb/fasting values, then all four locally
    si_g, be_g = np.meshgrid(axes[0], axes[1], indexing="ij")
    si, be = si_g.ravel(), be_g.ravel()
    c0 = ecfg["carb_scale_prior_centre"]
    loss, _ = evaluate(si, be, np.full(len(si), c0), np.zeros(len(si)))
    best = np.array([si[np.argmin(loss)], be[np.argmin(loss)], c0, 0.0])
    widths = np.array([0.6, 0.6, 0.3 if ecfg.get("fit_carb_scale", True) else 0.0, 0.12])
    rf = ecfg["refine"]
    for _ in range(rf["rounds"] + 1):
        k = rf["points"]
        grids = [np.linspace(best[0] - widths[0], best[0] + widths[0], k), np.linspace(best[1] - widths[1], best[1] + widths[1], k),
                 np.unique(np.clip(np.linspace(best[2] - widths[2], best[2] + widths[2], 3), *ecfg["carb_scale_bounds"])),
                 np.linspace(best[3] - widths[3], best[3] + widths[3], 3)]
        G = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 4)
        loss, sse = evaluate(G[:, 0], G[:, 1], G[:, 2], G[:, 3])
        best = G[int(np.argmin(loss))]
        widths = widths * rf["shrink"] * 2
    w = np.exp(-(loss - loss.min()) / 2)
    w /= w.sum()
    si_sd = float(np.sqrt(np.sum(w * (G[:, 0] - np.sum(w * G[:, 0])) ** 2)))
    be_sd = float(np.sqrt(np.sum(w * (G[:, 1] - np.sum(w * G[:, 1])) ** 2)))
    # a tight final grid understates uncertainty: floor it at half the last grid step
    si_sd = max(si_sd, (grids[0][1] - grids[0][0]) / 2)
    be_sd = max(be_sd, (grids[1][1] - grids[1][0]) / 2)

    lat = with_params(prior, np.array([best[0]]), np.array([best[1]]), np.array([best[3]]), pcfg)
    _, sse_prior = evaluate(np.zeros(1), np.zeros(1), np.full(1, 1.0), np.zeros(1))
    rmse_fit = float(np.sqrt(sse[int(np.argmin(loss))] / max(len(y), 1)))
    return TwinState(
        patient_id=obs.patient["patient_id"], latent=lat, prior=prior, carb_scale=float(best[2]), si_log=float(best[0]), beta_log=float(best[1]),
        gf_log=float(best[3]), si_log_sd=si_sd, beta_log_sd=be_sd, meals=meals,
        fit={"rmse_fit_mgdl": rmse_fit, "rmse_prior_mgdl": float(np.sqrt(sse_prior[0] / max(len(y), 1))), "n_points": int(len(y)),
             "n_logged_meals": int((meals["source"] == "logged").sum()), "n_imputed_meals": int((meals["source"] == "imputed").sum()),
             "fit_start": str(win.t0), "fit_days": win.n_min / 1440},
    )
