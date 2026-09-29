"""Model inference behind one small interface (the dashboard only talks to this module).

    predictor = TwinPredictor.load()
    out = predictor.predict(feature_rows)        # forecasts, 80 % band, peak, spike probability
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import config

ART = config.REPO_ROOT / "models" / "artifacts" / "fusion"


@dataclass
class TwinPredictor:
    meta: dict
    forecast: dict
    peak: dict
    spike: lgb.Booster

    @classmethod
    def load(cls, path: Path = ART) -> TwinPredictor:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        forecast = {}
        for h in meta["horizons_min"]:
            for kind in ("point", "q10", "q90"):
                forecast[(h, kind)] = lgb.Booster(model_file=str(path / f"forecast_{h}_{kind}.txt"))
        peak = {k: lgb.Booster(model_file=str(path / f"peak_{k}.txt")) for k in ("point", "q10", "q90")}
        return cls(meta=meta, forecast=forecast, peak=peak, spike=lgb.Booster(model_file=str(path / "spike.txt")))

    @property
    def features(self) -> list[str]:
        return self.meta["features"]

    @property
    def threshold(self) -> float:
        """Dashboard default alert threshold (configs/models.yaml, decision D-025); falls back to the
        validation max-F1 point stored with the model."""
        default = config.load("models").get("alerts", {}).get("default_threshold")
        return float(default if default is not None else self.meta["threshold"])

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        iso = self.meta["isotonic"]
        return np.interp(raw, iso["x"], iso["y"])

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """One output row per input row. Band = quantile models widened by the conformal offset."""
        X = rows[self.features]
        g0 = rows["cgm_now"].to_numpy(float)
        out = pd.DataFrame(index=rows.index)
        for h in self.meta["horizons_min"]:
            q = self.meta["conformal"][str(h)]
            point = g0 + self.forecast[(h, "point")].predict(X)
            out[f"f{h}"] = point
            out[f"f{h}_lo"] = np.minimum(g0 + self.forecast[(h, "q10")].predict(X) - q, point)
            out[f"f{h}_hi"] = np.maximum(g0 + self.forecast[(h, "q90")].predict(X) + q, point)
        q = self.meta["conformal"]["peak_120"]
        pk = g0 + self.peak["point"].predict(X)
        out["peak"] = pk
        out["peak_lo"] = np.minimum(g0 + self.peak["q10"].predict(X) - q, pk)
        out["peak_hi"] = np.maximum(g0 + self.peak["q90"].predict(X) + q, pk)
        out["spike_prob"] = self.calibrate(self.spike.predict(X))
        out["already_high"] = g0 > 180
        out["alert"] = (~out["already_high"]) & (out["spike_prob"] >= self.threshold)
        return out
