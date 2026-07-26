"""Depletion predictors.

Every predictor answers one question: given what is known this morning, how
depleted is the root zone right now [mm]? The policy layer turns that into a
decision, and the simulator advances the true state. Keeping estimation and
decision separate is what makes it possible to attribute a cost difference to
the model rather than to the policy.

The predictors are deliberately ordered from "no learning at all" upward, so
that any claim about machine learning has to be earned against a baseline
that a competent agronomist would have built anyway.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..climate.dubai import DailyWeather
from ..physics.crop import Crop, Soil
from ..physics import soil_water


@dataclass
class Observation:
    """What is available to a predictor on the morning of day i."""

    day_index: int
    day_of_year: int
    sensor_vwc: float | None
    et0_today_mm: float
    et0_forecast_mm: float
    weather: DailyWeather
    irrigation_yesterday_mm: float
    rainfall_yesterday_mm: float
    kc: float
    root_depth_m: float


class DepletionPredictor(ABC):
    name: str
    requires_training: bool = False

    @abstractmethod
    def predict(self, obs: Observation) -> float:
        """Estimated root-zone depletion [mm]."""

    def reset(self) -> None:
        """Clear any per-season internal state."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
class PhysicsBalance(DepletionPredictor):
    """FAO-56 water balance, no sensor at all.

    The honest baseline. This is what irrigation scheduling has used since
    1998 and it needs no hardware, no training data and no inference. Any ML
    model has to beat this to justify its existence.

    Its weakness is real: errors accumulate. Nothing corrects the running
    estimate, so an unmodelled loss drifts the balance over a season.
    """

    name = "Physics (FAO-56 balance)"

    def __init__(self, crop: Crop, soil: Soil) -> None:
        self.crop, self.soil = crop, soil
        self._depletion = 0.0

    def reset(self) -> None:
        self._depletion = 0.0

    def predict(self, obs: Observation) -> float:
        state = soil_water.step(
            depletion_mm=self._depletion,
            et0_mm=obs.et0_today_mm,
            crop=self.crop, soil=self.soil,
            root_depth_m=obs.root_depth_m, kc=obs.kc,
            irrigation_mm=obs.irrigation_yesterday_mm,
            rainfall_mm=obs.rainfall_yesterday_mm,
        )
        self._depletion = state.depletion_mm
        return self._depletion


class SensorDirect(DepletionPredictor):
    """Trust the probe. This is what most "smart irrigation" products do.

    Converts measured volumetric water content straight to depletion. Exposed
    to every sensor pathology at once - offset, drift, noise, dropout - with
    nothing to check them against.
    """

    name = "Sensor only"

    def __init__(self, soil: Soil) -> None:
        self.soil = soil
        self._last = 0.0

    def reset(self) -> None:
        self._last = 0.0

    def predict(self, obs: Observation) -> float:
        if obs.sensor_vwc is None:
            return self._last              # dropout: carry forward
        deficit = max(0.0, self.soil.field_capacity - obs.sensor_vwc)
        self._last = deficit * 1000.0 * obs.root_depth_m
        return self._last


class PhysicsSensorFusion(DepletionPredictor):
    """Physics balance corrected toward the sensor.

    A fixed-gain complementary filter: physics supplies the trend, the sensor
    supplies a slow correction against accumulated error. The gain is low
    deliberately - the probe is biased and drifting, so following it closely
    would import its faults.

    This is the version an engineer reaches for before reaching for ML, and
    it is the one that usually wins.
    """

    name = "Physics + sensor fusion"

    def __init__(self, crop: Crop, soil: Soil, gain: float = 0.15) -> None:
        self.crop, self.soil, self.gain = crop, soil, gain
        self._depletion = 0.0

    def reset(self) -> None:
        self._depletion = 0.0

    def predict(self, obs: Observation) -> float:
        state = soil_water.step(
            depletion_mm=self._depletion,
            et0_mm=obs.et0_today_mm,
            crop=self.crop, soil=self.soil,
            root_depth_m=obs.root_depth_m, kc=obs.kc,
            irrigation_mm=obs.irrigation_yesterday_mm,
            rainfall_mm=obs.rainfall_yesterday_mm,
        )
        physics = state.depletion_mm

        if obs.sensor_vwc is not None:
            sensor_depletion = max(
                0.0, (self.soil.field_capacity - obs.sensor_vwc) * 1000.0 * obs.root_depth_m
            )
            physics = (1 - self.gain) * physics + self.gain * sensor_depletion

        self._depletion = max(0.0, min(physics, self.soil.total_available_water_mm(obs.root_depth_m)))
        return self._depletion


# --------------------------------------------------------------------------
FEATURE_NAMES = [
    "sensor_vwc", "sensor_available", "et0_today", "et0_forecast",
    "tmax", "tmin", "rh_min", "wind", "irrigation_yesterday",
    "rainfall_yesterday", "kc", "doy_sin", "doy_cos",
    "sensor_lag1", "sensor_lag2", "et0_3day_sum",
]


def build_features(obs: Observation, history: dict[str, Any]) -> np.ndarray:
    sensor = obs.sensor_vwc if obs.sensor_vwc is not None else history.get("last_sensor", 0.10)
    doy_angle = 2 * np.pi * obs.day_of_year / 365.0
    return np.array([
        sensor,
        1.0 if obs.sensor_vwc is not None else 0.0,
        obs.et0_today_mm,
        obs.et0_forecast_mm,
        obs.weather.tmax_c,
        obs.weather.tmin_c,
        obs.weather.rh_min_pct,
        obs.weather.wind_2m_ms,
        obs.irrigation_yesterday_mm,
        obs.rainfall_yesterday_mm,
        obs.kc,
        np.sin(doy_angle),
        np.cos(doy_angle),
        history.get("sensor_lag1", sensor),
        history.get("sensor_lag2", sensor),
        history.get("et0_3day_sum", obs.et0_today_mm * 3),
    ], dtype=float)


class TabularModelPredictor(DepletionPredictor):
    """Base for supervised models over the engineered feature vector.

    Everything that decides *what the model is asked* lives here - the feature
    vector, the lag bookkeeping, the non-negativity clamp on the output. Only
    the estimator differs between subclasses, which is the point: a difference
    in cost between them is then attributable to the learning algorithm and
    not to one of them quietly getting better inputs.

    All of them see an `Observation` and nothing else, so the guarantee in
    CLAUDE.md invariant 4 holds for the whole family at once.
    """

    requires_training = True

    def __init__(self) -> None:
        self._model = self._build_model()
        self._history: dict[str, Any] = {}
        self._fitted = False

    def _build_model(self) -> Any:
        """Return an unfitted estimator with a scikit-learn fit/predict API."""
        raise NotImplementedError

    def reset(self) -> None:
        self._history = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, obs: Observation) -> float:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} used before fit()")
        x = build_features(obs, self._history)
        self._update_history(obs)
        # Clamped at zero: depletion below field capacity is not a physical
        # state, and a regressor has no constraint preventing it from
        # predicting one.
        return float(max(0.0, self._model.predict(x.reshape(1, -1))[0]))

    def _update_history(self, obs: Observation) -> None:
        if obs.sensor_vwc is not None:
            self._history["sensor_lag2"] = self._history.get("sensor_lag1", obs.sensor_vwc)
            self._history["sensor_lag1"] = self._history.get("last_sensor", obs.sensor_vwc)
            self._history["last_sensor"] = obs.sensor_vwc
        window = self._history.setdefault("et0_window", [])
        window.append(obs.et0_today_mm)
        self._history["et0_3day_sum"] = float(sum(window[-3:]))


class GradientBoostingPredictor(TabularModelPredictor):
    """Gradient-boosted trees on engineered features.

    Given the same information as the other predictors, plus lags. Trained on
    one simulated season, evaluated on a different one with a different seed -
    training and testing on the same weather would flatter it enormously.
    """

    name = "Gradient boosting"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        super().__init__()

    def _build_model(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6, random_state=self._seed
        )


class RandomForestPredictor(TabularModelPredictor):
    """Bagged trees. Named in the brief, and a useful contrast to boosting.

    Boosting fits residuals sequentially and can extrapolate a trend; a forest
    averages independent trees and cannot predict outside the range of its
    training targets at all. That difference is worth having in the comparison
    precisely because it changes the *direction* of the errors, and direction
    is what the asymmetric cost model charges for.
    """

    name = "Random forest"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        super().__init__()

    def _build_model(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=self._seed, n_jobs=-1,
        )


class XGBoostPredictor(TabularModelPredictor):
    """XGBoost, named explicitly in the brief.

    Hyperparameters are deliberately close to the scikit-learn booster's so
    the comparison isolates the implementation rather than the tuning budget.
    Both are gradient boosting; if they land far apart on cost, that is a
    finding about variance between runs, not about one library being better.
    """

    name = "XGBoost"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        super().__init__()

    def _build_model(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300, learning_rate=0.06, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            random_state=self._seed, n_jobs=-1, tree_method="hist",
        )
