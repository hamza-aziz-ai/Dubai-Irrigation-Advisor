"""
Season simulator and evaluation harness.

The simulator holds the *true* soil water state. A predictor only ever sees
what a field deployment would see: a noisy probe reading, weather, and its own
history. That separation is the whole reason the comparison is meaningful -
no predictor can accidentally read the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..climate.dubai import DubaiWeatherGenerator
from ..climate.et0_series import et0_for_day
from ..climate.sensor import SensorConfig, SoilMoistureSensor
from ..decision.policy import CostModel, decide
from ..physics import soil_water
from ..physics.crop import Crop, Soil
from .predictors import DepletionPredictor, Observation, build_features


@dataclass
class DayRecord:
    day_index: int
    day_of_year: int
    et0_mm: float
    true_depletion_mm: float
    predicted_depletion_mm: float
    applied_mm: float
    action: str
    ks: float
    stressed: bool
    drainage_mm: float
    costs: dict[str, float]
    reason: str


@dataclass
class SeasonResult:
    predictor_name: str
    records: list[DayRecord] = field(default_factory=list)

    # ---- decision-quality metrics (what actually matters) ----
    @property
    def total_cost(self) -> float:
        return sum(r.costs["total_cost"] for r in self.records)

    @property
    def water_applied_mm(self) -> float:
        return sum(r.applied_mm for r in self.records)

    @property
    def stress_days(self) -> int:
        return sum(1 for r in self.records if r.stressed)

    @property
    def severe_stress_days(self) -> int:
        return sum(1 for r in self.records if r.ks < 0.6)

    @property
    def drainage_mm(self) -> float:
        return sum(r.drainage_mm for r in self.records)

    @property
    def irrigation_events(self) -> int:
        return sum(1 for r in self.records if r.applied_mm > 0)

    # ---- prediction-quality metrics (diagnostic only) ----
    @property
    def depletion_rmse(self) -> float:
        e = [r.predicted_depletion_mm - r.true_depletion_mm for r in self.records]
        return float(np.sqrt(np.mean(np.square(e)))) if e else 0.0

    @property
    def depletion_bias(self) -> float:
        e = [r.predicted_depletion_mm - r.true_depletion_mm for r in self.records]
        return float(np.mean(e)) if e else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor_name,
            "total_cost_aed": round(self.total_cost, 1),
            "water_mm": round(self.water_applied_mm, 1),
            "irrigation_events": self.irrigation_events,
            "stress_days": self.stress_days,
            "severe_stress_days": self.severe_stress_days,
            "drainage_mm": round(self.drainage_mm, 1),
            "depletion_rmse_mm": round(self.depletion_rmse, 2),
            "depletion_bias_mm": round(self.depletion_bias, 2),
        }


def simulate_season(
    predictor: DepletionPredictor,
    crop: Crop,
    soil: Soil,
    *,
    start_day: int = 121,          # 1 May - into the Gulf summer
    days: int = 120,
    root_depth_m: float | None = None,
    kc: float | None = None,
    weather_seed: int = 42,
    sensor_seed: int = 7,
    sensor_config: SensorConfig | None = None,
    cost_model: CostModel | None = None,
    trigger_fraction: float = 0.85,
    collect_training_data: bool = False,
) -> tuple[SeasonResult, np.ndarray | None, np.ndarray | None]:
    """
    Run one irrigation season.

    Returns the result and, optionally, (X, y) for training a supervised
    predictor. Training labels are the TRUE depletion - available only in
    simulation, which is exactly why the simulator has to be physically
    honest for any of this to transfer.
    """
    cost_model = cost_model or CostModel()
    root_depth_m = root_depth_m or crop.root_depth_max_m
    kc = kc if kc is not None else crop.kc_mid

    weather_gen = DubaiWeatherGenerator(seed=weather_seed)
    sensor = SoilMoistureSensor(sensor_config, seed=sensor_seed)
    predictor.reset()

    weather = [weather_gen.day(((start_day + i - 1) % 365) + 1) for i in range(days + 1)]
    et0 = [et0_for_day(w).et0_mm_day for w in weather]

    result = SeasonResult(predictor_name=predictor.name)
    true_depletion = 0.0
    irrigation_yesterday = 0.0
    rainfall_yesterday = 0.0
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    history: dict[str, Any] = {}

    for i in range(days):
        w = weather[i]
        true_vwc = max(
            soil.wilting_point,
            soil.field_capacity - true_depletion / (1000.0 * root_depth_m),
        )
        reading = sensor.read(i, true_vwc)

        obs = Observation(
            day_index=i,
            day_of_year=w.day_of_year,
            sensor_vwc=reading.measured_vwc,
            et0_today_mm=et0[i],
            et0_forecast_mm=et0[i + 1],
            weather=w,
            irrigation_yesterday_mm=irrigation_yesterday,
            rainfall_yesterday_mm=rainfall_yesterday,
            kc=kc,
            root_depth_m=root_depth_m,
        )

        if collect_training_data:
            feature_rows.append(build_features(obs, history))
            labels.append(true_depletion)
            if reading.measured_vwc is not None:
                history["sensor_lag2"] = history.get("sensor_lag1", reading.measured_vwc)
                history["sensor_lag1"] = history.get("last_sensor", reading.measured_vwc)
                history["last_sensor"] = reading.measured_vwc
            win = history.setdefault("et0_window", [])
            win.append(et0[i])
            history["et0_3day_sum"] = float(sum(win[-3:]))

        predicted = predictor.predict(obs)

        decision = decide(
            predicted_depletion_mm=predicted,
            et0_forecast_mm=et0[i + 1],
            crop=crop, soil=soil,
            root_depth_m=root_depth_m, kc=kc,
            trigger_fraction=trigger_fraction,
        )

        # Advance the TRUE state with the decision that was actually made.
        state = soil_water.step(
            depletion_mm=true_depletion,
            et0_mm=et0[i],
            crop=crop, soil=soil,
            root_depth_m=root_depth_m, kc=kc,
            irrigation_mm=decision.depth_mm,
            rainfall_mm=w.rainfall_mm,
        )

        costs = cost_model.evaluate_day(
            applied_mm=decision.depth_mm,
            depletion_mm=state.depletion_mm,
            taw_mm=state.taw_mm,
            raw_mm=state.raw_mm,
        )

        result.records.append(DayRecord(
            day_index=i, day_of_year=w.day_of_year, et0_mm=et0[i],
            true_depletion_mm=state.depletion_mm,
            predicted_depletion_mm=predicted,
            applied_mm=decision.depth_mm, action=decision.action,
            ks=state.ks, stressed=state.stressed,
            drainage_mm=state.drainage_mm, costs=costs, reason=decision.reason,
        ))

        true_depletion = state.depletion_mm
        irrigation_yesterday = decision.depth_mm
        rainfall_yesterday = w.rainfall_mm

    if collect_training_data:
        return result, np.array(feature_rows), np.array(labels)
    return result, None, None
