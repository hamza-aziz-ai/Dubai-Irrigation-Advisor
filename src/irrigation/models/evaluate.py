"""
Head-to-head evaluation of depletion predictors.

Training and evaluation use DIFFERENT weather and sensor seeds. Sharing them
would let a supervised model memorize the season and report a result that
cannot survive contact with next year's weather.
"""
from __future__ import annotations

import numpy as np

from .predictors import (
    DepletionPredictor,
    GradientBoostingPredictor,
    PhysicsBalance,
    PhysicsSensorFusion,
    RandomForestPredictor,
    SensorDirect,
    XGBoostPredictor,
)
from .simulate import SeasonResult, simulate_season
from ..decision.policy import CostModel
from ..physics.crop import Crop, Soil

TRAIN_SEEDS = [11, 23, 37, 51, 67]
EVAL_WEATHER_SEED = 42
EVAL_SENSOR_SEED = 7


def build_training_set(
    crop: Crop, soil: Soil, root_depth_m: float, kc: float, days: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (features, true depletion) across several simulated seasons.

    The behaviour policy is the physics baseline: training data must come from
    a system that was operating sensibly, otherwise the model only ever sees
    states a broken controller visits.
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for seed in TRAIN_SEEDS:
        _, X, y = simulate_season(
            PhysicsBalance(crop, soil), crop, soil,
            root_depth_m=root_depth_m, kc=kc, days=days,
            weather_seed=seed, sensor_seed=seed + 100,
            collect_training_data=True,
        )
        # `simulate_season` returns None for both when collection is off. It
        # is on here, so this cannot fire - but the coupling between the flag
        # and the return lives in that function, not this one, and a silent
        # None would surface much later as an unreadable numpy error.
        if X is None or y is None:
            raise RuntimeError(
                "simulate_season returned no training data despite "
                "collect_training_data=True"
            )
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def default_predictors(crop: Crop, soil: Soil) -> list[DepletionPredictor]:
    """
    The comparison set, ordered from no learning to most learning.

    The three supervised models share a feature vector and a training set, so
    the spread between them measures the learning algorithm alone. They are
    listed after the physics and sensor baselines because the question this
    project asks is whether they earn their place against those - not which
    of them wins among themselves.
    """
    return [
        PhysicsBalance(crop, soil),
        SensorDirect(soil),
        PhysicsSensorFusion(crop, soil),
        RandomForestPredictor(),
        GradientBoostingPredictor(),
        XGBoostPredictor(),
    ]


def run_comparison(
    crop: Crop,
    soil: Soil,
    root_depth_m: float | None = None,
    kc: float | None = None,
    days: int = 120,
    cost_model: CostModel | None = None,
    predictors: list[DepletionPredictor] | None = None,
) -> list[SeasonResult]:
    root_depth_m = root_depth_m or crop.root_depth_max_m
    kc = kc if kc is not None else crop.kc_mid
    predictors = predictors or default_predictors(crop, soil)

    # Fitting is hoisted out of the simulation loop so the training set is a
    # plain pair of arrays rather than something optional that every later
    # line has to re-establish is present. It is built at most once and shared
    # by all of them, which is what makes a cost difference between the
    # supervised models attributable to the algorithm and not to the sample.
    trainable = [p for p in predictors if p.requires_training]
    if trainable:
        features, targets = build_training_set(crop, soil, root_depth_m, kc, days)
        for predictor in trainable:
            predictor.fit(features, targets)

    results: list[SeasonResult] = []

    for predictor in predictors:
        result, _, _ = simulate_season(
            predictor, crop, soil,
            root_depth_m=root_depth_m, kc=kc, days=days,
            weather_seed=EVAL_WEATHER_SEED, sensor_seed=EVAL_SENSOR_SEED,
            cost_model=cost_model,
        )
        results.append(result)

    return results


def render_table(results: list[SeasonResult]) -> str:
    header = (
        f"{'Predictor':<28} {'Cost AED':>9} {'Water mm':>9} {'Events':>7} "
        f"{'Stress d':>9} {'Severe d':>9} {'Drain mm':>9} {'RMSE mm':>8} {'Bias mm':>8}"
    )
    lines = [header, "-" * len(header)]
    best = min(r.total_cost for r in results)
    for r in results:
        s = r.summary()
        marker = "  <-- best" if abs(r.total_cost - best) < 1e-6 else ""
        lines.append(
            f"{s['predictor']:<28} {s['total_cost_aed']:>9.0f} {s['water_mm']:>9.0f} "
            f"{s['irrigation_events']:>7d} {s['stress_days']:>9d} "
            f"{s['severe_stress_days']:>9d} {s['drainage_mm']:>9.0f} "
            f"{s['depletion_rmse_mm']:>8.2f} {s['depletion_bias_mm']:>8.2f}{marker}"
        )
    return "\n".join(lines)
