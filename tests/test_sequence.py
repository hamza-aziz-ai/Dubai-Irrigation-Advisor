"""Tests for the sequence-model data pipeline.

Almost all of these are leakage tests, and that is deliberate. A sequence
model on a strongly autocorrelated target will happily report a superb score
obtained entirely from information it should not have had, and no loss curve,
metric or plot will look wrong. The pipeline is where that gets decided, so
the pipeline is where the assertions go.

Training itself is exercised once, tiny, purely to prove the loop runs and is
deterministic. Model quality is the notebook's job, not the suite's.
"""
from __future__ import annotations

import numpy as np
import pytest

from irrigation.data.nasa_power import load_records
from irrigation.models import sequence as S


@pytest.fixture(scope="module")
def records():
    return load_records()


@pytest.fixture(scope="module")
def forecast_config():
    return S.SequenceConfig(task="forecast", lookback_days=21)


@pytest.fixture(scope="module")
def estimate_config():
    return S.SequenceConfig(task="estimate", lookback_days=21)


@pytest.fixture(scope="module")
def forecast_dataset(records, forecast_config):
    return S.build_dataset(records, forecast_config)


@pytest.fixture(scope="module")
def estimate_dataset(records, estimate_config):
    return S.build_dataset(records, estimate_config)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def test_frame_has_one_row_per_record(records):
    features, targets, dates, columns = S.build_frame(records)
    assert features.shape == (len(records), len(columns))
    assert targets.shape == (len(records),)
    assert len(dates) == len(records)


def test_et0_is_computed_not_learned(records):
    """ET0 is a validated physical quantity, supplied rather than rediscovered."""
    features, _, _, columns = S.build_frame(records[:400])
    et0 = features[:, columns.index("et0_mm")]
    assert et0.min() > 0.5
    assert et0.max() < 15.0


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def test_estimate_task_never_sees_soil_moisture(estimate_dataset):
    """The whole point of the ESTIMATE task.

    If any soil moisture column survives into the features, the task silently
    becomes the much easier FORECAST task and its headline R^2 is meaningless.
    """
    for name in S.SOIL_FEATURES:
        assert name not in estimate_dataset.feature_names


def test_forecast_window_stops_before_the_target_day(records, forecast_config):
    """A forecast that can see the day it is forecasting is not a forecast.

    Checked structurally: the last row of window `i` must equal the raw
    feature row for day `i-1`, never day `i`.
    """
    features, targets, dates, _ = S.build_frame(records[:200])
    windows, labels, label_dates = S.make_windows(
        features, targets, dates, lookback=21, task="forecast"
    )
    assert np.allclose(windows[0][-1], features[20])
    assert label_dates[0] == dates[21]


def test_estimate_window_includes_the_target_day(records):
    """Today's weather is legitimately available when estimating today."""
    features, targets, dates, _ = S.build_frame(records[:200])
    windows, labels, label_dates = S.make_windows(
        features, targets, dates, lookback=21, task="estimate"
    )
    assert np.allclose(windows[0][-1], features[21])
    assert label_dates[0] == dates[21]


def test_splits_are_chronological_and_disjoint(forecast_dataset):
    """Random splitting a time series leaks almost perfectly.

    Soil moisture autocorrelation runs to weeks, so a shuffled split puts the
    day before and the day after a test sample into training. Contiguity is
    the only defence, and it is asserted rather than assumed.
    """
    train, val, test = (
        forecast_dataset.dates_train,
        forecast_dataset.dates_val,
        forecast_dataset.dates_test,
    )
    assert max(train) < min(val)
    assert max(val) < min(test)
    assert not set(train) & set(val)
    assert not set(val) & set(test)


def test_split_boundaries_match_the_configured_years(forecast_dataset, forecast_config):
    splits = forecast_config.splits
    assert max(forecast_dataset.dates_train).year == splits.train_end
    assert max(forecast_dataset.dates_val).year == splits.val_end
    assert max(forecast_dataset.dates_test).year == splits.test_end


def test_standardizer_is_fitted_on_training_data_only(records, forecast_dataset, forecast_config):
    """The invisible leak.

    Fitting the scaler over the full series lets the test years shift the
    training distribution. The effect is small, shows up nowhere, and makes
    the reported score irreproducible on genuinely unseen data.

    Asserted against the scaler's own statistics rather than by checking that
    the standardized training block has zero mean. That round-trip is true but
    untestable at useful precision: the block is float32 and about 170,000
    rows deep, and for soil moisture - small magnitude, small spread - the
    accumulated rounding divided by a std of ~0.02 is larger than any
    tolerance that would still catch a real leak.
    """
    features, targets, dates, columns = S.build_frame(records)
    keep = [columns.index(name) for name in forecast_config.feature_names]
    windows, _, label_dates = S.make_windows(
        features[:, keep], targets, dates,
        forecast_config.lookback_days, forecast_config.task,
    )
    years = np.array([d.year for d in label_dates])

    train_only = windows[years <= forecast_config.splits.train_end]
    expected = train_only.reshape(-1, train_only.shape[-1]).astype(np.float64).mean(axis=0)

    # Compared in units of each feature's own spread. A relative tolerance is
    # meaningless here: `doy_sin` averages to about -1e-05 over whole years,
    # so float32 against float64 differs by orders of magnitude in relative
    # terms while being irrelevant in the units the scaler actually divides by.
    difference = np.abs(forecast_dataset.mean - expected)
    assert np.all(difference < 1e-3 * forecast_dataset.std)


def test_scaler_ignores_a_regime_shift_confined_to_the_test_years(records, forecast_config):
    """Proves the previous test can actually fail.

    On the real record the training-years mean and the whole-series mean agree
    to four significant figures, because Dubai's climate is stationary over
    thirty years. So a scaler that wrongly saw the test years would produce
    almost identical statistics and no assertion on real data could tell the
    difference.

    Here the test years are shifted by a large, obviously artificial offset.
    A correctly fitted scaler is unmoved by it; one fitted on everything is
    dragged upward and fails immediately.
    """
    shifted = [
        r if r.date.year <= forecast_config.splits.val_end
        else type(r)(**{**r.__dict__, "tmax_c": r.tmax_c + 100.0})
        for r in records
    ]

    baseline = S.build_dataset(records, forecast_config)
    perturbed = S.build_dataset(shifted, forecast_config)

    index = perturbed.feature_names.index("tmax_c")
    assert perturbed.mean[index] == pytest.approx(baseline.mean[index], rel=1e-6)


def test_no_nan_or_inf_reaches_the_model(forecast_dataset, estimate_dataset):
    for dataset in (forecast_dataset, estimate_dataset):
        for block in (dataset.x_train, dataset.x_val, dataset.x_test):
            assert np.isfinite(block).all()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_persistence_recovers_the_previous_days_wetness(forecast_dataset):
    """De-standardization must invert exactly, or the baseline is handicapped.

    A baseline that is accidentally weakened is worse than no baseline: it
    makes the model look good and gives no warning.
    """
    predictions = S.persistence_baseline(forecast_dataset)
    assert predictions.min() > 0.0
    assert predictions.max() < 1.0
    metrics = S.regression_metrics(forecast_dataset.y_test, predictions)
    assert metrics["r2"] > 0.9


def test_persistence_is_unavailable_without_soil_history(estimate_dataset):
    with pytest.raises(ValueError):
        S.persistence_baseline(estimate_dataset)


def test_climatology_uses_only_training_years(records, forecast_dataset, forecast_config):
    """A day-of-year table built over all years would encode the test years."""
    predictions = S.climatology_baseline(records, forecast_dataset, forecast_config)
    assert len(predictions) == len(forecast_dataset.y_test)

    train_only = [r for r in records if r.date.year <= forecast_config.splits.train_end]
    assert predictions.min() >= min(r.wetness_root for r in train_only)
    assert predictions.max() <= max(r.wetness_root for r in train_only)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_metrics_are_correct_on_a_known_case():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5])
    metrics = S.regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["bias"] == pytest.approx(0.5)


def test_perfect_prediction_scores_r2_of_one():
    y = np.array([0.1, 0.2, 0.3])
    assert S.regression_metrics(y, y)["r2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_training_runs_and_is_deterministic(records, cell: S.CellType):
    """Tiny by design - this asserts the loop works, not that it works well."""
    pytest.importorskip("torch")
    config = S.SequenceConfig(
        task="forecast", cell=cell, lookback_days=7,
        hidden_size=8, num_layers=1, max_epochs=2, patience=2, seed=0,
    )
    dataset = S.build_dataset(records, config)

    first = S.train_sequence_model(dataset, config)
    second = S.train_sequence_model(dataset, config)

    assert first.predictions_test.shape == dataset.y_test.shape
    assert np.allclose(first.predictions_test, second.predictions_test, atol=1e-5)
    assert np.isfinite(first.metrics["rmse"])
