"""
Sequence models over real Dubai soil moisture.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR

The rest of `models/` compares depletion predictors inside a simulator. That
comparison is honest about costs, but it has one unavoidable weakness: the
true state comes from this project's own water balance, so a model fitted to
it is partly fitted to my assumptions.

Unlike the rest of the package, this module imports torch at module level. It
is the only module that cannot work without it, and a locally defined
`nn.Module` cannot be serialized by `torch.save`, so the class lives at module
scope where it can be saved, subclassed and inspected.

This module removes that weakness for one specific question. The target here
is NASA POWER's `GWETROOT` - root-zone soil wetness produced by MERRA-2's land
surface model, from satellite and reanalysis inputs, by people with no
knowledge of this repository. Nothing about it derives from the simulator.

What it is not: an irrigated field. The Dubai grid cell is overwhelmingly bare
desert, so this series describes how real sand in this climate dries after
rain and rewets when rain comes - not how a watered turf plot behaves. The
transferable finding is the soil-atmosphere response, not a depletion
predictor that can be dropped into `evaluate.py`. Claiming otherwise would
repeat exactly the error this project was built to expose.

TWO TASKS, DELIBERATELY SEPARATED

`FORECAST` - predict tomorrow's wetness given the past, including past
wetness. Soil moisture is strongly autocorrelated, so persistence is a
formidable baseline and beating it is the real test. This is the task an
operator with a working probe faces.

`ESTIMATE` - predict today's wetness from weather alone, with no soil moisture
input at any lag. Much harder, and much closer to the operational situation
this project cares about: a site with no probe, or one whose probe has
drifted, where the water balance is all you have. The FAO-56 balance is a
direct competitor here.

Reporting only the first would flatter the models enormously. Both are run.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..climate.et0_series import et0_for_day
from ..data.nasa_power import PowerRecord, to_daily_weather

Task = Literal["forecast", "estimate"]
CellType = Literal["lstm", "gru"]

# Weather-only drivers. No soil moisture at any lag, so this list is safe for
# both tasks; the ESTIMATE task uses exactly these and nothing else.
WEATHER_FEATURES = [
    "tmax_c", "tmin_c", "dewpoint_c", "solar_mj",
    "wind_2m_ms", "rainfall_mm", "et0_mm", "doy_sin", "doy_cos",
]

# Added only for the FORECAST task.
SOIL_FEATURES = ["wetness_root", "wetness_top"]

TARGET = "wetness_root"


@dataclass(frozen=True)
class SplitYears:
    """
    Chronological split boundaries, inclusive.

    Years rather than random rows, and contiguous rather than interleaved.
    Randomly splitting a time series puts a Tuesday in train and the
    surrounding Monday and Wednesday in test; with soil moisture, whose
    autocorrelation is measured in weeks, that leaks the answer almost
    perfectly and produces a model that looks superb and forecasts nothing.

    This mirrors CLAUDE.md invariant 5 - the simulator keeps TRAIN_SEEDS and
    EVAL_WEATHER_SEED disjoint for the same reason.
    """

    train_end: int = 2016
    val_end: int = 2020
    test_end: int = 2024


@dataclass
class SequenceConfig:
    """Everything that defines a run. Defaults are the reported configuration."""

    task: Task = "forecast"
    cell: CellType = "lstm"
    lookback_days: int = 21
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8
    seed: int = 0
    splits: SplitYears = field(default_factory=SplitYears)

    @property
    def feature_names(self) -> list[str]:
        if self.task == "forecast":
            return [*WEATHER_FEATURES, *SOIL_FEATURES]
        return list(WEATHER_FEATURES)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def build_frame(records: Sequence[PowerRecord]) -> tuple[np.ndarray, np.ndarray, list[_dt.date], list[str]]:
    """
    Daily feature matrix, target vector, dates, and column names.

    ET0 is computed here rather than left to the network. It is the physically
    meaningful combination of temperature, humidity, wind and radiation, and
    it is already validated against FAO's worked examples - making a model
    rediscover it from the raw fields would waste capacity on something that
    is known exactly.
    """
    columns = [*WEATHER_FEATURES, *SOIL_FEATURES]
    rows = []
    targets = []
    dates = []

    for record in records:
        angle = 2.0 * math.pi * record.day_of_year / 365.25
        et0 = et0_for_day(to_daily_weather(record)).et0_mm_day
        values = {
            "tmax_c": record.tmax_c,
            "tmin_c": record.tmin_c,
            "dewpoint_c": record.dewpoint_c,
            "solar_mj": record.solar_mj,
            "wind_2m_ms": record.wind_2m_ms,
            "rainfall_mm": record.rainfall_mm,
            "et0_mm": et0,
            "doy_sin": math.sin(angle),
            "doy_cos": math.cos(angle),
            "wetness_root": record.wetness_root,
            "wetness_top": record.wetness_top,
        }
        rows.append([values[name] for name in columns])
        targets.append(record.wetness_root)
        dates.append(record.date)

    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32), dates, columns


def make_windows(
    features: np.ndarray,
    targets: np.ndarray,
    dates: Sequence[_dt.date],
    lookback: int,
    task: Task,
) -> tuple[np.ndarray, np.ndarray, list[_dt.date]]:
    """
    Slice into (window, target) pairs.

    FORECAST: days [i-lookback, i-1] predict day i. The window stops the day
    before the target, so tomorrow's weather is never visible - that would be
    a forecast using the future it is forecasting.

    ESTIMATE: days [i-lookback+1, i] predict day i. Today's weather is
    legitimately available; today's soil moisture is not in the feature set at
    all for this task, so there is nothing to leak.

    The returned dates are the target dates, which is what the split boundary
    must be applied to.
    """
    windows, labels, label_dates = [], [], []
    for i in range(lookback, len(features)):
        window = features[i - lookback:i] if task == "forecast" else features[i - lookback + 1:i + 1]
        windows.append(window)
        labels.append(targets[i])
        label_dates.append(dates[i])
    return np.asarray(windows), np.asarray(labels), label_dates


@dataclass
class Dataset:
    """Windowed, split, and standardized arrays ready for training."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dates_train: list[_dt.date]
    dates_val: list[_dt.date]
    dates_test: list[_dt.date]
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray


def build_dataset(records: Sequence[PowerRecord], config: SequenceConfig) -> Dataset:
    """
    Windows, chronological split, and standardization fitted on train only.

    The standardizer is the subtle leak. Fitting mean and standard deviation
    over the whole series lets the test years influence the scaling of the
    training years. The effect is small and invisible in any loss curve, and
    it is still enough to make a reported test score unreproducible on
    genuinely unseen data.
    """
    features, targets, dates, columns = build_frame(records)

    keep = [columns.index(name) for name in config.feature_names]
    features = features[:, keep]

    windows, labels, label_dates = make_windows(
        features, targets, dates, config.lookback_days, config.task
    )

    years = np.array([d.year for d in label_dates])
    train = years <= config.splits.train_end
    val = (years > config.splits.train_end) & (years <= config.splits.val_end)
    test = (years > config.splits.val_end) & (years <= config.splits.test_end)

    # Accumulated in float64 even though the windows are float32. The training
    # block is around 170,000 rows deep, and a float32 running sum over that
    # many values loses enough precision to shift the mean of a
    # small-magnitude feature - soil wetness has a standard deviation of about
    # 0.03 - by a noticeable fraction of its own spread. The arrays stay
    # float32 for the model; only the statistics are computed wide.
    flat = windows[train].reshape(-1, windows.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0

    def standardize(block: np.ndarray) -> np.ndarray:
        return ((block - mean) / std).astype(np.float32)

    def dates_where(mask: np.ndarray) -> list[_dt.date]:
        return [d for d, keep_it in zip(label_dates, mask) if keep_it]

    return Dataset(
        x_train=standardize(windows[train]), y_train=labels[train],
        x_val=standardize(windows[val]), y_val=labels[val],
        x_test=standardize(windows[test]), y_test=labels[test],
        dates_train=dates_where(train),
        dates_val=dates_where(val),
        dates_test=dates_where(test),
        feature_names=config.feature_names,
        mean=mean, std=std,
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class SoilMoistureRNN(nn.Module):
    """
    Recurrent encoder over the lookback window, last state -> scalar.

    Only the final hidden state is used. Soil moisture responds to accumulated
    conditions over weeks, so the summary of the window is the quantity of
    interest; per-timestep outputs would be predicting days that are already
    in the inputs.

    Defined at module scope, not inside a factory. A class created inside a
    function cannot be pickled, so `torch.save(model)` on it fails and the
    trained weights cannot leave the process that made them.
    """

    def __init__(self, config: SequenceConfig, n_features: int) -> None:
        super().__init__()
        cell = nn.LSTM if config.cell == "lstm" else nn.GRU
        self.rnn = cell(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def build_network(config: SequenceConfig, n_features: int) -> SoilMoistureRNN:
    """LSTM or GRU followed by a linear head, sized for this feature set."""
    return SoilMoistureRNN(config, n_features)


@dataclass
class TrainingResult:
    """A fitted model plus everything needed to report on it honestly."""

    model: SoilMoistureRNN
    config: SequenceConfig
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    predictions_test: np.ndarray
    y_test: np.ndarray
    dates_test: list[_dt.date]

    @property
    def metrics(self) -> dict[str, float]:
        return regression_metrics(self.y_test, self.predictions_test)


def train_sequence_model(dataset: Dataset, config: SequenceConfig) -> TrainingResult:
    """
    Fit with early stopping on the validation years.

    Early stopping watches the validation split and the best weights are
    restored at the end, so the reported test score comes from the model that
    generalized best rather than the one that trained longest. The test years
    are touched exactly once, after that choice is already made.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_network(config, dataset.x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    def tensors(x: np.ndarray, y: np.ndarray):
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )

    x_train, y_train = tensors(dataset.x_train, dataset.y_train)
    x_val, y_val = tensors(dataset.x_val, dataset.y_val)
    x_test, _ = tensors(dataset.x_test, dataset.y_test)

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).cpu().numpy()

    return TrainingResult(
        model=model, config=config,
        train_losses=train_losses, val_losses=val_losses, best_epoch=best_epoch,
        predictions_test=predictions, y_test=dataset.y_test,
        dates_test=dataset.dates_test,
    )


# --------------------------------------------------------------------------
# Baselines and metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, bias and R^2, all in units of soil wetness [0-1]."""
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    variance = float(np.var(y_true))
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.mean(error ** 2) / variance) if variance > 0 else float("nan"),
    }


def persistence_baseline(dataset: Dataset) -> np.ndarray:
    """
    Tomorrow equals today. The bar the FORECAST task must clear.

    Reads the last timestep's `wetness_root` back out of the standardized
    window, which is only possible because that feature exists for this task -
    hence the guard. For ESTIMATE there is no soil moisture input and no
    persistence baseline to construct.
    """
    if TARGET not in dataset.feature_names:
        raise ValueError("persistence needs soil moisture in the features (forecast task)")
    index = dataset.feature_names.index(TARGET)
    standardized = dataset.x_test[:, -1, index]
    return standardized * dataset.std[index] + dataset.mean[index]


def climatology_baseline(
    records: Sequence[PowerRecord], dataset: Dataset, config: SequenceConfig
) -> np.ndarray:
    """
    Day-of-year mean from the training years only.

    A seasonal lookup table with no weather input at all. It is the honest
    floor for the ESTIMATE task: any model that cannot beat "what is normal
    for this date" has learned nothing about this particular year.
    """
    by_doy: dict[int, list[float]] = {}
    for record in records:
        if record.date.year <= config.splits.train_end:
            by_doy.setdefault(record.day_of_year, []).append(record.wetness_root)
    means = {doy: float(np.mean(values)) for doy, values in by_doy.items()}
    overall = float(np.mean([v for values in by_doy.values() for v in values]))
    return np.array(
        [means.get(d.timetuple().tm_yday, overall) for d in dataset.dates_test],
        dtype=float,
    )
