"""
Simulated soil-moisture probe.

Models the failure modes that actually break field deployments, because a
pipeline validated only on clean data will meet none of them:

  * calibration offset  - capacitance probes ship uncalibrated for the local soil
  * drift               - slow bias growth from salinity and electrode ageing
  * measurement noise   - thermal and electronic
  * dropout             - power, radio or waterlogging outages
  * quantization        - a 12-bit ADC does not produce a real number

Salinity drift is not incidental in the Gulf: irrigation water is commonly
desalinated or brackish, salts accumulate in the root zone, and capacitance
probes read that as moisture. A model trained on drift-free data will silently
under-irrigate as the season progresses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorConfig:
    calibration_offset: float = 0.015     # m3/m3, systematic
    drift_per_day: float = 0.00012        # m3/m3/day, salinity accumulation
    noise_std: float = 0.006              # m3/m3
    dropout_probability: float = 0.02
    adc_bits: int = 12
    range_max: float = 0.50               # m3/m3 at full scale


@dataclass(frozen=True)
class SensorReading:
    day_index: int
    true_vwc: float
    measured_vwc: float | None            # None when the sensor dropped out

    @property
    def available(self) -> bool:
        return self.measured_vwc is not None


class SoilMoistureSensor:
    def __init__(self, config: SensorConfig | None = None, seed: int = 7) -> None:
        self.config = config or SensorConfig()
        self._rng = np.random.default_rng(seed)

    def read(self, day_index: int, true_vwc: float) -> SensorReading:
        c = self.config
        if self._rng.random() < c.dropout_probability:
            return SensorReading(day_index, true_vwc, None)

        value = (
            true_vwc
            + c.calibration_offset
            + c.drift_per_day * day_index
            + self._rng.normal(0.0, c.noise_std)
        )
        value = float(np.clip(value, 0.0, c.range_max))

        levels = 2 ** c.adc_bits
        value = round(value / c.range_max * (levels - 1)) / (levels - 1) * c.range_max
        return SensorReading(day_index, true_vwc, value)

    def read_series(self, true_series: list[float]) -> list[SensorReading]:
        return [self.read(i, v) for i, v in enumerate(true_series)]


def interpolate_dropouts(readings: list[SensorReading]) -> list[float]:
    """
    Fill gaps by linear interpolation, holding at the edges.

    A gap must be filled, not skipped: irrigation has to be decided every day
    whether the probe reported. Interpolating is honest about the
    uncertainty in a way that carrying the last value forward is not - a held
    value looks like a real measurement to everything downstream.
    """
    values: list[float | None] = [r.measured_vwc for r in readings]

    # Keyed by index rather than a list of indices into `values`. Both express
    # the same thing, but this one carries the guarantee in the type: every
    # value in `known` is a float, so no lookup below needs a reader - or a
    # type checker - to reconstruct why `values[known[0]]` cannot be None.
    known: dict[int, float] = {
        i: value for i, value in enumerate(values) if value is not None
    }
    if not known:
        raise ValueError("Sensor produced no readings at all - cannot interpolate")

    reported = sorted(known)
    first, last = reported[0], reported[-1]

    out: list[float] = []
    for i, value in enumerate(values):
        if value is not None:
            out.append(value)
        elif i < first:
            out.append(known[first])
        elif i > last:
            out.append(known[last])
        else:
            lo = max(k for k in reported if k < i)
            hi = min(k for k in reported if k > i)
            weight = (i - lo) / (hi - lo)
            out.append(known[lo] * (1 - weight) + known[hi] * weight)
    return out
