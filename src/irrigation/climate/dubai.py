"""
Dubai climate baseline and a stochastic weather generator.

WHY SYNTHETIC DATA IS THE RIGHT ANSWER HERE, NOT A COMPROMISE

The brief this project responds to pairs an AI/ML workstream with a separate
IoT workstream building the ESP32 sensor rig. The modelling cannot wait for
the hardware, and a sequence model trained on nothing is not a modelling
problem - it is an impossibility.

So the pipeline is developed against a physically-grounded simulator instead:
climate normals drive FAO-56 ET0, ET0 drives a real soil water balance, and
the balance drives a simulated probe with realistic noise, drift and dropout.
When the hardware lands, the simulator is replaced by a data source behind
the same interface and nothing above it changes.

The numbers below are approximate published climate normals for Dubai
(~25.25 N, 55.33 E, near sea level), used to place the simulation in the
right regime - a hyper-arid Gulf summer. They are not a substitute for
station data. In production this module is replaced by a weather API client;
the interface is `DailyWeather`, and that is the only contract that matters.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..physics.penman_monteith import wind_speed_at_2m

DUBAI_LATITUDE = 25.25
DUBAI_LONGITUDE = 55.33
DUBAI_ALTITUDE_M = 5.0

# index 0 = January. Approximate normals; see module docstring on provenance.
TMAX_C = [24.0, 25.4, 28.2, 32.9, 37.6, 39.5, 40.8, 41.3, 38.9, 35.4, 30.5, 26.2]
TMIN_C = [14.3, 15.4, 17.6, 20.9, 24.5, 27.2, 29.6, 30.0, 27.3, 23.7, 19.9, 16.3]
RH_MAX_PCT = [85, 84, 82, 78, 75, 76, 78, 80, 81, 82, 83, 85]
RH_MIN_PCT = [45, 43, 39, 33, 28, 30, 34, 37, 38, 39, 42, 45]
# Wind normals are reported at the standard meteorological height of 10 m,
# NOT the 2 m the Penman-Monteith equation requires. The height is carried
# explicitly here and converted at point of use (FAO-56 Eq. 47) rather than
# being pre-baked into the constants: a bare list of numbers with no stated
# height is exactly how this error gets reintroduced later.
#
# Feeding 10 m wind straight into ET0 inflated annual reference ET by ~9%
# (2446 mm against a published UAE range of roughly 2000-2200 mm). The
# seasonal shape looked entirely correct, which is why only a magnitude
# check against published climatology caught it.
WIND_10M_MS = [3.4, 3.7, 3.9, 3.8, 3.9, 4.1, 3.8, 3.5, 3.3, 3.2, 3.2, 3.3]
WIND_MEASUREMENT_HEIGHT_M = 10.0
SUNSHINE_H = [8.6, 8.8, 8.7, 10.0, 11.3, 11.4, 10.7, 10.6, 10.3, 10.2, 9.8, 8.7]
RAIN_MM_MONTH = [18.8, 25.0, 22.1, 7.2, 0.4, 0.0, 0.8, 0.0, 0.0, 1.1, 2.7, 16.2]

MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


@dataclass(frozen=True)
class DailyWeather:
    """
    One day of weather. The interface a real data source must satisfy.

    The first eight fields are the minimum a station must provide. The
    trailing optional fields carry *measured* quantities that would otherwise
    have to be estimated, and they exist for the same reason the wind
    measurement height does (see the module docstring): a value and the way it
    was obtained are not separable without losing accuracy you cannot recover.

    FAO-56 ranks the ways of obtaining each input. For actual vapour pressure
    the order is dewpoint (Eq. 14) > RHmax/RHmin (Eq. 17) > RHmean (Eq. 19).
    For solar radiation it is measured > sunshine hours (Eq. 35) > temperature
    range (Eq. 50). When a source supplies the better input, discarding it to
    fit a narrower struct would silently downgrade every ET0 downstream, so
    the struct widens instead and `et0_for_day` picks the best available path.

    Attributes:
        day_of_year: Day number, 1-366. Drives the astronomical terms.
        tmax_c: Daily maximum air temperature at 2 m [C].
        tmin_c: Daily minimum air temperature at 2 m [C].
        rh_max_pct: Daily maximum relative humidity [%].
        rh_min_pct: Daily minimum relative humidity [%].
        wind_2m_ms: Wind speed *already reduced to 2 m* [m/s]. Converting at
            the point of construction is deliberate; see WIND_10M_MS above.
        sunshine_hours: Bright sunshine duration [h]. Used for Rs only when
            `solar_radiation_mj` is absent.
        rainfall_mm: Precipitation [mm].
        date: Calendar date, when the source is a real record rather than a
            generated day. Required for chronological train/test splits -
            `day_of_year` alone cannot distinguish 2015 from 2024.
        solar_radiation_mj: Measured incoming shortwave radiation
            [MJ m-2 day-1]. Preferred over the sunshine-hours estimate.
        dewpoint_c: Measured dewpoint temperature [C]. Preferred over both
            relative-humidity routes to ea.
    """

    day_of_year: int
    tmax_c: float
    tmin_c: float
    rh_max_pct: float
    rh_min_pct: float
    wind_2m_ms: float
    sunshine_hours: float
    rainfall_mm: float
    date: _dt.date | None = None
    solar_radiation_mj: float | None = None
    dewpoint_c: float | None = None

    @property
    def tmean_c(self) -> float:
        return (self.tmax_c + self.tmin_c) / 2.0


def month_of_year(day_of_year: int) -> int:
    """0-indexed month containing a given day of year."""
    for i in range(11, -1, -1):
        if day_of_year >= MONTH_STARTS[i]:
            return i
    return 0


def interpolate_monthly(values: Sequence[float], day_of_year: int) -> float:
    """
    Smooth monthly normals across the year.

    Stepping between monthly values would create artificial discontinuities on
    the first of each month, which a sequence model would happily learn as
    signal. Interpolating on a circular year avoids inventing that structure.

    Takes a Sequence rather than a list because `list` is invariant: the
    humidity normals above are whole numbers and therefore `list[int]`, which
    is not a `list[float]` however convertible the elements are. Sequence is
    covariant, and this function only ever reads.
    """
    pos = (day_of_year - 15.0) / 365.0 * 12.0
    lo = math.floor(pos) % 12
    hi = (lo + 1) % 12
    frac = pos - math.floor(pos)
    return values[lo] * (1.0 - frac) + values[hi] * frac


class DubaiWeatherGenerator:
    """
    Stochastic daily weather consistent with Dubai's climate normals.

    Deliberately simple and fully seeded: reproducibility matters more than
    meteorological sophistication, because the point of this data is to
    exercise the pipeline, not to forecast weather.
    """

    def __init__(self, seed: int = 42) -> None:
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def day(self, day_of_year: int) -> DailyWeather:
        rng = self._rng
        tmax = interpolate_monthly(TMAX_C, day_of_year) + rng.normal(0, 2.0)
        tmin = interpolate_monthly(TMIN_C, day_of_year) + rng.normal(0, 1.5)
        tmin = min(tmin, tmax - 3.0)      # keep the diurnal range physical

        rh_max = float(min(100.0, max(20.0,
            interpolate_monthly(RH_MAX_PCT, day_of_year) + rng.normal(0, 5.0))))
        rh_min = float(min(rh_max - 2.0, max(5.0,
            interpolate_monthly(RH_MIN_PCT, day_of_year) + rng.normal(0, 6.0))))

        wind_10m = max(0.5, interpolate_monthly(WIND_10M_MS, day_of_year)
                       + rng.normal(0, 1.0))
        wind = float(wind_speed_at_2m(wind_10m, WIND_MEASUREMENT_HEIGHT_M))
        sun = float(max(0.0, min(13.0, interpolate_monthly(SUNSHINE_H, day_of_year)
                                 + rng.normal(0, 1.2))))

        # Rainfall: rare, and intense when it happens. A Gaussian would be the
        # wrong shape entirely - most days are exactly zero.
        monthly = interpolate_monthly(RAIN_MM_MONTH, day_of_year)
        rain = 0.0
        if monthly > 0.1:
            p_wet = min(0.25, monthly / 120.0)
            if rng.random() < p_wet:
                rain = float(rng.exponential(monthly / max(p_wet * 30.0, 1e-6)))

        return DailyWeather(
            day_of_year=day_of_year,
            tmax_c=float(tmax), tmin_c=float(tmin),
            rh_max_pct=rh_max, rh_min_pct=rh_min,
            wind_2m_ms=wind, sunshine_hours=sun, rainfall_mm=rain,
        )

    def year(self, start_day: int = 1, days: int = 365) -> list[DailyWeather]:
        return [self.day(((start_day + i - 1) % 365) + 1) for i in range(days)]


def normals_day(day_of_year: int) -> DailyWeather:
    """
    Deterministic climatological day - no stochastic component.

    Used for the ET0 climatology and for any test that must not depend on a
    random seed.
    """
    return DailyWeather(
        day_of_year=day_of_year,
        tmax_c=interpolate_monthly(TMAX_C, day_of_year),
        tmin_c=interpolate_monthly(TMIN_C, day_of_year),
        rh_max_pct=interpolate_monthly(RH_MAX_PCT, day_of_year),
        rh_min_pct=interpolate_monthly(RH_MIN_PCT, day_of_year),
        wind_2m_ms=wind_speed_at_2m(
            interpolate_monthly(WIND_10M_MS, day_of_year), WIND_MEASUREMENT_HEIGHT_M
        ),
        sunshine_hours=interpolate_monthly(SUNSHINE_H, day_of_year),
        rainfall_mm=0.0,
    )
