"""Tests for the NASA POWER data layer and the real-data ET0 climatology.

Two jobs here, and the second is the interesting one.

The first is ordinary: the cache parses, the units are what the metadata says,
the conversion to `DailyWeather` is lossless where it claims to be.

The second is a genuine external check on the physics. Until now the FAO-56
implementation was validated against FAO's own worked examples - necessary,
but those examples are Bangkok, Brussels and Lyon. Nothing established that
the stack produces sane numbers for a hyper-arid Gulf summer.

Driving it with 30 years of independent NASA observations does establish that,
and it cross-validates the synthetic generator at the same time: the two were
built from unrelated inputs (published monthly normals versus satellite and
reanalysis daily fields) and neither was tuned to match the other. If they
agree, that is evidence. These tolerances are therefore deliberately tight
enough to fail if either side drifts.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

import pytest

from irrigation.climate.dubai import normals_day
from irrigation.climate.et0_series import et0_for_day
from irrigation.data import nasa_power
from irrigation.physics.penman_monteith import saturation_vapour_pressure


@pytest.fixture(scope="module")
def records() -> list[nasa_power.PowerRecord]:
    return nasa_power.load_records()


@pytest.fixture(scope="module")
def weather():
    return nasa_power.load_weather()


@pytest.fixture(scope="module")
def et0_series(weather) -> list[float]:
    return [et0_for_day(day).et0_mm_day for day in weather]


def _date_of(day) -> dt.date:
    """`DailyWeather.date` is optional in general - a generated day has none.

    Every day loaded from NASA POWER carries one, so the assertion states that
    in a single place rather than leaving each caller to assume it.
    """
    date = day.date
    assert date is not None, "NASA POWER days always carry a calendar date"
    return date


# --------------------------------------------------------------------------
# The cache itself
# --------------------------------------------------------------------------
def test_cache_covers_thirty_complete_years(records):
    assert records[0].date == dt.date(1995, 1, 1)
    assert records[-1].date == dt.date(2024, 12, 31)
    # 30 years with 7 leap days between 1996 and 2024 inclusive.
    assert len(records) == 30 * 365 + 8


def test_no_gaps_in_the_series(records):
    """Contiguity matters specifically because sequence models assume it.

    A dropped day inside a 14-day input window silently shifts the lag
    structure by one, which no loss curve would reveal.
    """
    for previous, current in zip(records, records[1:]):
        assert current.date - previous.date == dt.timedelta(days=1)


def test_fill_values_are_absent(records):
    """POWER's -999 sentinel is a float, not a NaN - it must be filtered."""
    for record in records:
        assert record.tmax_c > -50.0
        assert record.solar_mj > 0.0
        assert not math.isnan(record.wetness_root)


def test_temperature_extremes_are_ordered(records):
    for record in records:
        assert record.tmin_c <= record.tmean_c <= record.tmax_c


def test_wind_is_slower_near_the_ground(records):
    """A logarithmic wind profile means 2 m wind is always below 10 m wind.

    This is the invariant the original wind-height bug violated. Asserting it
    on the real data confirms POWER's WS2M is genuinely at 2 m and does not
    need the Eq. 47 conversion applied a second time.
    """
    for record in records:
        assert record.wind_2m_ms < record.wind_10m_ms


def test_soil_wetness_is_a_saturation_fraction(records):
    for record in records:
        assert 0.0 <= record.wetness_top <= 1.0
        assert 0.0 <= record.wetness_root <= 1.0
        assert 0.0 <= record.wetness_prof <= 1.0


def test_radiation_is_in_megajoules_not_kilowatt_hours(records):
    """Guards the unit trap documented in POWER_PARAMETERS.

    The same POWER parameter is served in kWh m-2 day-1 to the RE community.
    That is 3.6x smaller, and it would pass every other test in this file
    while making ET0 wrong by roughly half. Dubai's clear-sky daily total sits
    around 20-30 MJ m-2; in kWh it would be 5-8.
    """
    annual_mean = sum(r.solar_mj for r in records) / len(records)
    assert 15.0 < annual_mean < 30.0


# --------------------------------------------------------------------------
# Conversion to the weather contract
# --------------------------------------------------------------------------
def test_measured_fields_survive_conversion(records):
    record = records[0]
    day = nasa_power.to_daily_weather(record)
    assert day.solar_radiation_mj == record.solar_mj
    assert day.dewpoint_c == record.dewpoint_c
    assert day.date == record.date


def test_sunshine_hours_is_nan_when_unmeasured(records):
    """Deliberately unusable rather than plausible.

    POWER reports no sunshine duration. Filling the field with a believable
    number would let a later edit drop the radiation column and fall back to
    the Angstrom estimate without anything failing.
    """
    day = nasa_power.to_daily_weather(records[0])
    assert math.isnan(day.sunshine_hours)


def test_derived_humidity_extremes_are_ordered_and_bounded(records):
    for record in records[::97]:
        rh_max, rh_min = nasa_power.relative_humidity_extremes(
            record.tmax_c, record.tmin_c, record.dewpoint_c
        )
        assert rh_min <= rh_max
        assert 1.0 <= rh_min <= 100.0
        assert 1.0 <= rh_max <= 100.0


def test_et0_takes_the_dewpoint_route_when_available(weather):
    """FAO-56 Eq. 14 must win over Eq. 17, or the hierarchy is decorative."""
    day = weather[0]
    # Bound to a local before the None check. Narrowing an attribute
    # expression works in some checkers and not others; narrowing a plain
    # local works everywhere.
    dewpoint = day.dewpoint_c
    assert dewpoint is not None, "POWER supplies dewpoint for every day"
    assert et0_for_day(day).ea == pytest.approx(saturation_vapour_pressure(dewpoint))


def test_et0_uses_measured_radiation_when_available(weather):
    day = weather[0]
    radiation = day.solar_radiation_mj
    assert radiation is not None, "POWER supplies radiation for every day"
    assert et0_for_day(day).rs == pytest.approx(radiation)


# --------------------------------------------------------------------------
# The external check on the physics
# --------------------------------------------------------------------------
def _annual_totals(weather, et0_series) -> dict[int, float]:
    totals: dict[int, float] = defaultdict(float)
    for day, et0 in zip(weather, et0_series):
        totals[_date_of(day).year] += et0
    return dict(totals)


def test_real_annual_et0_is_physically_plausible_for_dubai(weather, et0_series):
    """Roughly 2100-2400 mm/year, from 30 years of independent observations.

    The README previously cited a published UAE range of 2000-2200 mm. That is
    a national envelope including cooler inland and mountain areas; the Dubai
    cell sits at and slightly above its top end, which is what a coastal
    hyper-arid site should do. The band asserted here is Dubai-specific and
    derived from the data, so it is a regression guard rather than an
    independent claim - the independent claim is the agreement test below.
    """
    totals = _annual_totals(weather, et0_series)
    mean_annual = sum(totals.values()) / len(totals)
    assert 2100.0 < mean_annual < 2400.0
    assert all(1900.0 < total < 2600.0 for total in totals.values())


def test_synthetic_climatology_agrees_with_real_observations(weather, et0_series):
    """The cross-validation: two unrelated derivations, one answer.

    `climate/dubai.py` interpolates published monthly normals. This series is
    satellite irradiance plus reanalysis. Nothing links them. Agreement to a
    few per cent is meaningful evidence that both are right; a divergence would
    mean at least one is wrong and would be worth chasing.
    """
    totals = _annual_totals(weather, et0_series)
    real_annual = sum(totals.values()) / len(totals)
    synthetic_annual = sum(
        et0_for_day(normals_day(day)).et0_mm_day for day in range(1, 366)
    )
    relative_error = abs(real_annual - synthetic_annual) / real_annual
    assert relative_error < 0.05, (
        f"real {real_annual:.0f} mm vs synthetic {synthetic_annual:.0f} mm"
    )


def test_seasonal_shape_matches_month_by_month(weather, et0_series):
    """Annual totals can agree while the seasonal shape is wrong.

    That is not hypothetical here: the wind-height bug preserved the shape
    perfectly and only broke the magnitude. This test is the mirror image of
    the check that caught it.
    """
    real_by_month: dict[int, list[float]] = defaultdict(list)
    for day, et0 in zip(weather, et0_series):
        real_by_month[_date_of(day).month].append(et0)

    synthetic_by_month: dict[int, list[float]] = defaultdict(list)
    for day_of_year in range(1, 366):
        month = (dt.date(2001, 1, 1) + dt.timedelta(day_of_year - 1)).month
        synthetic_by_month[month].append(
            et0_for_day(normals_day(day_of_year)).et0_mm_day
        )

    for month in range(1, 13):
        real = sum(real_by_month[month]) / len(real_by_month[month])
        synthetic = sum(synthetic_by_month[month]) / len(synthetic_by_month[month])
        assert abs(real - synthetic) < 0.5, (
            f"month {month}: real {real:.2f} vs synthetic {synthetic:.2f} mm/day"
        )


def test_summer_peak_lands_in_the_right_month(weather, et0_series):
    """June or July. If ET0 peaks in April the astronomical terms are broken."""
    by_month: dict[int, list[float]] = defaultdict(list)
    for day, et0 in zip(weather, et0_series):
        by_month[_date_of(day).month].append(et0)
    peak_month = max(by_month, key=lambda m: sum(by_month[m]) / len(by_month[m]))
    assert peak_month in (6, 7)


def test_annual_rainfall_matches_published_dubai_climatology(records):
    """Roughly 100 mm a year. Guards a units confusion that actually shipped.

    The dashboard once reported 3,272 mm of annual rainfall for Dubai - the
    30-year total presented as an annual figure - directly above a caption
    stating that rainfall never offsets demand. The same slip put monthly
    totals on a chart axis labelled mm/day, where they sat about 30x too tall
    against ET0.

    Neither error touched the physics, so no other test noticed. This one
    anchors the quantity to published climatology, which is the only thing
    that distinguishes 109 mm from 3,272 mm.
    """
    annual = sum(r.rainfall_mm for r in records) / 30.0
    assert 70.0 < annual < 150.0, f"annual rainfall {annual:.0f} mm"


def test_daily_rainfall_never_approaches_daily_demand(records):
    """The premise of the whole project, asserted rather than assumed.

    If any month's mean daily rainfall rivalled its mean daily ET0, irrigation
    scheduling in that month would be a different problem.
    """
    rain_by_month: dict[int, list[float]] = defaultdict(list)
    for record in records:
        rain_by_month[record.date.month].append(record.rainfall_mm)

    wettest = max(
        sum(values) / len(values) for values in rain_by_month.values()
    )
    assert wettest < 1.5, f"wettest month averages {wettest:.2f} mm/day of rain"


def test_root_zone_wetness_is_drier_in_summer(records):
    """Real soil moisture must fall through the season ET0 rises.

    Cheap to assert, and it is the assumption the whole irrigation argument
    rests on. It also confirms the GWETROOT column is not accidentally the
    surface or profile layer, which have different seasonal amplitudes.
    """
    by_month: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_month[record.date.month].append(record.wetness_root)
    means = {m: sum(v) / len(v) for m, v in by_month.items()}
    winter = (means[1] + means[2] + means[3]) / 3.0
    summer = (means[7] + means[8] + means[9]) / 3.0
    assert summer < winter
