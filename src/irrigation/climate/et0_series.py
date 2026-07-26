"""
Turn weather into ET0 - the join between the climate and physics layers.

This module owns exactly one judgement: given whatever a data source happened
to measure, which FAO-56 route to ea and Rs should be used. FAO-56 ranks those
routes by accuracy, and the ranking is not a detail - choosing the temperature
based radiation estimate when measured radiation was available is a several
per cent error in ET0 for no reason at all.

The choice is made here rather than inside `eto_penman_monteith` on purpose.
The physics function takes ea and Rs as plain numbers so that a reader can see
exactly what went in; hiding the fallback chain inside it would bury a real
source of uncertainty in a function whose whole value is being auditable.
"""
from __future__ import annotations

from ..physics.penman_monteith import (
    ET0Result,
    actual_vapour_pressure_from_dewpoint,
    actual_vapour_pressure_from_rh,
    eto_penman_monteith,
    solar_radiation_from_sunshine,
)
from .dubai import DUBAI_ALTITUDE_M, DUBAI_LATITUDE, DailyWeather


def actual_vapour_pressure_for(weather: DailyWeather) -> float:
    """
    Best available ea for a day [kPa], following the FAO-56 hierarchy.

    Dewpoint (Eq. 14) is preferred because it is a direct measurement of the
    quantity ea *is*. The RHmax/RHmin route (Eq. 17) reconstructs it from two
    humidity extremes paired with two temperature extremes, which is accurate
    but inherits error from all four.
    """
    if weather.dewpoint_c is not None:
        return actual_vapour_pressure_from_dewpoint(weather.dewpoint_c)
    return actual_vapour_pressure_from_rh(
        weather.tmax_c, weather.tmin_c, weather.rh_max_pct, weather.rh_min_pct
    )


def solar_radiation_for(
    weather: DailyWeather, latitude_deg: float = DUBAI_LATITUDE
) -> float:
    """
    Best available Rs for a day [MJ m-2 day-1].

    A measured value is used as-is. Otherwise, the Angstrom formula (Eq. 35)
    converts sunshine duration, which carries the uncertainty of the a_s/b_s
    coefficients - calibrated regionally, and left at FAO's default here.
    """
    if weather.solar_radiation_mj is not None:
        return weather.solar_radiation_mj
    return solar_radiation_from_sunshine(
        weather.sunshine_hours, latitude_deg, weather.day_of_year
    )


def et0_for_day(
    weather: DailyWeather,
    latitude_deg: float = DUBAI_LATITUDE,
    altitude_m: float = DUBAI_ALTITUDE_M,
) -> ET0Result:
    """
    FAO-56 reference evapotranspiration for one day of weather.

    Args:
        weather: The day's observations. Optional measured fields are used
            when present; see `actual_vapour_pressure_for` and
            `solar_radiation_for` for the fallback order.
        latitude_deg: Site latitude, positive north. Drives Ra.
        altitude_m: Site elevation, used for atmospheric pressure and Rso.

    Returns:
        ET0Result carrying ET0 and every intermediate, so the explanation
        layer can cite them and the tests can assert on them.
    """
    ea = actual_vapour_pressure_for(weather)
    rs = solar_radiation_for(weather, latitude_deg)
    # G is taken as zero: negligible at a daily step (FAO-56 Eq. 42).
    return eto_penman_monteith(
        tmax_c=weather.tmax_c, tmin_c=weather.tmin_c, ea_kpa=ea,
        u2_ms=weather.wind_2m_ms, rs_mj=rs,
        latitude_deg=latitude_deg, altitude_m=altitude_m,
        day_of_year=weather.day_of_year, g_mj=0.0,
    )
