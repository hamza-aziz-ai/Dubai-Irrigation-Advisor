"""FAO-56 reference evapotranspiration (Penman-Monteith).

Reference: Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
"Crop evapotranspiration - Guidelines for computing crop water requirements."
FAO Irrigation and Drainage Paper 56. https://www.fao.org/4/x0490e/x0490e00.htm

Equation numbers in docstrings refer to that paper, so any line here can be
checked against the source. The published worked examples (17, 18, 20) are
implemented as tests - this module is verified against FAO's own arithmetic,
not against my expectations of it.

Every function is pure. No I/O, no state, no LLM. ET0 is the physical
foundation the whole system rests on; if it is wrong, nothing downstream is
worth computing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (FAO-56 Chapter 3)
SOLAR_CONSTANT = 0.0820          # MJ m-2 min-1
STEFAN_BOLTZMANN = 4.903e-9      # MJ K-4 m-2 day-1
ALBEDO_GRASS = 0.23              # reference crop albedo
LATENT_HEAT = 2.45               # MJ kg-1, at ~20 C


# --------------------------------------------------------------------------
# Atmospheric parameters
# --------------------------------------------------------------------------
def atmospheric_pressure(altitude_m: float) -> float:
    """Eq. 7 - atmospheric pressure [kPa] from elevation."""
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa: float) -> float:
    """Eq. 8 - psychrometric constant gamma [kPa/C]."""
    return 0.665e-3 * pressure_kpa


def saturation_vapour_pressure(temp_c: float) -> float:
    """Eq. 11 - saturation vapour pressure e0(T) [kPa]."""
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def slope_vapour_pressure_curve(temp_c: float) -> float:
    """Eq. 13 - slope of the saturation vapour pressure curve [kPa/C]."""
    es = saturation_vapour_pressure(temp_c)
    return 4098.0 * es / (temp_c + 237.3) ** 2


def mean_saturation_vapour_pressure(tmax_c: float, tmin_c: float) -> float:
    """Eq. 12 - es from daily temperature extremes.

    Averaging e0(Tmax) and e0(Tmin) rather than taking e0(Tmean): the curve is
    convex, so e0(Tmean) systematically underestimates es. FAO-56 is explicit
    that the mean of the extremes must be used.
    """
    return (saturation_vapour_pressure(tmax_c) + saturation_vapour_pressure(tmin_c)) / 2.0


def actual_vapour_pressure_from_rh(
    tmax_c: float, tmin_c: float, rh_max: float, rh_min: float
) -> float:
    """Eq. 17 - ea from maximum and minimum relative humidity [kPa]."""
    return (
        saturation_vapour_pressure(tmin_c) * rh_max / 100.0
        + saturation_vapour_pressure(tmax_c) * rh_min / 100.0
    ) / 2.0


def actual_vapour_pressure_from_dewpoint(tdew_c: float) -> float:
    """Eq. 14 - ea from dewpoint temperature [kPa]."""
    return saturation_vapour_pressure(tdew_c)


# --------------------------------------------------------------------------
# Radiation
# --------------------------------------------------------------------------
def inverse_relative_distance_earth_sun(day_of_year: int) -> float:
    """Eq. 23 - dr [-]."""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)


def solar_declination(day_of_year: int) -> float:
    """Eq. 24 - solar declination [rad]."""
    return 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)


def sunset_hour_angle(latitude_rad: float, declination_rad: float) -> float:
    """Eq. 25 - sunset hour angle omega_s [rad].

    The argument of arccos is clamped: inside the polar circles it exceeds
    [-1, 1] during polar day/night, which is physically meaningful (the sun
    never sets or never rises) but raises a domain error if passed through
    unguarded.
    """
    x = -math.tan(latitude_rad) * math.tan(declination_rad)
    return math.acos(max(-1.0, min(1.0, x)))


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """Eq. 21 - Ra [MJ m-2 day-1]."""
    phi = math.radians(latitude_deg)
    dr = inverse_relative_distance_earth_sun(day_of_year)
    decl = solar_declination(day_of_year)
    ws = sunset_hour_angle(phi, decl)
    return (
        (24.0 * 60.0 / math.pi)
        * SOLAR_CONSTANT
        * dr
        * (
            ws * math.sin(phi) * math.sin(decl)
            + math.cos(phi) * math.cos(decl) * math.sin(ws)
        )
    )


def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """Eq. 34 - N [hours]."""
    phi = math.radians(latitude_deg)
    ws = sunset_hour_angle(phi, solar_declination(day_of_year))
    return 24.0 / math.pi * ws


def solar_radiation_from_sunshine(
    sunshine_hours: float, latitude_deg: float, day_of_year: int,
    a_s: float = 0.25, b_s: float = 0.50,
) -> float:
    """Eq. 35 - Angstrom formula, Rs [MJ m-2 day-1]."""
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    n_over_N = sunshine_hours / daylight_hours(latitude_deg, day_of_year)
    return (a_s + b_s * n_over_N) * ra


def solar_radiation_from_temp_range(
    tmax_c: float, tmin_c: float, latitude_deg: float, day_of_year: int,
    k_rs: float = 0.16,
) -> float:
    """Eq. 50 - Hargreaves radiation estimate when Rs is unmeasured.

    k_rs = 0.16 for interior locations, 0.19 for coastal. Dubai is coastal,
    but the reference example (Lyon) is interior - the default matches FAO's
    worked example so the validation test is exact.
    """
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    return k_rs * math.sqrt(max(tmax_c - tmin_c, 0.0)) * ra


def clear_sky_radiation(altitude_m: float, ra: float) -> float:
    """Eq. 37 - Rso [MJ m-2 day-1]."""
    return (0.75 + 2e-5 * altitude_m) * ra


def net_shortwave_radiation(rs: float, albedo: float = ALBEDO_GRASS) -> float:
    """Eq. 38 - Rns [MJ m-2 day-1]."""
    return (1.0 - albedo) * rs


def net_longwave_radiation(
    tmax_c: float, tmin_c: float, ea_kpa: float, rs: float, rso: float
) -> float:
    """Eq. 39 - Rnl [MJ m-2 day-1]."""
    tmax_k4 = (tmax_c + 273.16) ** 4
    tmin_k4 = (tmin_c + 273.16) ** 4
    cloud = 1.35 * min(rs / rso, 1.0) - 0.35 if rso > 0 else 0.0
    return (
        STEFAN_BOLTZMANN
        * ((tmax_k4 + tmin_k4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(max(ea_kpa, 0.0)))
        * cloud
    )


def soil_heat_flux_monthly(t_month_c: float, t_prev_month_c: float) -> float:
    """Eq. 43 - G for monthly periods [MJ m-2 day-1].

    Negligible for daily steps (Eq. 42, G = 0) but not for monthly, where the
    soil is warming or cooling across the period.
    """
    return 0.14 * (t_month_c - t_prev_month_c)


def wind_speed_at_2m(wind_speed: float, measurement_height_m: float) -> float:
    """Eq. 47 - adjust wind speed to the 2 m reference height [m/s]."""
    if measurement_height_m == 2.0:
        return wind_speed
    return wind_speed * 4.87 / math.log(67.8 * measurement_height_m - 5.42)


# --------------------------------------------------------------------------
# ET0
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ET0Result:
    """ET0 with its intermediate quantities retained.

    The intermediates are kept deliberately: they are what the validation
    tests assert against, and what the explanation layer cites. A bare float
    would make both impossible.
    """

    et0_mm_day: float
    radiation_term: float
    aero_term: float
    rn: float
    rns: float
    rnl: float
    ra: float
    rs: float
    rso: float
    es: float
    ea: float
    vpd: float
    delta: float
    gamma: float
    u2: float
    g: float


def eto_penman_monteith(
    tmax_c: float,
    tmin_c: float,
    ea_kpa: float,
    u2_ms: float,
    rs_mj: float,
    latitude_deg: float,
    altitude_m: float,
    day_of_year: int,
    g_mj: float = 0.0,
) -> ET0Result:
    """Eq. 6 - FAO-56 Penman-Monteith reference evapotranspiration [mm/day].

    Takes Rs directly so the caller decides how it was obtained (measured,
    from sunshine hours, or from temperature range). Mixing that decision into
    the ET0 function would hide a significant source of uncertainty.
    """
    tmean = (tmax_c + tmin_c) / 2.0
    delta = slope_vapour_pressure_curve(tmean)
    pressure = atmospheric_pressure(altitude_m)
    gamma = psychrometric_constant(pressure)

    es = mean_saturation_vapour_pressure(tmax_c, tmin_c)
    vpd = max(es - ea_kpa, 0.0)

    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    rso = clear_sky_radiation(altitude_m, ra)
    rns = net_shortwave_radiation(rs_mj)
    rnl = net_longwave_radiation(tmax_c, tmin_c, ea_kpa, rs_mj, rso)
    rn = rns - rnl

    denominator = delta + gamma * (1.0 + 0.34 * u2_ms)
    radiation_term = 0.408 * (rn - g_mj) * delta / denominator
    aero_term = gamma * (900.0 / (tmean + 273.0)) * u2_ms * vpd / denominator

    return ET0Result(
        et0_mm_day=radiation_term + aero_term,
        radiation_term=radiation_term,
        aero_term=aero_term,
        rn=rn, rns=rns, rnl=rnl, ra=ra, rs=rs_mj, rso=rso,
        es=es, ea=ea_kpa, vpd=vpd, delta=delta, gamma=gamma, u2=u2_ms, g=g_mj,
    )
