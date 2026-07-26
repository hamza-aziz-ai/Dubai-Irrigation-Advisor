"""Validation against FAO-56's own published worked examples.

This is the load-bearing test file. Everything downstream - soil water
balance, irrigation decisions, the models, the explanations - is built on
ET0. If these three examples do not reproduce, nothing else is meaningful.

The examples are taken from FAO Irrigation and Drainage Paper 56, Chapter 4:
https://www.fao.org/4/x0490e/x0490e08.htm

Intermediates are asserted as well as the final answer. A wrong Ra that
happens to cancel against a wrong Rnl would pass a final-answer-only test.
"""
from __future__ import annotations

import pytest

from irrigation.physics.penman_monteith import (
    actual_vapour_pressure_from_dewpoint,
    actual_vapour_pressure_from_rh,
    atmospheric_pressure,
    clear_sky_radiation,
    daylight_hours,
    eto_penman_monteith,
    extraterrestrial_radiation,
    mean_saturation_vapour_pressure,
    net_longwave_radiation,
    net_shortwave_radiation,
    psychrometric_constant,
    saturation_vapour_pressure,
    slope_vapour_pressure_curve,
    soil_heat_flux_monthly,
    solar_radiation_from_sunshine,
    solar_radiation_from_temp_range,
    wind_speed_at_2m,
)

# FAO prints intermediates to 2-3 significant figures, so tolerances here
# reflect the precision of the published source, not of the computation.
ABS_2DP = 0.01
ABS_1DP = 0.05


class TestExample17Bangkok:
    """Example 17 - ETo with mean monthly data. Bangkok, April.

    13 deg 44' N, altitude 2 m. Published answer: ETo = 5.72 mm/day.
    """

    LAT = 13 + 44 / 60      # 13.73 N
    ALT = 2.0
    DOY = 105               # 15 April
    TMAX, TMIN = 34.8, 25.6
    EA = 2.85
    U2 = 2.0
    SUNSHINE = 8.5

    def test_derived_parameters(self):
        tmean = (self.TMAX + self.TMIN) / 2
        assert tmean == pytest.approx(30.2, abs=ABS_2DP)
        assert slope_vapour_pressure_curve(tmean) == pytest.approx(0.246, abs=1e-3)
        # FAO prints P to one decimal (101.3); exact value is 101.276.
        assert atmospheric_pressure(self.ALT) == pytest.approx(101.3, abs=ABS_1DP)
        assert psychrometric_constant(atmospheric_pressure(self.ALT)) == pytest.approx(
            0.0674, abs=1e-4
        )

    def test_vapour_pressure(self):
        assert saturation_vapour_pressure(self.TMAX) == pytest.approx(5.56, abs=ABS_2DP)
        assert saturation_vapour_pressure(self.TMIN) == pytest.approx(3.28, abs=ABS_2DP)
        es = mean_saturation_vapour_pressure(self.TMAX, self.TMIN)
        assert es == pytest.approx(4.42, abs=ABS_2DP)
        assert es - self.EA == pytest.approx(1.57, abs=ABS_2DP)

    def test_radiation(self):
        ra = extraterrestrial_radiation(self.LAT, self.DOY)
        assert ra == pytest.approx(38.06, abs=ABS_1DP)
        assert daylight_hours(self.LAT, self.DOY) == pytest.approx(12.31, abs=ABS_2DP)

        rs = solar_radiation_from_sunshine(self.SUNSHINE, self.LAT, self.DOY)
        assert rs == pytest.approx(22.65, abs=ABS_1DP)
        assert clear_sky_radiation(self.ALT, ra) == pytest.approx(28.54, abs=ABS_1DP)
        assert net_shortwave_radiation(rs) == pytest.approx(17.44, abs=ABS_1DP)

        rso = clear_sky_radiation(self.ALT, ra)
        rnl = net_longwave_radiation(self.TMAX, self.TMIN, self.EA, rs, rso)
        assert rnl == pytest.approx(3.11, abs=ABS_1DP)

    def test_soil_heat_flux_not_ignored_for_monthly_step(self):
        """G is negligible daily but not monthly - FAO Eq. 43."""
        assert soil_heat_flux_monthly(30.2, 29.2) == pytest.approx(0.14, abs=1e-3)

    def test_eto(self):
        rs = solar_radiation_from_sunshine(self.SUNSHINE, self.LAT, self.DOY)
        g = soil_heat_flux_monthly(30.2, 29.2)
        r = eto_penman_monteith(
            self.TMAX, self.TMIN, self.EA, self.U2, rs, self.LAT, self.ALT, self.DOY, g
        )
        assert r.rn == pytest.approx(14.33, abs=ABS_1DP)
        assert r.radiation_term == pytest.approx(3.97, abs=ABS_1DP)
        assert r.aero_term == pytest.approx(1.75, abs=ABS_1DP)
        assert r.et0_mm_day == pytest.approx(5.72, abs=ABS_2DP)


class TestExample18Brussels:
    """Example 18 - ETo with daily data. Uccle (Brussels), 6 July.

    50 deg 48' N, altitude 100 m. Published answer: ETo = 3.88 mm/day.
    Exercises the RH path and the wind-height adjustment, neither of which
    Example 17 touches.
    """

    LAT = 50 + 48 / 60      # 50.80 N
    ALT = 100.0
    DOY = 187               # 6 July
    TMAX, TMIN = 21.5, 12.3
    RH_MAX, RH_MIN = 84.0, 63.0
    SUNSHINE = 9.25

    def test_wind_adjusted_from_10m(self):
        u10 = 10 / 3.6                      # 10 km/h -> m/s
        assert u10 == pytest.approx(2.78, abs=ABS_2DP)
        assert wind_speed_at_2m(u10, 10.0) == pytest.approx(2.078, abs=1e-3)

    def test_vapour_pressure_from_relative_humidity(self):
        assert saturation_vapour_pressure(self.TMAX) == pytest.approx(2.564, abs=1e-3)
        assert saturation_vapour_pressure(self.TMIN) == pytest.approx(1.431, abs=1e-3)
        assert mean_saturation_vapour_pressure(self.TMAX, self.TMIN) == pytest.approx(
            1.997, abs=1e-3
        )
        ea = actual_vapour_pressure_from_rh(self.TMAX, self.TMIN, self.RH_MAX, self.RH_MIN)
        assert ea == pytest.approx(1.409, abs=1e-3)

    def test_radiation(self):
        ra = extraterrestrial_radiation(self.LAT, self.DOY)
        assert ra == pytest.approx(41.09, abs=ABS_1DP)
        assert daylight_hours(self.LAT, self.DOY) == pytest.approx(16.1, abs=ABS_1DP)
        rs = solar_radiation_from_sunshine(self.SUNSHINE, self.LAT, self.DOY)
        assert rs == pytest.approx(22.07, abs=0.1)
        assert clear_sky_radiation(self.ALT, ra) == pytest.approx(30.90, abs=ABS_1DP)

    def test_eto(self):
        u2 = wind_speed_at_2m(10 / 3.6, 10.0)
        ea = actual_vapour_pressure_from_rh(self.TMAX, self.TMIN, self.RH_MAX, self.RH_MIN)
        rs = solar_radiation_from_sunshine(self.SUNSHINE, self.LAT, self.DOY)
        r = eto_penman_monteith(
            self.TMAX, self.TMIN, ea, u2, rs, self.LAT, self.ALT, self.DOY, 0.0
        )
        assert r.rn == pytest.approx(13.28, abs=0.1)
        assert r.et0_mm_day == pytest.approx(3.88, abs=ABS_1DP)


class TestExample20Lyon:
    """Example 20 - ETo with missing data. Lyon, July, temperature only.

    45 deg 43' N, altitude 200 m. Published answer: ETo = 4.56 mm/day.
    This is the case that matters operationally: a field site rarely has
    radiation or humidity instrumentation, so the degraded path has to be
    correct too.
    """

    LAT = 45 + 43 / 60      # 45.72 N
    ALT = 200.0
    DOY = 196               # 15 July
    TMAX, TMIN = 26.6, 14.8
    U2_ASSUMED = 2.0        # FAO's recommended default when unmeasured

    def test_humidity_estimated_from_tmin(self):
        """Tdew ~ Tmin (Eq. 48) - valid where the air saturates overnight."""
        ea = actual_vapour_pressure_from_dewpoint(self.TMIN)
        assert ea == pytest.approx(1.68, abs=ABS_2DP)

    def test_radiation_estimated_from_temperature_range(self):
        rs = solar_radiation_from_temp_range(self.TMAX, self.TMIN, self.LAT, self.DOY)
        assert rs == pytest.approx(22.29, abs=0.1)

    def test_eto(self):
        ea = actual_vapour_pressure_from_dewpoint(self.TMIN)
        rs = solar_radiation_from_temp_range(self.TMAX, self.TMIN, self.LAT, self.DOY)
        r = eto_penman_monteith(
            self.TMAX, self.TMIN, ea, self.U2_ASSUMED, rs, self.LAT, self.ALT, self.DOY, 0.0
        )
        assert r.rn == pytest.approx(13.48, abs=0.1)
        assert r.et0_mm_day == pytest.approx(4.56, abs=ABS_1DP)

    def test_wind_sensitivity_matches_published_range(self):
        """FAO states ETo would be ~7% lower at 1 m/s, ~6% higher at 3 m/s.

        Worth pinning: it bounds how much error an assumed wind speed can
        introduce, which is the largest uncertainty at an uninstrumented site.
        """
        ea = actual_vapour_pressure_from_dewpoint(self.TMIN)
        rs = solar_radiation_from_temp_range(self.TMAX, self.TMIN, self.LAT, self.DOY)
        base, low, high = (
            eto_penman_monteith(
                self.TMAX, self.TMIN, ea, u, rs, self.LAT, self.ALT, self.DOY, 0.0
            ).et0_mm_day
            for u in (2.0, 1.0, 3.0)
        )
        assert low == pytest.approx(4.2, abs=0.1)
        assert high == pytest.approx(4.8, abs=0.1)
        assert (base - low) / base == pytest.approx(0.07, abs=0.02)
        assert (high - base) / base == pytest.approx(0.06, abs=0.02)


class TestPhysicalSanity:
    """Properties that must hold regardless of any published example."""

    def test_polar_night_does_not_raise(self):
        """arccos domain guard - Tromso in December."""
        assert extraterrestrial_radiation(69.6, 355) >= 0.0

    def test_es_from_extremes_exceeds_es_from_mean(self):
        """The vapour pressure curve is convex; e0(Tmean) underestimates es."""
        tmax, tmin = 40.0, 20.0
        assert mean_saturation_vapour_pressure(tmax, tmin) > saturation_vapour_pressure(
            (tmax + tmin) / 2
        )

    def test_eto_rises_with_vapour_pressure_deficit(self):
        common = dict(
            tmax_c=40.0, tmin_c=28.0, u2_ms=2.0, rs_mj=25.0,
            latitude_deg=25.25, altitude_m=5.0, day_of_year=196,
        )
        humid = eto_penman_monteith(ea_kpa=4.0, **common).et0_mm_day
        arid = eto_penman_monteith(ea_kpa=1.0, **common).et0_mm_day
        assert arid > humid

    def test_cloudiness_ratio_is_clamped(self):
        """Rs can exceed Rso on a bright day; the ratio must not run away."""
        ra = extraterrestrial_radiation(25.25, 196)
        rso = clear_sky_radiation(5.0, ra)
        normal = net_longwave_radiation(40.0, 28.0, 2.0, rso * 0.9, rso)
        over = net_longwave_radiation(40.0, 28.0, 2.0, rso * 1.4, rso)
        assert over == pytest.approx(net_longwave_radiation(40.0, 28.0, 2.0, rso, rso))
        assert over > normal
