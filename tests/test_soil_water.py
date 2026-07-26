from __future__ import annotations

import pytest

from irrigation.physics.crop import CROPS, SANDY, SANDY_LOAM, SOILS
from irrigation.physics.soil_water import step, water_stress_coefficient

TURF = CROPS["turfgrass"]


class TestAvailableWater:
    def test_sand_holds_far_less_than_loam(self):
        """The core Dubai constraint: sand gives you almost no buffer."""
        assert SANDY.total_available_water_mm(0.5) == pytest.approx(35.0)
        assert SANDY_LOAM.total_available_water_mm(0.5) == pytest.approx(55.0)

    def test_raw_is_a_fraction_of_taw(self):
        taw = SANDY.total_available_water_mm(0.5)
        raw = SANDY.readily_available_water_mm(0.5, 0.5)
        assert raw == pytest.approx(taw * 0.5)


class TestDepletionFractionAdjustment:
    def test_high_demand_lowers_the_stress_threshold(self):
        """FAO-56 Eq. 83. Dubai summer ET0 is 8-10 mm/day, not the tabulated 5."""
        p_mild = TURF.adjusted_depletion_fraction(5.0)
        p_summer = TURF.adjusted_depletion_fraction(9.0)
        assert p_mild == pytest.approx(TURF.depletion_fraction)
        assert p_summer < p_mild

    def test_clamped_to_physical_range(self):
        assert 0.1 <= TURF.adjusted_depletion_fraction(30.0) <= 0.8
        assert 0.1 <= TURF.adjusted_depletion_fraction(0.1) <= 0.8


class TestStressCoefficient:
    def test_no_stress_within_raw(self):
        assert water_stress_coefficient(10.0, 100.0, 50.0) == 1.0

    def test_linear_decline_beyond_raw(self):
        assert water_stress_coefficient(75.0, 100.0, 50.0) == pytest.approx(0.5)

    def test_zero_at_total_depletion(self):
        assert water_stress_coefficient(100.0, 100.0, 50.0) == 0.0

    def test_never_negative(self):
        assert water_stress_coefficient(150.0, 100.0, 50.0) == 0.0


class TestWaterBalance:
    def test_evapotranspiration_increases_depletion(self):
        s = step(0.0, et0_mm=8.0, crop=TURF, soil=SANDY, root_depth_m=0.5, kc=0.85)
        assert s.depletion_mm == pytest.approx(6.8, abs=0.01)

    def test_irrigation_reduces_depletion_net_of_efficiency(self):
        s = step(20.0, et0_mm=0.0, crop=TURF, soil=SANDY, root_depth_m=0.5,
                 kc=0.85, irrigation_mm=10.0, irrigation_efficiency=0.9)
        assert s.depletion_mm == pytest.approx(11.0, abs=0.01)

    def test_overwatering_drains_and_is_not_stored(self):
        """Water applied past field capacity is gone - the sand cannot hold it."""
        s = step(5.0, et0_mm=0.0, crop=TURF, soil=SANDY, root_depth_m=0.5,
                 kc=0.85, irrigation_mm=50.0, irrigation_efficiency=1.0)
        assert s.depletion_mm == pytest.approx(0.0)
        assert s.drainage_mm == pytest.approx(45.0, abs=0.01)

    def test_depletion_cannot_exceed_taw(self):
        s = step(34.0, et0_mm=20.0, crop=TURF, soil=SANDY, root_depth_m=0.5, kc=0.85)
        assert s.depletion_mm <= s.taw_mm

    def test_stress_reduces_actual_below_potential_et(self):
        s = step(30.0, et0_mm=9.0, crop=TURF, soil=SANDY, root_depth_m=0.5, kc=0.85)
        assert s.stressed
        assert s.etc_mm < s.etc_potential_mm

    def test_intense_rainfall_partly_runs_off(self):
        s = step(30.0, et0_mm=0.0, crop=TURF, soil=SANDY, root_depth_m=0.5,
                 kc=0.85, rainfall_mm=200.0)
        assert s.runoff_mm > 0

    def test_irrigating_before_et_matches_same_day_availability(self):
        """Applying water in the morning must relieve stress that same day."""
        dry = step(30.0, et0_mm=9.0, crop=TURF, soil=SANDY, root_depth_m=0.5, kc=0.85)
        wet = step(30.0, et0_mm=9.0, crop=TURF, soil=SANDY, root_depth_m=0.5,
                   kc=0.85, irrigation_mm=25.0)
        assert dry.ks < 1.0
        assert wet.ks == 1.0

    def test_volumetric_conversion_matches_probe_units(self):
        s = step(0.0, et0_mm=0.0, crop=TURF, soil=SANDY, root_depth_m=0.5, kc=0.85)
        assert s.volumetric_water_content(SANDY, 0.5) == pytest.approx(SANDY.field_capacity)
