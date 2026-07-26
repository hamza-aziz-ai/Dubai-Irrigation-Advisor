from __future__ import annotations

import pytest

from irrigation.climate.dubai import (
    MONTH_STARTS, WIND_10M_MS, WIND_MEASUREMENT_HEIGHT_M,
    DubaiWeatherGenerator, normals_day,
)
from irrigation.climate.et0_series import et0_for_day
from irrigation.climate.sensor import (
    SensorConfig, SensorReading, SoilMoistureSensor, interpolate_dropouts,
)

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class TestET0Climatology:
    """Magnitude checks against published UAE reference ET.

    The seasonal SHAPE was correct even when the wind height was wrong, so
    only a total-magnitude check catches that class of error. This is the
    regression test for it.
    """

    def annual_et0(self) -> float:
        return sum(
            et0_for_day(normals_day(s + 14)).et0_mm_day * DAYS_IN_MONTH[i]
            for i, s in enumerate(MONTH_STARTS)
        )

    def test_annual_total_within_published_range(self):
        assert 1900 <= self.annual_et0() <= 2300

    def test_summer_peak_in_expected_band(self):
        june = et0_for_day(normals_day(166)).et0_mm_day
        assert 7.5 <= june <= 10.0

    def test_winter_trough_in_expected_band(self):
        january = et0_for_day(normals_day(15)).et0_mm_day
        assert 2.5 <= january <= 4.0

    def test_summer_demand_at_least_double_winter(self):
        assert et0_for_day(normals_day(166)).et0_mm_day > 2 * et0_for_day(
            normals_day(15)
        ).et0_mm_day

    def test_wind_normals_are_converted_from_measurement_height(self):
        """Regression: 10 m wind used as 2 m inflated annual ET0 by ~9%."""
        assert WIND_MEASUREMENT_HEIGHT_M == 10.0
        day = normals_day(166)
        assert day.wind_2m_ms < max(WIND_10M_MS)
        assert day.wind_2m_ms == pytest.approx(
            _interp_wind(166) * 0.748, abs=0.05
        )


def _interp_wind(doy: int) -> float:
    from irrigation.climate.dubai import _interpolate_monthly
    return _interpolate_monthly(WIND_10M_MS, doy)


class TestWeatherGenerator:
    def test_reproducible_for_a_given_seed(self):
        a = DubaiWeatherGenerator(5).year(days=30)
        b = DubaiWeatherGenerator(5).year(days=30)
        assert [d.tmax_c for d in a] == [d.tmax_c for d in b]

    def test_diurnal_range_stays_physical(self):
        for d in DubaiWeatherGenerator(9).year(days=365):
            assert d.tmax_c > d.tmin_c
            assert d.rh_max_pct > d.rh_min_pct

    def test_summer_is_essentially_rainless(self):
        gen = DubaiWeatherGenerator(3)
        summer = [gen.day(d) for d in range(152, 244)]
        assert sum(d.rainfall_mm for d in summer) < 15.0

    def test_rainfall_is_intermittent_not_smeared(self):
        """Most days are exactly zero - a Gaussian would be the wrong shape."""
        gen = DubaiWeatherGenerator(3)
        year = [gen.day(d) for d in range(1, 366)]
        assert sum(1 for d in year if d.rainfall_mm == 0.0) / 365 > 0.9


class TestSensor:
    def test_reads_close_to_truth_before_drift(self):
        s = SoilMoistureSensor(SensorConfig(dropout_probability=0.0), seed=1)
        r = s.read(0, 0.10)
        assert r.available
        assert abs(r.measured_vwc - 0.10) < 0.05

    def test_drift_biases_upward_over_a_season(self):
        """Salinity accumulation makes the probe read progressively wetter."""
        cfg = SensorConfig(dropout_probability=0.0, noise_std=0.0)
        s = SoilMoistureSensor(cfg, seed=1)
        assert s.read(120, 0.10).measured_vwc > s.read(0, 0.10).measured_vwc

    def test_dropouts_occur_and_are_flagged(self):
        s = SoilMoistureSensor(SensorConfig(dropout_probability=0.5), seed=2)
        rd = s.read_series([0.10] * 100)
        assert any(not r.available for r in rd)
        assert all(r.measured_vwc is None for r in rd if not r.available)

    def test_quantised_to_adc_resolution(self):
        cfg = SensorConfig(dropout_probability=0.0, noise_std=0.0,
                           calibration_offset=0.0, drift_per_day=0.0, adc_bits=8)
        s = SoilMoistureSensor(cfg, seed=1)
        step = cfg.range_max / 255
        v = s.read(0, 0.1234).measured_vwc
        assert abs(round(v / step) * step - v) < 1e-9

    def test_interpolation_fills_every_gap(self):
        s = SoilMoistureSensor(SensorConfig(dropout_probability=0.4), seed=4)
        filled = interpolate_dropouts(s.read_series([0.10] * 60))
        assert len(filled) == 60
        assert all(v is not None for v in filled)

    def test_interpolation_refuses_a_fully_dead_sensor(self):
        s = SoilMoistureSensor(SensorConfig(dropout_probability=1.0), seed=4)
        with pytest.raises(ValueError, match="no readings"):
            interpolate_dropouts(s.read_series([0.10] * 10))

    # The tests above run on a constant series, where any gap-filling rule
    # looks correct because every answer is 0.10. These use a varying series,
    # so the interior slope and the edge-holding rule are actually exercised.

    def test_interior_gap_is_filled_by_linear_interpolation(self):
        readings = [
            SensorReading(0, 0.0, 0.10),
            SensorReading(1, 0.0, None),
            SensorReading(2, 0.0, None),
            SensorReading(3, 0.0, 0.40),
        ]
        assert interpolate_dropouts(readings) == pytest.approx([0.10, 0.20, 0.30, 0.40])

    def test_leading_and_trailing_gaps_hold_the_nearest_reading(self):
        """Held flat, not extrapolated.

        Extrapolating past the last real measurement would invent a trend the
        probe never reported, and the irrigation decision would act on it.
        """
        readings = [
            SensorReading(0, 0.0, None),
            SensorReading(1, 0.0, 0.20),
            SensorReading(2, 0.0, 0.30),
            SensorReading(3, 0.0, None),
        ]
        assert interpolate_dropouts(readings) == pytest.approx([0.20, 0.20, 0.30, 0.30])

    def test_a_single_surviving_reading_fills_the_whole_series(self):
        readings = [
            SensorReading(0, 0.0, None),
            SensorReading(1, 0.0, 0.17),
            SensorReading(2, 0.0, None),
        ]
        assert interpolate_dropouts(readings) == pytest.approx([0.17, 0.17, 0.17])
