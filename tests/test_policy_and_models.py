from __future__ import annotations

import pytest

from irrigation.decision.policy import CostModel, decide
from irrigation.explain.advisor import explain_decision, retrieve
from irrigation.models.evaluate import run_comparison
from irrigation.models.predictors import PhysicsBalance, PhysicsSensorFusion
from irrigation.models.simulate import simulate_season
from irrigation.physics.crop import CROPS, SOILS
from irrigation.physics.soil_water import step

TURF, SAND = CROPS["turfgrass"], SOILS["sand"]
KW = dict(crop=TURF, soil=SAND, root_depth_m=0.5, kc=0.85)

# The supervised models, referred to as a set rather than individually. Their
# ordering on cost depends on one seeded training set, so a test that names
# one of them as the winner would be asserting that seed.
SUPERVISED_MODELS = {"Random forest", "Gradient boosting", "XGBoost"}


class TestPolicy:
    def test_holds_when_root_zone_is_wet(self):
        d = decide(predicted_depletion_mm=2.0, et0_forecast_mm=8.0, **KW)
        assert d.action == "hold" and d.depth_mm == 0.0

    def test_irrigates_when_depletion_approaches_raw(self):
        d = decide(predicted_depletion_mm=14.0, et0_forecast_mm=8.5, **KW)
        assert d.action == "irrigate" and d.depth_mm > 0

    def test_trigger_leads_the_deficit_rather_than_following_it(self):
        """Firing only at RAW means stressing on the day you irrigate."""
        d = decide(predicted_depletion_mm=10.0, et0_forecast_mm=9.0, **KW)
        assert d.predicted_depletion_mm < d.raw_mm
        assert d.action == "irrigate"

    def test_hot_forecast_brings_irrigation_forward(self):
        mild = decide(predicted_depletion_mm=9.0, et0_forecast_mm=3.0, **KW)
        hot = decide(predicted_depletion_mm=9.0, et0_forecast_mm=10.0, **KW)
        assert mild.action == "hold"
        assert hot.action == "irrigate"

    def test_application_never_exceeds_what_the_profile_holds(self):
        d = decide(predicted_depletion_mm=500.0, et0_forecast_mm=9.0, **KW)
        assert d.depth_mm <= d.taw_mm / 0.9 + 1e-9

    def test_gross_depth_accounts_for_efficiency(self):
        d = decide(predicted_depletion_mm=20.0, et0_forecast_mm=9.0,
                   irrigation_efficiency=0.5, **KW)
        assert d.depth_mm == pytest.approx(40.0, abs=0.1)

    @pytest.mark.parametrize("kc", [0.6, 1.15])
    def test_policy_adjusts_p_by_forecast_crop_et(self, kc):
        d = decide(predicted_depletion_mm=5.0, et0_forecast_mm=9.0,
                   crop=TURF, soil=SAND, root_depth_m=0.5, kc=kc)
        p = TURF.adjusted_depletion_fraction(9.0 * kc)
        assert d.raw_mm == pytest.approx(SAND.readily_available_water_mm(0.5, p))

    def test_reason_is_populated_for_both_actions(self):
        for depletion in (2.0, 20.0):
            assert decide(predicted_depletion_mm=depletion,
                          et0_forecast_mm=8.0, **KW).reason


class TestCostAsymmetry:
    """The central claim: under- and over-watering are not equally bad."""

    def test_stress_costs_more_than_the_equivalent_wasted_water(self):
        cm = CostModel()
        wasted = cm.evaluate_day(applied_mm=10.0, depletion_mm=5.0,
                                 taw_mm=35.0, raw_mm=17.5)
        starved = cm.evaluate_day(applied_mm=0.0, depletion_mm=27.5,
                                  taw_mm=35.0, raw_mm=17.5)
        assert starved["total_cost"] > wasted["total_cost"]

    def test_severe_depletion_is_penalised_superlinearly(self):
        cm = CostModel()
        moderate = cm.evaluate_day(0.0, depletion_mm=26.0, taw_mm=35.0, raw_mm=17.5)
        severe = cm.evaluate_day(0.0, depletion_mm=30.0, taw_mm=35.0, raw_mm=17.5)
        ratio = severe["stress_cost"] / moderate["stress_cost"]
        assert ratio > (severe["deficit_mm"] / moderate["deficit_mm"])

    def test_no_stress_cost_within_raw(self):
        cm = CostModel()
        assert cm.evaluate_day(0.0, depletion_mm=10.0, taw_mm=35.0,
                               raw_mm=17.5)["stress_cost"] == 0.0


class TestModelComparison:
    """The finding this project exists to demonstrate."""

    @pytest.fixture(scope="class")
    def results(self):
        return run_comparison(TURF, SAND, root_depth_m=0.5, kc=0.85)

    def test_all_predictors_complete_a_season(self, results):
        assert len(results) == 6
        assert len({r.predictor_name for r in results}) == 6
        assert all(len(r.records) == 120 for r in results)

    def test_physics_baseline_beats_trusting_the_sensor(self, results):
        physics = next(r for r in results if r.predictor_name.startswith("Physics ("))
        sensor = next(r for r in results if r.predictor_name == "Sensor only")
        assert physics.total_cost < sensor.total_cost

    def test_sensor_bias_causes_chronic_under_irrigation(self, results):
        """Calibration offset + drift read wet, so the controller starves the crop."""
        sensor = next(r for r in results if r.predictor_name == "Sensor only")
        assert sensor.depletion_bias < -5.0
        assert sensor.stress_days > 50

    def test_physics_baseline_is_the_simulators_own_model(self, results):
        """The physics predictor runs the simulator's balance on the
        simulator's weather, so in here it tracks the truth exactly.

        That is internal consistency, not field accuracy, and every result in
        this class is read against it: the comparison asks how close the other
        methods get to a perfect process model, and what the gap costs.
        """
        physics = next(r for r in results if r.predictor_name.startswith("Physics ("))
        assert physics.depletion_rmse == pytest.approx(0.0, abs=1e-9)

    def test_physics_wins_at_the_default_cost_ratio(self, results):
        best_cost = min(results, key=lambda r: r.total_cost)
        assert best_cost.predictor_name.startswith("Physics (")

    def test_physics_still_wins_when_stress_is_catastrophically_expensive(self):
        """No crossover in the tested range.

        Stress at 48x the price of water is the most extreme ratio swept. The
        supervised models err toward under-estimating depletion, so making
        stress dearer widens their gap to the physics baseline rather than
        closing it. If a supervised model ever wins here, the README's
        sensitivity table needs rewriting, not this test relaxing.
        """
        extreme = CostModel(stress_cost_per_mm_deficit=120.0)
        results = run_comparison(TURF, SAND, root_depth_m=0.5, kc=0.85,
                                 cost_model=extreme)
        winner = min(results, key=lambda r: r.total_cost)
        assert winner.predictor_name.startswith("Physics (")

    def test_every_supervised_model_errs_on_the_wet_side(self, results):
        """The direction of the error is shared, which is what the cost model
        charges for.

        Random forest, gradient boosting and XGBoost all estimate the root
        zone as wetter than it is, so all three let it dry past RAW on more
        days than the physics baseline. Under asymmetric cost that direction
        is the expensive one, and RMSE alone does not reveal it.
        """
        physics = next(r for r in results if r.predictor_name.startswith("Physics ("))
        supervised = [r for r in results if r.predictor_name in SUPERVISED_MODELS]
        assert len(supervised) == 3
        for model in supervised:
            assert model.depletion_bias < 0.0, model.predictor_name
            assert model.stress_days > physics.stress_days, model.predictor_name


class TestSimulatorIntegrity:
    def test_predictor_cannot_observe_true_state(self):
        """Observation carries only field-available information."""
        from dataclasses import fields as dataclass_fields

        from irrigation.models.predictors import Observation
        fields = {f.name for f in dataclass_fields(Observation)}
        assert "true_depletion" not in fields
        assert "depletion_mm" not in fields

    def test_evaluation_uses_unseen_weather(self):
        from irrigation.models.evaluate import EVAL_WEATHER_SEED, TRAIN_SEEDS
        assert EVAL_WEATHER_SEED not in TRAIN_SEEDS

    def test_season_is_reproducible(self):
        a, _, _ = simulate_season(PhysicsBalance(TURF, SAND), TURF, SAND,
                                  root_depth_m=0.5, kc=0.85)
        b, _, _ = simulate_season(PhysicsBalance(TURF, SAND), TURF, SAND,
                                  root_depth_m=0.5, kc=0.85)
        assert a.total_cost == pytest.approx(b.total_cost)


def _observation(day: int, et0: float, irrigation: float = 0.0,
                 sensor_vwc: float | None = None, kc: float = 0.85):
    from irrigation.climate.dubai import normals_day
    from irrigation.models.predictors import Observation

    return Observation(
        day_index=day, day_of_year=150 + day, sensor_vwc=sensor_vwc,
        et0_today_mm=et0, et0_forecast_mm=et0, weather=normals_day(150 + day),
        irrigation_yesterday_mm=irrigation, rainfall_yesterday_mm=0.0,
        kc=kc, root_depth_m=0.5,
    )


class TestPredictionTiming:
    """A morning estimate describes the morning: yesterday's balance, closed
    with yesterday's ET and yesterday's water."""

    def test_first_morning_takes_no_step(self):
        assert PhysicsBalance(TURF, SAND).predict(_observation(0, et0=9.0)) == 0.0

    def test_closes_yesterday_with_yesterdays_et(self):
        physics = PhysicsBalance(TURF, SAND)
        physics.predict(_observation(0, et0=4.0))
        estimate = physics.predict(_observation(1, et0=10.0))
        assert estimate == pytest.approx(step(0.0, et0_mm=4.0, **KW).depletion_mm)

    def test_yesterdays_water_is_paired_with_yesterdays_et(self):
        physics = PhysicsBalance(TURF, SAND)
        physics.predict(_observation(0, et0=4.0))
        physics.predict(_observation(1, et0=4.0))
        estimate = physics.predict(_observation(2, et0=10.0, irrigation=5.0))
        day1 = step(0.0, et0_mm=4.0, **KW).depletion_mm
        expected = step(day1, et0_mm=4.0, irrigation_mm=5.0, **KW).depletion_mm
        assert estimate == pytest.approx(expected)

    def test_reset_forgets_the_previous_season(self):
        physics = PhysicsBalance(TURF, SAND)
        physics.predict(_observation(0, et0=9.0))
        physics.predict(_observation(1, et0=9.0))
        physics.reset()
        assert physics.predict(_observation(0, et0=9.0)) == 0.0

    def test_fusion_with_zero_gain_is_the_physics_balance(self):
        physics, fusion = PhysicsBalance(TURF, SAND), PhysicsSensorFusion(TURF, SAND, gain=0.0)
        for day, et0 in enumerate([4.0, 7.0, 10.0]):
            obs = _observation(day, et0=et0, sensor_vwc=0.08)
            assert fusion.predict(obs) == pytest.approx(physics.predict(obs))

    def test_fusion_follows_physics_through_a_dropout(self):
        physics, fusion = PhysicsBalance(TURF, SAND), PhysicsSensorFusion(TURF, SAND)
        for day, et0 in enumerate([4.0, 7.0, 10.0]):
            obs = _observation(day, et0=et0, sensor_vwc=None)
            assert fusion.predict(obs) == pytest.approx(physics.predict(obs))

    def test_scored_against_the_morning_not_the_evening(self):
        """A perfect morning estimate scores zero even though the post-step
        state differs from it on every day it irrigates or evaporates."""
        result, _, _ = simulate_season(PhysicsBalance(TURF, SAND), TURF, SAND,
                                       root_depth_m=0.5, kc=0.85, days=30)
        assert result.depletion_rmse == pytest.approx(0.0, abs=1e-9)
        assert any(r.true_depletion_mm != r.morning_depletion_mm for r in result.records)
        for r in result.records:
            assert r.predicted_depletion_mm == pytest.approx(r.morning_depletion_mm)


class TestExplanations:
    @pytest.fixture(autouse=True)
    def offline_engine(self, monkeypatch):
        """The default engine follows the environment; these tests need offline."""
        monkeypatch.delenv("IRRIGATION_EXPLAIN_ENGINE", raising=False)

    def test_irrigate_explanation_cites_the_computed_numbers(self):
        d = decide(predicted_depletion_mm=15.0, et0_forecast_mm=8.5, **KW)
        e = explain_decision(d, 8.5, "Warm-season turfgrass", "Sand")
        assert "IRRIGATE" in e.headline
        assert f"{d.raw_mm:.1f}" in e.body
        assert e.citations

    def test_hold_explanation_is_produced(self):
        d = decide(predicted_depletion_mm=1.0, et0_forecast_mm=3.0, **KW)
        e = explain_decision(d, 3.0, "Warm-season turfgrass", "Sand")
        assert "HOLD" in e.headline

    def test_retrieval_is_conditional_on_the_decision(self):
        d = decide(predicted_depletion_mm=15.0, et0_forecast_mm=9.0, **KW)
        hot = {c["key"] for c in retrieve(d, 9.0, sandy=True)}
        mild = {c["key"] for c in retrieve(d, 3.0, sandy=False)}
        assert "high_demand_adjustment" in hot
        assert "high_demand_adjustment" not in mild

    def test_every_citation_names_a_source(self):
        d = decide(predicted_depletion_mm=15.0, et0_forecast_mm=9.0, **KW)
        e = explain_decision(d, 9.0, "Turf", "Sand")
        assert all(c["source"].startswith("FAO-56") for c in e.citations)

    def test_explanation_runs_with_no_model(self):
        d = decide(predicted_depletion_mm=15.0, et0_forecast_mm=8.0, **KW)
        assert explain_decision(d, 8.0, "Turf", "Sand").engine == "offline"
