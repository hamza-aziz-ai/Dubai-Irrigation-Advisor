from __future__ import annotations

import pytest

from irrigation.decision.policy import CostModel, decide
from irrigation.explain.advisor import explain_decision, retrieve
from irrigation.physics.crop import CROPS, SOILS
from irrigation.models.evaluate import run_comparison
from irrigation.models.predictors import PhysicsBalance, SensorDirect
from irrigation.models.simulate import simulate_season

TURF, SAND = CROPS["turfgrass"], SOILS["sand"]
KW = dict(crop=TURF, soil=SAND, root_depth_m=0.5, kc=0.85)

# The supervised models, referred to as a set rather than individually.
# They sit within ~1% of each other on cost, so any test that names one of
# them specifically is asserting noise.
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

    def test_best_rmse_does_not_win_on_cost(self, results):
        """THE headline result.

        Gradient boosting has the lowest depletion RMSE and does NOT have the
        lowest operating cost, because RMSE weights over- and under-estimation
        equally and the field does not. If this ever inverts, the claim in the
        README needs revisiting rather than the test being relaxed.
        """
        best_rmse = min(results, key=lambda r: r.depletion_rmse)
        best_cost = min(results, key=lambda r: r.total_cost)
        assert best_rmse.predictor_name != best_cost.predictor_name

    def test_physics_wins_at_the_default_cost_ratio(self, results):
        best_cost = min(results, key=lambda r: r.total_cost)
        assert best_cost.predictor_name.startswith("Physics (")

    def test_ml_wins_once_stress_is_catastrophically_expensive(self):
        """The crossover is real - ML's over-watering becomes cheap insurance.

        Which of the supervised models wins is not the claim and is not
        asserted. They sit within about 1% of each other on cost, so the
        ordering among them is noise; pinning one name here would make the
        test fail on a library upgrade while the finding held.
        """
        extreme = CostModel(stress_cost_per_mm_deficit=120.0)
        results = run_comparison(TURF, SAND, root_depth_m=0.5, kc=0.85,
                                 cost_model=extreme)
        winner = min(results, key=lambda r: r.total_cost)
        assert winner.predictor_name in SUPERVISED_MODELS

    def test_every_supervised_model_overwaters(self, results):
        """The systematic error is a property of the framing, not of one model.

        Random forest, gradient boosting and XGBoost are three different
        learning algorithms - bagged trees that cannot extrapolate, and two
        boosters that can. Trained on the same features against a squared-error
        objective, all three land on the same side of the true depletion and
        all three drain more water past the root zone than the physics
        baseline does.

        That is what makes the headline result a statement about the objective
        rather than about a model choice: swapping the estimator does not fix
        it, because the estimator was never what was wrong.
        """
        physics = next(r for r in results if r.predictor_name.startswith("Physics ("))
        supervised = [r for r in results if r.predictor_name in SUPERVISED_MODELS]
        assert len(supervised) == 3
        for model in supervised:
            assert model.depletion_bias > 0.0, model.predictor_name
            assert model.water_applied_mm > physics.water_applied_mm, model.predictor_name
            assert model.drainage_mm > physics.drainage_mm, model.predictor_name


class TestSimulatorIntegrity:
    def test_predictor_cannot_observe_true_state(self):
        """Observation carries only field-available information."""
        from irrigation.models.predictors import Observation
        fields = set(Observation.__dataclass_fields__)
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


class TestExplanations:
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
