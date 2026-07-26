"""Tests for the grounded LLM explanation layer.

The point of these tests is that they do not need a language model. The
grounding guarantee is a property of the verifier, and a verifier that can
only be tested by sampling a stochastic model is not a guarantee at all - so
the model is stubbed and the verifier is tested directly and adversarially.

One live test exercises the real Ollama path and skips when it is unavailable,
which is the normal case on CI and on a fresh clone.
"""
from __future__ import annotations

import pytest

from irrigation.decision.policy import decide
from irrigation.explain.advisor import retrieve
from irrigation.explain.llm import (
    OllamaExplainer,
    build_prompt,
    facts_from_context,
    normalize,
    ungrounded_numbers,
)
from irrigation.physics.crop import CROPS, SOILS

TURF, SAND = CROPS["turfgrass"], SOILS["sand"]


@pytest.fixture
def context() -> dict:
    decision = decide(
        predicted_depletion_mm=13.8, et0_forecast_mm=8.4,
        crop=TURF, soil=SAND, root_depth_m=0.5, kc=0.85,
    )
    return {
        "decision": decision,
        "et0_forecast_mm": 8.4,
        "crop_name": "turfgrass",
        "soil_name": "sand",
        "citations": retrieve(decision, 8.4, True),
    }


@pytest.fixture
def facts(context):
    return facts_from_context(context)


class StubOllama(OllamaExplainer):
    """An explainer whose model returns a fixed string, or raises."""

    def __init__(self, output: str | None = None, error: Exception | None = None):
        super().__init__(model="stub")
        self._output = output
        self._error = error

    def _generate(self, context, facts) -> str:
        if self._error is not None:
            raise self._error
        return normalize(self._output or "")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
class TestNormalize:
    def test_folds_unicode_punctuation_to_ascii(self):
        assert normalize("root‑zone ‘wet’") == "root-zone 'wet'"

    def test_output_is_pure_ascii(self):
        """A cp1252 console raises on non-ASCII, which would crash the demo."""
        result = normalize("ET₀ ≈ 8.4 mm day⁻¹ × 2")
        assert result.encode("ascii")

    def test_collapses_whitespace(self):
        assert normalize("a  b\n\nc") == "a b c"


# --------------------------------------------------------------------------
# The grounding verifier
# --------------------------------------------------------------------------
class TestGrounding:
    def test_supplied_numbers_pass(self, facts, context):
        text = "Depletion is 13.8 mm of 35.0 mm total available."
        assert ungrounded_numbers(text, facts, context["citations"]) == []

    def test_rounded_numbers_pass(self, facts, context):
        """Presenting 13.84 as 13.8 or 14 is good writing, not fabrication."""
        assert ungrounded_numbers("about 14 mm", facts, context["citations"]) == []

    def test_invented_number_is_caught(self, facts, context):
        """The failure this whole layer exists to prevent.

        A fluent sentence with one wrong figure is indistinguishable from a
        correct one by eye, and it is the output that would actually get
        someone's turf killed.
        """
        found = ungrounded_numbers("Apply 22.7 mm today.", facts, context["citations"])
        assert found == [22.7]

    def test_arithmetic_the_model_did_itself_is_caught(self, facts, context):
        """13.8 and 12.7 are both supplied; their difference is not.

        This is the subtle case. Every input is legitimate, the operation is
        trivial, and the result is still an unverified number produced by a
        language model rather than by the physics layer.
        """
        found = ungrounded_numbers(
            "Depletion exceeds RAW by 1.1 mm.", facts, context["citations"]
        )
        assert 1.1 in found

    def test_superscript_digits_cannot_smuggle_a_number(self, facts, context):
        """The reason normalization runs before verification, not after.

        Unicode superscripts are digits that the number regex does not match.
        Without folding, this string would verify clean while stating a
        fabricated application depth.
        """
        raw = "Apply ²².⁷ mm today."
        assert ungrounded_numbers(normalize(raw), facts, context["citations"]) == [22.7]

    def test_unit_exponents_are_not_treated_as_quantities(self, facts, context):
        """`mm day-1` and `MJ m-2` are units. Flagging them would fail
        every explanation written in SI form."""
        text = "Reference ET0 is 8.4 mm day-1 with radiation in MJ m-2 day-1."
        assert ungrounded_numbers(text, facts, context["citations"]) == []

    def test_numbers_quoted_from_citations_pass(self, facts, context):
        """Sandy soils hold ~70 mm/m per FAO-56 Table 19 - grounded in a
        source rather than in a computation, which is equally acceptable."""
        text = "Sandy soils hold roughly 70 mm of available water per metre."
        assert ungrounded_numbers(text, facts, context["citations"]) == []


# --------------------------------------------------------------------------
# Fact sheet and prompt
# --------------------------------------------------------------------------
class TestFacts:
    def test_facts_come_from_the_decision_object(self, context, facts):
        values = {fact.value for fact in facts}
        assert context["decision"].predicted_depletion_mm in values
        assert context["decision"].raw_mm in values
        assert context["decision"].taw_mm in values

    def test_application_depth_offered_only_when_irrigating(self, context):
        hold = decide(
            predicted_depletion_mm=2.0, et0_forecast_mm=5.0,
            crop=TURF, soil=SAND, root_depth_m=0.5, kc=0.85,
        )
        assert hold.action == "hold"
        labels = {f.label for f in facts_from_context({**context, "decision": hold})}
        assert "recommended application depth" not in labels

    def test_prompt_contains_every_fact_and_citation(self, context, facts):
        prompt = build_prompt(context, facts)
        for fact in facts:
            assert fact.label in prompt
        for citation in context["citations"]:
            assert citation["source"] in prompt


# --------------------------------------------------------------------------
# Fallback behaviour
# --------------------------------------------------------------------------
class TestFallback:
    def test_generation_failure_falls_back(self, context):
        result = StubOllama(error=ConnectionError("ollama not running")).explain(context)
        assert result.engine.startswith("offline (fallback:")
        assert "ConnectionError" in result.engine

    def test_empty_output_falls_back(self, context):
        result = StubOllama(output="   ").explain(context)
        assert "empty output" in result.engine

    def test_ungrounded_output_is_discarded(self, context):
        """Discarded, not repaired. There is no safe way to edit a number out
        of a sentence whose meaning depended on it."""
        result = StubOllama(output="Apply 99.9 mm right away.").explain(context)
        assert result.engine.startswith("offline (fallback:")
        assert "99.9" in result.engine
        assert "99.9" not in result.body

    def test_fallback_still_produces_a_usable_decision(self, context):
        """Degrade the wording, never the reasoning."""
        result = StubOllama(error=RuntimeError("boom")).explain(context)
        assert "IRRIGATE" in result.headline
        assert f"{context['decision'].depth_mm:.1f}" in result.headline
        assert result.citations == context["citations"]

    def test_verified_output_is_kept_and_attributed(self, context):
        text = "Depletion of 13.8 mm exceeds the 12.7 mm readily available limit."
        result = StubOllama(output=text).explain(context)
        assert result.engine == "ollama:stub"
        assert "13.8 mm" in result.body

    def test_headline_is_templated_even_on_the_model_path(self, context):
        """The one number that must survive every failure mode.

        The body is the model's; the headline carries the actionable figure
        and is assembled from the decision object regardless.
        """
        result = StubOllama(output="Water is needed soon.").explain(context)
        assert result.engine == "ollama:stub"
        assert f"IRRIGATE {context['decision'].depth_mm:.1f} mm" in result.headline


# --------------------------------------------------------------------------
# Live model
# --------------------------------------------------------------------------
def test_live_ollama_produces_grounded_output(context):
    """Skipped unless Ollama is running with the configured model pulled."""
    explainer = OllamaExplainer()
    if not explainer.available():
        pytest.skip(f"Ollama model {explainer.model} unavailable")

    result = explainer.explain(context)
    assert result.engine.startswith("ollama:"), result.engine
    assert result.body.strip()
    assert result.body.encode("ascii")
