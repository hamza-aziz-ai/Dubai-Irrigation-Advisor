"""Grounded explanation of an irrigation decision.

Same discipline as the physics layer: the language model never computes.
Every number in an explanation is passed in already calculated, and the
retrieval layer supplies the agronomic justification from a citable source.
The model's only job is to assemble those into a sentence a grounds manager
can act on.

An offline template engine implements the identical interface, so the system
explains its decisions with no API key and no network - and an LLM outage
degrades the wording, never the reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..decision.policy import IrrigationDecision

# Minimal citable knowledge base. In production this is the RAG corpus -
# FAO-56, local extension guidance, the client's own agronomy notes - indexed
# in a vector store. The retrieval contract is the same either way: a decision
# fact maps to a passage with a source.
KNOWLEDGE: dict[str, dict[str, str]] = {
    "raw_trigger": {
        "text": (
            "Irrigation should be scheduled before root-zone depletion exceeds "
            "readily available water (RAW). Beyond RAW the stress coefficient Ks "
            "falls below 1 and transpiration is reduced below its potential rate."
        ),
        "source": "FAO-56 Ch. 8, Eq. 84",
    },
    "high_demand_adjustment": {
        "text": (
            "Tabulated depletion fractions assume ET0 near 5 mm/day. Where "
            "evaporative demand is higher the fraction must be reduced, because "
            "water moves to the root surface more slowly than the atmosphere "
            "removes it from the leaf."
        ),
        "source": "FAO-56 Ch. 8, Eq. 83",
    },
    "sandy_soil": {
        "text": (
            "Sandy soils hold roughly 70 mm of available water per metre of "
            "depth, against 140-200 mm for loams. Irrigation must therefore be "
            "lighter and more frequent; a large application drains below the "
            "root zone rather than being stored."
        ),
        "source": "FAO-56 Ch. 8, Table 19",
    },
    "drainage_loss": {
        "text": (
            "Water applied in excess of field capacity percolates below the root "
            "zone within a day and is unavailable to the crop. On desalinated "
            "supply this is a direct and unrecoverable cost."
        ),
        "source": "FAO-56 Ch. 8, Eq. 85",
    },
}


def retrieve(decision: IrrigationDecision, et0_mm: float, sandy: bool) -> list[dict[str, str]]:
    """Select the passages that justify this specific decision."""
    keys = ["raw_trigger"]
    if et0_mm > 6.0:
        keys.append("high_demand_adjustment")
    if sandy:
        keys.append("sandy_soil")
    if decision.action == "irrigate" and decision.depth_mm > decision.taw_mm:
        keys.append("drainage_loss")
    return [{"key": k, **KNOWLEDGE[k]} for k in keys]


@dataclass
class Explanation:
    headline: str
    body: str
    citations: list[dict[str, str]]
    engine: str

    def render(self) -> str:
        cites = "\n".join(f"    [{c['key']}] {c['source']}" for c in self.citations)
        return f"{self.headline}\n{self.body}\n  Grounded in:\n{cites}"


class ExplanationEngine(Protocol):
    """What an explanation engine must provide.

    `name` is declared read-only rather than as a plain attribute. A mutable
    attribute in a Protocol requires implementations to expose a settable one,
    and `OllamaExplainer.name` is a computed property - it derives the model
    tag - so it could not satisfy the stricter form. A read-only declaration is
    also the honest one: nothing assigns to this.
    """

    @property
    def name(self) -> str: ...

    def explain(self, context: dict[str, Any]) -> Explanation: ...


class OfflineExplainer:
    """Deterministic explanation from computed facts. No model required."""

    name = "offline"

    def explain(self, context: dict[str, Any]) -> Explanation:
        d: IrrigationDecision = context["decision"]
        et0 = context["et0_forecast_mm"]
        depletion_pct = (
            d.predicted_depletion_mm / d.taw_mm * 100 if d.taw_mm else 0.0
        )

        if d.action == "irrigate":
            headline = (
                f"IRRIGATE {d.depth_mm:.1f} mm today "
                f"({context['crop_name']}, {context['soil_name']})"
            )
            body = (
                f"  Root zone is {d.predicted_depletion_mm:.1f} mm depleted of "
                f"{d.taw_mm:.1f} mm total available ({depletion_pct:.0f}%). "
                f"Tomorrow's reference ET is {et0:.1f} mm/day, which would carry "
                f"depletion past the readily available limit of {d.raw_mm:.1f} mm "
                f"and reduce transpiration below potential."
            )
        else:
            headline = f"HOLD - no irrigation today ({context['crop_name']})"
            body = (
                f"  Root zone is {d.predicted_depletion_mm:.1f} mm depleted of "
                f"{d.taw_mm:.1f} mm ({depletion_pct:.0f}%), and tomorrow's "
                f"reference ET of {et0:.1f} mm/day keeps it within the "
                f"{d.raw_mm:.1f} mm readily available limit. Applying water now "
                f"would drain below the root zone rather than be stored."
            )

        return Explanation(
            headline=headline, body=body,
            citations=context["citations"], engine=self.name,
        )


def build_explainer() -> ExplanationEngine:
    """Resolve an explanation engine. Offline is the default and the fallback."""
    return OfflineExplainer()


def explain_decision(
    decision: IrrigationDecision,
    et0_forecast_mm: float,
    crop_name: str,
    soil_name: str,
    sandy: bool = True,
    engine: ExplanationEngine | None = None,
) -> Explanation:
    engine = engine or build_explainer()
    return engine.explain({
        "decision": decision,
        "et0_forecast_mm": et0_forecast_mm,
        "crop_name": crop_name,
        "soil_name": soil_name,
        "citations": retrieve(decision, et0_forecast_mm, sandy),
    })
