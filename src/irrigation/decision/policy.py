"""
Irrigation decisions under asymmetric cost.

THE CENTRAL ARGUMENT OF THIS PROJECT

Irrigation is not a prediction problem, it is a decision problem, and the two
costs are not symmetric:

  * Over-irrigating by 5 mm wastes water and leaches nutrients. In Dubai that
    is expensive - most irrigation water is desalinated - but it is recoverable.
  * Under-irrigating by 5 mm in a July where ET0 is 8.5 mm/day and sandy soil
    holds ~35 mm total pushes the crop into stress within a day. Ks falls,
    growth stops, and severe depletion is not recoverable at all.

A model can therefore have excellent RMSE and still be operationally useless,
because RMSE weights both errors equally and the field does not. Everything
here is evaluated on decision cost, and prediction error is reported only as
a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..physics.crop import Crop, Soil

Action = Literal["irrigate", "hold"]


@dataclass(frozen=True)
class CostModel:
    """
    Costs in AED per hectare-day.

    water_cost_per_mm reflects desalinated supply, which is the relevant
    marginal source for UAE landscape and protected agriculture.

    stress_cost_per_mm_deficit is the penalty per mm of depletion beyond RAW.
    The 12:1 ratio against water is a policy choice, not a measurement - it
    encodes "a stressed crop is much worse than a wasted litre". It is exposed
    as a parameter precisely because an agronomist should be able to argue
    with it, and because the whole evaluation is sensitive to it.
    """

    water_cost_per_mm: float = 2.50
    stress_cost_per_mm_deficit: float = 30.0
    severe_stress_multiplier: float = 4.0
    severe_stress_threshold: float = 0.80   # depletion fraction of TAW
    pumping_cost_per_event: float = 1.50

    def evaluate_day(
        self, applied_mm: float, depletion_mm: float, taw_mm: float, raw_mm: float
    ) -> dict[str, float]:
        water = applied_mm * self.water_cost_per_mm
        pumping = self.pumping_cost_per_event if applied_mm > 0 else 0.0

        deficit = max(0.0, depletion_mm - raw_mm)
        stress = deficit * self.stress_cost_per_mm_deficit
        if taw_mm > 0 and depletion_mm / taw_mm > self.severe_stress_threshold:
            stress *= self.severe_stress_multiplier

        return {
            "water_cost": water,
            "pumping_cost": pumping,
            "stress_cost": stress,
            "total_cost": water + pumping + stress,
            "deficit_mm": deficit,
        }


@dataclass(frozen=True)
class IrrigationDecision:
    action: Action
    depth_mm: float
    predicted_depletion_mm: float
    raw_mm: float
    taw_mm: float
    reason: str
    trigger_fraction: float


def decide(
    predicted_depletion_mm: float,
    et0_forecast_mm: float,
    crop: Crop,
    soil: Soil,
    root_depth_m: float,
    kc: float,
    trigger_fraction: float = 0.85,
    refill_to_fraction: float = 0.0,
    irrigation_efficiency: float = 0.90,
    max_application_mm: float | None = None,
) -> IrrigationDecision:
    """
    Decide today's irrigation from predicted depletion.

    The trigger is a fraction of RAW rather than of TAW, and it fires *before*
    RAW is reached. Waiting until depletion actually reaches RAW means the
    crop stresses during the day it is irrigated - the decision has to lead
    the deficit, not follow it.

    Tomorrow's ET0 is included so a hot day is anticipated rather than
    reacted to. This is the one place forecast data changes the decision.
    """
    p = crop.adjusted_depletion_fraction(et0_forecast_mm)
    taw = soil.total_available_water_mm(root_depth_m)
    raw = soil.readily_available_water_mm(root_depth_m, p)

    projected = predicted_depletion_mm + et0_forecast_mm * kc
    threshold = raw * trigger_fraction

    if projected < threshold:
        return IrrigationDecision(
            action="hold", depth_mm=0.0,
            predicted_depletion_mm=predicted_depletion_mm,
            raw_mm=raw, taw_mm=taw, trigger_fraction=trigger_fraction,
            reason=(
                f"Projected depletion {projected:.1f} mm stays below the "
                f"{threshold:.1f} mm trigger ({trigger_fraction:.0%} of RAW "
                f"{raw:.1f} mm). No irrigation needed."
            ),
        )

    target = raw * refill_to_fraction
    gross = max(0.0, (predicted_depletion_mm - target) / irrigation_efficiency)
    if max_application_mm is not None:
        gross = min(gross, max_application_mm)

    # Applying more than the profile can hold is drainage, not irrigation.
    gross = min(gross, taw / irrigation_efficiency)

    return IrrigationDecision(
        action="irrigate", depth_mm=gross,
        predicted_depletion_mm=predicted_depletion_mm,
        raw_mm=raw, taw_mm=taw, trigger_fraction=trigger_fraction,
        reason=(
            f"Projected depletion {projected:.1f} mm reaches the "
            f"{threshold:.1f} mm trigger. Applying {gross:.1f} mm gross "
            f"({gross * irrigation_efficiency:.1f} mm net at "
            f"{irrigation_efficiency:.0%} efficiency) to refill the root zone."
        ),
    )
