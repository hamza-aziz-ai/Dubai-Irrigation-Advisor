"""Root-zone soil water balance (FAO-56 Chapter 8).

The state variable is root-zone depletion Dr [mm] - how far below field
capacity the root zone sits. Dr = 0 means at field capacity; Dr = TAW means
the crop has exhausted all available water.

Depletion rather than volumetric water content is the operational variable:
it is what determines irrigation depth, it is additive over a day, and it
maps directly onto the decision "how many mm do I apply".
"""
from __future__ import annotations

from dataclasses import dataclass

from .crop import Crop, Soil


@dataclass(frozen=True)
class WaterBalanceState:
    depletion_mm: float          # Dr - below field capacity
    taw_mm: float                # total available water
    raw_mm: float                # readily available water (no stress below this)
    ks: float                    # water stress coefficient [0-1]
    etc_mm: float                # actual crop evapotranspiration
    etc_potential_mm: float      # unstressed crop evapotranspiration
    drainage_mm: float           # deep percolation lost below the root zone
    runoff_mm: float             # not infiltrated
    stressed: bool

    @property
    def depletion_fraction(self) -> float:
        return self.depletion_mm / self.taw_mm if self.taw_mm else 0.0

    def volumetric_water_content(self, soil: Soil, root_depth_m: float) -> float:
        """Convert depletion back to the units a probe reports [m3/m3]."""
        deficit = self.depletion_mm / (1000.0 * root_depth_m) if root_depth_m else 0.0
        return max(soil.wilting_point, soil.field_capacity - deficit)


def water_stress_coefficient(depletion_mm: float, taw_mm: float, raw_mm: float) -> float:
    """Eq. 84 - Ks [0-1].

    Ks = 1 while depletion stays within RAW; beyond that transpiration falls
    linearly to zero at TAW. This piecewise-linear form is what makes
    under-watering nonlinear in consequence: the first mm past RAW costs
    little, the last costs the crop.
    """
    if taw_mm <= 0:
        return 0.0
    if depletion_mm <= raw_mm:
        return 1.0
    remaining = taw_mm - raw_mm
    if remaining <= 0:
        return 0.0
    return max(0.0, (taw_mm - depletion_mm) / remaining)


def step(
    depletion_mm: float,
    et0_mm: float,
    crop: Crop,
    soil: Soil,
    root_depth_m: float,
    kc: float,
    irrigation_mm: float = 0.0,
    rainfall_mm: float = 0.0,
    irrigation_efficiency: float = 0.90,
) -> WaterBalanceState:
    """Advance the balance one day. Eq. 85.

        Dr(i) = Dr(i-1) - (P - RO) - I*eff + ETc + DP

    Order matters: water is added before evapotranspiration is removed,
    because irrigation applied in the morning is available to the crop that
    same day. Reversing it biases the model toward over-irrigating.
    """
    p = crop.adjusted_depletion_fraction(et0_mm)
    taw = soil.total_available_water_mm(root_depth_m)
    raw = soil.readily_available_water_mm(root_depth_m, p)

    effective_irrigation = irrigation_mm * irrigation_efficiency

    # Rainfall beyond what the profile can absorb runs off. Dubai rainfall is
    # rare but intense, so this is not a negligible term on the days it falls.
    capacity = depletion_mm
    total_in = effective_irrigation + rainfall_mm
    runoff = max(0.0, rainfall_mm - soil.infiltration_mm_hr * 2.0)
    total_in -= runoff

    after_input = depletion_mm - total_in

    # Anything that fills past field capacity drains below the root zone.
    drainage = max(0.0, -after_input)
    after_input = max(0.0, after_input)

    ks = water_stress_coefficient(after_input, taw, raw)
    etc_potential = et0_mm * kc
    etc_actual = etc_potential * ks

    new_depletion = min(after_input + etc_actual, taw)

    return WaterBalanceState(
        depletion_mm=new_depletion,
        taw_mm=taw,
        raw_mm=raw,
        ks=ks,
        etc_mm=etc_actual,
        etc_potential_mm=etc_potential,
        drainage_mm=drainage,
        runoff_mm=runoff,
        stressed=new_depletion > raw,
    )
