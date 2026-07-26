"""
Crop coefficients and soil properties (FAO-56 Chapters 6-8).

Values are taken from FAO-56's published tables rather than tuned. Where a
Dubai-specific choice departs from the table default, the reason is recorded
in the dataclass - a number nobody can trace is a number nobody can maintain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GrowthStage = Literal["initial", "development", "mid", "late"]


@dataclass(frozen=True)
class Crop:
    """
    Crop water-use characteristics.

    kc_* are FAO-56 Table 12 single crop coefficients.
    depletion_fraction (p) is Table 22 - the fraction of available water that
    can be depleted before the crop experiences stress.
    """

    name: str
    kc_initial: float
    kc_mid: float
    kc_end: float
    root_depth_max_m: float
    depletion_fraction: float
    notes: str = ""

    def kc(self, stage: GrowthStage) -> float:
        return {
            "initial": self.kc_initial,
            "development": (self.kc_initial + self.kc_mid) / 2.0,
            "mid": self.kc_mid,
            "late": self.kc_end,
        }[stage]

    def adjusted_depletion_fraction(self, et0_mm_day: float) -> float:
        """
        Eq. 83 - p adjusted for evaporative demand.

        Tabulated p assumes ET0 ~ 5 mm/day. Dubai summer runs 8-10 mm/day, at
        which the crop stresses at a *lower* depletion than the table implies:
        water moves to the leaf slower than the atmosphere pulls it away.
        Using the unadjusted table value in a Gulf summer is a common and
        expensive mistake - it schedules irrigation later than the plant can
        tolerate.
        """
        p = self.depletion_fraction + 0.04 * (5.0 - et0_mm_day)
        return max(0.1, min(0.8, p))


# FAO-56 Table 12 / Table 22. Species chosen for Dubai landscape and
# protected-agriculture contexts rather than field crops.
DATE_PALM = Crop(
    name="Date palm",
    kc_initial=0.90, kc_mid=0.95, kc_end=0.95,
    root_depth_max_m=1.5, depletion_fraction=0.50,
    notes="FAO-56 Table 12. Dominant UAE tree crop; Kc near-constant year round.",
)

TURFGRASS_WARM = Crop(
    name="Warm-season turfgrass",
    kc_initial=0.80, kc_mid=0.85, kc_end=0.85,
    root_depth_max_m=0.5, depletion_fraction=0.50,
    notes="FAO-56 Table 12 warm season turf. Dominant Dubai landscape surface.",
)

TOMATO_GREENHOUSE = Crop(
    name="Tomato (protected)",
    kc_initial=0.60, kc_mid=1.15, kc_end=0.80,
    root_depth_max_m=1.0, depletion_fraction=0.40,
    notes="FAO-56 Table 12 tomato. Protected cropping is a growing UAE segment.",
)

CROPS: dict[str, Crop] = {
    "date_palm": DATE_PALM,
    "turfgrass": TURFGRASS_WARM,
    "tomato": TOMATO_GREENHOUSE,
}


@dataclass(frozen=True)
class Soil:
    """
    Soil water-holding characteristics.

    field_capacity and wilting_point are volumetric water contents [m3/m3] -
    the same units a capacitance soil-moisture probe reports, so sensor
    readings map onto this model without conversion.
    """

    name: str
    field_capacity: float
    wilting_point: float
    saturation: float
    infiltration_mm_hr: float
    notes: str = ""

    @property
    def available_water_fraction(self) -> float:
        return self.field_capacity - self.wilting_point

    def total_available_water_mm(self, root_depth_m: float) -> float:
        """Eq. 82 - TAW [mm] over the root zone."""
        return 1000.0 * self.available_water_fraction * root_depth_m

    def readily_available_water_mm(self, root_depth_m: float, p: float) -> float:
        """Eq. 83 - RAW [mm]. Depletion beyond this induces stress."""
        return p * self.total_available_water_mm(root_depth_m)


# Typical UAE profiles. Sandy soil dominates and is the hard case: very low
# water-holding capacity means the irrigation decision has little margin.
SANDY = Soil(
    name="Sand (typical UAE)",
    field_capacity=0.12, wilting_point=0.05, saturation=0.36,
    infiltration_mm_hr=50.0,
    notes="FAO-56 Table 19 sand. AWC ~70 mm/m - roughly a third of a loam.",
)

SANDY_LOAM = Soil(
    name="Sandy loam (amended)",
    field_capacity=0.21, wilting_point=0.10, saturation=0.43,
    infiltration_mm_hr=25.0,
    notes="FAO-56 Table 19. Represents an amended landscape bed.",
)

SOILS: dict[str, Soil] = {"sand": SANDY, "sandy_loam": SANDY_LOAM}
