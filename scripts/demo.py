#!/usr/bin/env python3
"""End-to-end demo. No API keys, no network, no hardware.

    python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irrigation.climate.dubai import MONTH_STARTS, normals_day  # noqa: E402
from irrigation.climate.et0_series import et0_for_day  # noqa: E402
from irrigation.decision.policy import CostModel, decide  # noqa: E402
from irrigation.explain.advisor import explain_decision  # noqa: E402
from irrigation.models.evaluate import render_table, run_comparison  # noqa: E402
from irrigation.physics.crop import CROPS, SOILS  # noqa: E402
from irrigation.physics.penman_monteith import (  # noqa: E402
    actual_vapour_pressure_from_dewpoint, actual_vapour_pressure_from_rh,
    eto_penman_monteith, soil_heat_flux_monthly, solar_radiation_from_sunshine,
    solar_radiation_from_temp_range, wind_speed_at_2m,
)

RULE = "─" * 86
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def banner(t: str) -> None:
    print(f"\n{RULE}\n  {t}\n{RULE}")


def main() -> int:
    crop, soil = CROPS["turfgrass"], SOILS["sand"]
    root, kc = 0.5, crop.kc_mid

    banner("1 · PHYSICS VALIDATED AGAINST FAO-56's OWN WORKED EXAMPLES")
    rs = solar_radiation_from_sunshine(8.5, 13.7333, 105)
    a = eto_penman_monteith(34.8, 25.6, 2.85, 2.0, rs, 13.7333, 2.0, 105,
                            soil_heat_flux_monthly(30.2, 29.2)).et0_mm_day
    u2 = wind_speed_at_2m(10 / 3.6, 10.0)
    ea = actual_vapour_pressure_from_rh(21.5, 12.3, 84, 63)
    rs = solar_radiation_from_sunshine(9.25, 50.80, 187)
    b = eto_penman_monteith(21.5, 12.3, ea, u2, rs, 50.80, 100.0, 187).et0_mm_day
    ea = actual_vapour_pressure_from_dewpoint(14.8)
    rs = solar_radiation_from_temp_range(26.6, 14.8, 45.7167, 196)
    c = eto_penman_monteith(26.6, 14.8, ea, 2.0, rs, 45.7167, 200.0, 196).et0_mm_day
    print(f"  Example 17  Bangkok, monthly data      ET0 = {a:.2f}   published 5.72")
    print(f"  Example 18  Brussels, daily + RH       ET0 = {b:.2f}   published 3.88")
    print(f"  Example 20  Lyon, temperature only     ET0 = {c:.2f}   published 4.56")

    banner("2 · DUBAI ET0 CLIMATOLOGY FROM THE VALIDATED MODEL")
    total = 0.0
    for i, s in enumerate(MONTH_STARTS):
        v = et0_for_day(normals_day(s + 14)).et0_mm_day
        total += v * DAYS_IN_MONTH[i]
        print(f"    {MONTHS[i]}  {v:5.2f} mm/day  {'█' * int(v * 4)}")
    print(f"\n  Annual reference ET  {total:,.0f} mm    published UAE range ~2000-2200 mm")

    banner("3 · DECISIONS EXPLAINED - THE SAME DEPLETION, TWO SEASONS")
    for label, doy, depletion in (("mid-July", 196, 9.0), ("mid-January", 15, 9.0)):
        w = normals_day(doy)
        et0 = et0_for_day(w).et0_mm_day
        d = decide(predicted_depletion_mm=depletion, et0_forecast_mm=et0,
                   crop=crop, soil=soil, root_depth_m=root, kc=kc)
        print(f"\n  [{label}]  ET0 {et0:.1f} mm/day, root zone {depletion:.0f} mm depleted")
        print(explain_decision(d, et0, crop.name, soil.name).render())
    print("\n  Identical soil state, opposite decisions. Evaporative demand - not")
    print("  moisture alone - is what determines whether today needs water.")

    banner("4 · PREDICTOR COMPARISON OVER A 120-DAY SUMMER SEASON")
    print(f"  Crop {crop.name} · Soil {soil.name} · TAW "
          f"{soil.total_available_water_mm(root):.0f} mm · 1 May to 28 Aug\n")
    results = run_comparison(crop, soil, root_depth_m=root, kc=kc)
    print(render_table(results))

    # Found by score, never by name. Three supervised models sit within ~1% of
    # each other, so which one is most accurate changes with a library version
    # - and a hardcoded name silently prints the wrong number when it does.
    best_rmse = min(results, key=lambda r: r.depletion_rmse)
    phys = next(r for r in results if r.predictor_name.startswith("Physics ("))
    print(f"\n  {best_rmse.predictor_name} has the LOWEST prediction error "
          f"({best_rmse.depletion_rmse:.2f} vs {phys.depletion_rmse:.2f} mm RMSE)")
    print(f"  and the HIGHER operating cost "
          f"({best_rmse.total_cost:,.0f} vs {phys.total_cost:,.0f} AED).")
    print(f"  It over-applies {best_rmse.water_applied_mm - phys.water_applied_mm:,.0f} mm of water, "
          f"{best_rmse.drainage_mm - phys.drainage_mm:,.0f} mm of which drains away.")
    print("  RMSE weights over- and under-estimation equally. The field does not.")

    banner("5 · WHERE THE RANKING FLIPS")
    # Columns are derived from the results, so adding a predictor cannot leave
    # the header describing a different table from the one printed below it.
    # Blind truncation produced "Physics + se" and "Gradient boo", so the
    # abbreviations are curated, with truncation only as a fallback.
    abbreviations = {
        "Physics (FAO-56 balance)": "Physics",
        "Sensor only": "Sensor",
        "Physics + sensor fusion": "Fusion",
        "Random forest": "RF",
        "Gradient boosting": "GBM",
        "XGBoost": "XGBoost",
    }
    short_names = [
        abbreviations.get(r.predictor_name, r.predictor_name[:12]) for r in results
    ]
    print(f"  {'stress AED/mm':>14} {'ratio':>7}   " +
          "".join(f"{n:>13}" for n in short_names) + "   winner")
    for stress in (2.5, 15.0, 30.0, 60.0, 120.0):
        cm = CostModel(stress_cost_per_mm_deficit=stress)
        rs_ = run_comparison(crop, soil, root_depth_m=root, kc=kc, cost_model=cm)
        costs = [r.total_cost for r in rs_]
        winner_name = rs_[costs.index(min(costs))].predictor_name
        win = abbreviations.get(winner_name, winner_name)
        print(f"  {stress:>14.1f} {stress / cm.water_cost_per_mm:>6.0f}x   " +
              "".join(f"{c:>13,.0f}" for c in costs) + f"   {win}")
    print("\n  Machine learning earns its place only where stress is catastrophically")
    print("  expensive and its systematic over-watering becomes cheap insurance.")
    print("  At every ordinary cost ratio, a 1998 water balance wins - for free.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
