# Agronomy notes

Every constant in this project traces to a published source. This file is the
trace, so a reviewer can check the science independently of the code.

## Reference evapotranspiration

FAO-56 Penman-Monteith (Eq. 6) is the sole standard method. Implemented in
`physics/penman_monteith.py`, with equation numbers in the docstrings.

Choices worth noting:

- **`es` from the mean of `e°(Tmax)` and `e°(Tmin)`, not `e°(Tmean)`** (Eq. 12).
  The saturation vapour pressure curve is convex, so using the mean
  temperature systematically underestimates `es`. There is a test for this.
- **`Rs` is passed in, not computed inside `eto_penman_monteith`.** How solar
  radiation was obtained — measured, from sunshine hours (Eq. 35), or from
  temperature range (Eq. 50) — is a significant source of uncertainty. Hiding
  that choice inside the ET0 function would conceal it.
- **`G = 0` at a daily step** (Eq. 42), but `0.14 (Tmonth − Tmonth−1)` at a
  monthly step (Eq. 43). Soil heat flux is not negligible over a month.
- **Wind measured at 10 m is converted with Eq. 47.** See the README for why
  this matters more than it looks.
- **The cloudiness ratio `Rs/Rso` is clamped at 1.0.** On a very clear day
  measured `Rs` can exceed the clear-sky estimate; unclamped, `Rnl` runs away.
- **`arccos` in the sunset hour angle is clamped to [−1, 1].** Inside the
  polar circles the argument legitimately exceeds that range (polar day or
  night). Not relevant at 25 °N, but a latent crash for any other deployment.

## Crop coefficients

FAO-56 Table 12 single crop coefficients. Species chosen for UAE landscape and
protected agriculture rather than field crops:

| Crop | Kc ini | Kc mid | Kc end | Root depth | p |
|---|---:|---:|---:|---:|---:|
| Date palm | 0.90 | 0.95 | 0.95 | 1.5 m | 0.50 |
| Warm-season turfgrass | 0.80 | 0.85 | 0.85 | 0.5 m | 0.50 |
| Tomato (protected) | 0.60 | 1.15 | 0.80 | 1.0 m | 0.40 |

### The depletion fraction adjustment matters more here than elsewhere

Tabulated `p` values (Table 22) assume ET0 near **5 mm/day**. Dubai summer runs
**8–9 mm/day**. FAO-56 Eq. 83 corrects for this:

```
p_adjusted = p_table + 0.04 (5 − ET0)
```

At ET0 = 8.5 this drops turfgrass `p` from 0.50 to 0.36 — RAW falls from
17.5 mm to 12.7 mm on sand at 0.5 m rooting depth. **The crop stresses at a
lower depletion than the table implies**, because water moves to the root
surface more slowly than the atmosphere removes it from the leaf.

Using the unadjusted table value in a Gulf summer schedules irrigation roughly
a third of a RAW too late. It is a common and expensive mistake.

## Soil water

FAO-56 Table 19. Sand dominates UAE soils and is the hard case:

| Soil | Field capacity | Wilting point | AWC | TAW at 0.5 m |
|---|---:|---:|---:|---:|
| Sand | 0.12 | 0.05 | 70 mm/m | 35 mm |
| Sandy loam | 0.21 | 0.10 | 110 mm/m | 55 mm |

35 mm of total available water against 8.5 mm/day of demand is roughly **four
days of buffer at full profile** — and the usable fraction is RAW, about
12.7 mm, or a day and a half. This is why the decision has so little margin,
why irrigation must be light and frequent, and why a large application drains
rather than being stored.

The balance is kept in **depletion `Dr` [mm]** rather than volumetric water
content: it is additive over a day, it maps directly onto "how many mm do I
apply", and it is the form Eq. 85 is written in. Conversion back to m³/m³ is
available for comparison against probe readings.

Water is added before evapotranspiration is removed, because irrigation
applied in the morning is available to the crop the same day. Reversing the
order biases toward over-irrigation.

## Stress coefficient

Eq. 84. `Ks = 1` while depletion stays within RAW, then falls linearly to zero
at TAW. That piecewise-linear form is what makes under-watering *nonlinear in
consequence* — the first millimetre past RAW costs little, the last costs the
crop — and it is the physical basis for the asymmetric cost model.

## What is not modelled

- Dual crop coefficient (separate evaporation and transpiration, Eq. 69). The
  single Kc is adequate at a daily step for established cover; it is not for
  bare soil after planting.
- Root growth through the season — rooting depth is fixed.
- Salinity leaching requirement. Relevant in the UAE and would *increase*
  applied depth beyond what this model recommends.
- Capillary rise from a water table (assumed zero — reasonable for sand).
- Spatial variability, canopy microclimate, and irrigation non-uniformity
  beyond a single scalar efficiency term.
