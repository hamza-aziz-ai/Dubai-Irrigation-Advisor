# Data provenance

Every number in this project comes from one of three places: a published
equation, a committed observational record, or a stated policy choice. This
document covers the second, and says plainly what it can and cannot support.

---

## NASA POWER daily record

**File:** `data/raw/nasa_power_dubai_daily.csv`
**Metadata:** `data/raw/nasa_power_dubai_daily.meta.json`
**Loader:** `src/irrigation/data/nasa_power.py`
**Refresh:** `python scripts/fetch_nasa_power.py`

|           |                                                             |
|-----------|-------------------------------------------------------------|
| Source    | NASA Langley Research Center POWER Project                  |
| Period    | 1995-01-01 to 2024-12-31                                    |
| Rows      | 10,958 days, no gaps, no fill values                        |
| Grid cell | 25.25 N, 55.33 E (cell centre), reported elevation 67 m     |
| Cell size | 0.5 deg latitude x 0.625 deg longitude (roughly 55 x 65 km) |
| Licence   | Public domain, no account or key required                   |
| Access    | One HTTP request, run by hand, cached and committed         |

### Why this source

The UAE National Center of Meteorology publishes no free bulk historical
download. Commercial weather APIs are paid and, more importantly, are not
reproducible: the same request re-issued next year returns different numbers
after a reanalysis update, with no version to pin.

NASA POWER is public domain, needs no credentials, covers 1981-present daily,
and serves every FAO-56 Penman-Monteith input *plus* a root-zone soil moisture
field. Downloading once and committing the result makes every figure in this
repository reproducible offline and stable over time.

### Fields used

| Column         | POWER parameter     | Unit         | Notes                                     |
|----------------|---------------------|--------------|-------------------------------------------|
| `tmax_c`       | `T2M_MAX`           | C            |                                           |
| `tmin_c`       | `T2M_MIN`           | C            |                                           |
| `tmean_c`      | `T2M`               | C            |                                           |
| `dewpoint_c`   | `T2MDEW`            | C            | Preferred route to ea (FAO-56 Eq. 14)     |
| `rh_mean_pct`  | `RH2M`              | %            | Mean only; extremes are derived           |
| `wind_2m_ms`   | `WS2M`              | m/s          | Already at 2 m - no Eq. 47 conversion     |
| `wind_10m_ms`  | `WS10M`             | m/s          | Retained to verify the above              |
| `solar_mj`     | `ALLSKY_SFC_SW_DWN` | MJ m-2 day-1 | Measured; preferred over Eq. 35           |
| `rainfall_mm`  | `PRECTOTCORR`       | mm/day       | Bias-corrected                            |
| `wetness_top`  | `GWETTOP`           | 0-1          | Surface layer                             |
| `wetness_root` | `GWETROOT`          | 0-1          | **Root zone - the sequence-model target** |
| `wetness_prof` | `GWETPROF`          | 0-1          | Full profile                              |

**The radiation unit is load-bearing.** POWER serves
`ALLSKY_SFC_SW_DWN` in MJ m-2 day-1 to the `AG` community and in
kWh m-2 day-1 to the `RE` community. Same parameter name, values 3.6x apart.
Reading the wrong one produces an ET0 roughly half of the truth, with a
perfectly normal-looking seasonal shape. `test_radiation_is_in_megajoules_not_kilowatt_hours`
guards this.

### Derived quantities

**RHmax and RHmin** are not published by POWER, and `DailyWeather` is defined
in terms of them. They are derived from dewpoint via `RH = 100 * ea / e0(T)`,
holding ea constant across the day - the assumption FAO-56 itself makes when
treating dewpoint as a daily constant. RHmax pairs with Tmin because air is
most saturated when coldest.

These derived values are **consistent with** ea rather than independent
evidence about it, which is precisely why `et0_for_day` uses the dewpoint
directly (Eq. 14) and never touches them.

**Sunshine hours** are set to NaN, not to a plausible number. POWER measures
no sunshine duration. A believable filler would be silently used the moment
someone removed the radiation column.

---

## What this data can and cannot support

### It can

- Establish reference evapotranspiration for regional Dubai to within a few
  per cent, cross-checked against an independent derivation from published
  climate normals (agreement: 2,275 mm/yr against 2,224 mm/yr).
- Provide a soil moisture target that is genuinely independent of this
  project's simulator, so a sequence model fitted to it is not fitted to my
  own assumptions.
- Establish that rainfall never meaningfully offsets demand in any month.

### It cannot

**Represent a specific field.** The grid cell is about 55 x 65 km. For
reference ET0 - defined over a hypothetical uniform grass surface - that scale
is appropriate. For a particular farm's microclimate it is not.

**Represent an irrigated root zone.** `GWETROOT` for this cell is dominated by
bare desert. Its dynamics are rain-driven, not irrigation-driven. It describes
how untouched Dubai sand dries and rewets, which is real and useful, and it is
**not** a depletion series for a watered plot. The sequence models in
`notebooks/02` learn the soil-atmosphere response; they are not drop-in
replacements for the depletion predictors in `models/evaluate.py`, and
`src/irrigation/models/sequence.py` says so at the top.

**Be treated as measurement.** POWER combines satellite irradiance with the
MERRA-2 reanalysis. `GWETROOT` in particular is a land-surface *model* output.
Independent of this project, but modelled.

**Validate the sensor simulator.** There is no in-situ probe data here. The
sensor pathologies in `climate/sensor.py` - calibration offset, salinity
drift, dropout, quantisation - are modelled from the literature on why field
deployments fail, not fitted to logged failures.

---

## Upgrade path: NASA SMAP

`GWETROOT` is a modelled product. A direct satellite retrieval exists: **SMAP
L4 Global Root Zone Soil Moisture**, 9 km, 3-hourly, 2015-present.

It is not the default here because it requires a free NASA Earthdata login,
and this project's working assumption is that anyone who clones the repository
can reproduce every number without registering for anything.

To use it instead: register at `urs.earthdata.nasa.gov`, configure `~/.netrc`,
and write a loader emitting the same `PowerRecord` shape. Nothing above the
data layer changes.

---

## Published constants

Crop coefficients, depletion fractions and soil water-holding properties are
cited individually in [AGRONOMY.md](AGRONOMY.md). Dubai monthly climate
normals in `climate/dubai.py` are approximate published values used to place
the simulation in the right regime; they are not station data, and the NASA
record above is what validates them.

---

## Citation

> These data were obtained from the NASA Langley Research Center POWER
> Project, funded through the NASA Earth Science Directorate Applied Science
> Program. <https://power.larc.nasa.gov/>
