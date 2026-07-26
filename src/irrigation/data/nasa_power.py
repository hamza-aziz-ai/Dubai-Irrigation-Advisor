"""
NASA POWER daily data for Dubai - the project's real-observation source.

WHY THIS SOURCE

The brief asks for historical weather feeding a soil-moisture model. The UAE's
National Center of Meteorology does not publish a free bulk historical
download. Commercial weather APIs are both paid and unreproducible: the same
request next year returns different numbers after a reanalysis update, with no
way to pin a version.

NASA POWER solves all three problems at once. It is US-government, public
domain, needs no account or key, covers 1981-present at daily resolution, and
serves every input FAO-56 Penman-Monteith requires *plus* a root-zone soil
moisture field. One request, one provenance story, no credentials.

WHAT THE FIELDS ACTUALLY ARE

POWER is not a weather station. It is satellite-derived irradiance (NASA
CERES/SRB) combined with the MERRA-2 atmospheric reanalysis, sampled at the
grid cell containing the requested point. Two consequences worth stating
plainly rather than burying:

1. The grid cell is 0.5 deg latitude by 0.625 deg longitude - roughly 55 x 65
   km. This is regional Dubai, not a specific field. For reference ET0, which
   is defined over a hypothetical uniform grass surface anyway, that is an
   appropriate scale. For a specific farm's microclimate it is not, and the
   dashboard says so.

2. `GWETROOT` is *modelled* root-zone soil wetness from MERRA-2's land
   surface, not a probe reading and not a direct satellite retrieval. It is
   the degree of saturation of the root zone, dimensionless on [0, 1].

Point 2 deserves emphasis because it is the reason this module exists. The
sequence models in `notebooks/` need a soil-moisture target that was not
produced by this project's own simulator - otherwise they learn my water
balance and the evaluation proves nothing (see CLAUDE.md invariant 4 for the
same principle applied to predictors). GWETROOT is independent of anything
here: different physics, different authors, different inputs. It is a weaker
target than a calibrated in-situ probe would be, and a stronger one than
self-generated data can ever be.

NASA SMAP L4 would give a satellite retrieval proper, at 9 km. It requires a
free NASA Earthdata login, so it is a deliberate upgrade path rather than the
default: this module stays runnable by anyone who clones the repository.

CITATION

These data were obtained from the NASA Langley Research Center POWER Project,
funded through the NASA Earth Science Directorate Applied Science Program.
<https://power.larc.nasa.gov/>
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..climate.dubai import (
    DUBAI_ALTITUDE_M,
    DUBAI_LATITUDE,
    DUBAI_LONGITUDE,
    DailyWeather,
)
from ..physics.penman_monteith import saturation_vapour_pressure

# --------------------------------------------------------------------------
# Request definition
# --------------------------------------------------------------------------
POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"

#: Maps each POWER parameter name to the column name used here, its unit, and
#: a description.
#:
#: Units are recorded because POWER's own defaults differ by community: the
#: AG community returns irradiance in MJ m-2 day-1, the RE community returns
#: kWh m-2 day-1 for the same parameter name. A bare number labelled
#: "radiation" is a 3.6x error waiting to happen, and it would show up as a
#: plausible-looking ET0 rather than an obvious failure.
POWER_PARAMETERS: dict[str, tuple[str, str, str]] = {
    "T2M_MAX":          ("tmax_c",       "C",             "max air temperature at 2 m"),
    "T2M_MIN":          ("tmin_c",       "C",             "min air temperature at 2 m"),
    "T2M":              ("tmean_c",      "C",             "mean air temperature at 2 m"),
    "T2MDEW":           ("dewpoint_c",   "C",             "dewpoint temperature at 2 m"),
    "RH2M":             ("rh_mean_pct",  "%",             "mean relative humidity at 2 m"),
    "WS2M":             ("wind_2m_ms",   "m/s",           "wind speed at 2 m"),
    "WS10M":            ("wind_10m_ms",  "m/s",           "wind speed at 10 m"),
    "ALLSKY_SFC_SW_DWN": ("solar_mj",    "MJ m-2 day-1",  "all-sky downward shortwave"),
    "PRECTOTCORR":      ("rainfall_mm",  "mm/day",        "bias-corrected precipitation"),
    "GWETTOP":          ("wetness_top",  "0-1",           "surface soil wetness"),
    "GWETROOT":         ("wetness_root", "0-1",           "root-zone soil wetness"),
    "GWETPROF":         ("wetness_prof", "0-1",           "profile soil wetness"),
}

#: POWER substitutes this for missing values. It is a plausible-looking float,
#: not a NaN, so it must be filtered explicitly, or it silently poisons means.
POWER_FILL_VALUE = -999.0

#: Complete calendar years keep the chronological train/val/test split clean
#: and make annual totals directly comparable to published climatology.
DEFAULT_START = _dt.date(1995, 1, 1)
DEFAULT_END = _dt.date(2024, 12, 31)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
DEFAULT_CSV = _DATA_DIR / "nasa_power_dubai_daily.csv"
DEFAULT_METADATA = _DATA_DIR / "nasa_power_dubai_daily.meta.json"


# --------------------------------------------------------------------------
# Download (the only networked code in the project)
# --------------------------------------------------------------------------
def build_request_url(
    start: _dt.date = DEFAULT_START,
    end: _dt.date = DEFAULT_END,
    latitude_deg: float = DUBAI_LATITUDE,
    longitude_deg: float = DUBAI_LONGITUDE,
) -> str:
    """
    Compose the POWER request URL.

    Split out from `download` so the exact request can be printed, pasted into
    a browser, and checked by a reviewer without running any code.
    """
    params = ",".join(POWER_PARAMETERS)
    return (
        f"{POWER_ENDPOINT}?parameters={params}&community=AG"
        f"&latitude={latitude_deg}&longitude={longitude_deg}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    )


def download(
    csv_path: Path = DEFAULT_CSV,
    metadata_path: Path = DEFAULT_METADATA,
    start: _dt.date = DEFAULT_START,
    end: _dt.date = DEFAULT_END,
    latitude_deg: float = DUBAI_LATITUDE,
    longitude_deg: float = DUBAI_LONGITUDE,
    timeout_s: float = 180.0,
) -> Path:
    """
    Fetch the POWER record and cache it. Requires network; nothing else does.

    Run by hand to create or refresh `data/raw/`. The resulting CSV is
    committed, which is what keeps the library, tests, demo and dashboard
    fully offline.

    The response's own header block - API version, data sources, fill value,
    and the *actual* grid-cell coordinates POWER snapped to - is written
    alongside as JSON. Those snapped coordinates are the honest answer to
    "where is this data from", and they are not the coordinates requested.

    Returns:
        Path to the written CSV.
    """
    import urllib.request

    url = build_request_url(start, end, latitude_deg, longitude_deg)
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.load(response)

    # The payload comes back from `json.load` as Any, so the shape is declared
    # here rather than inferred. Without it the date keys carry no type, and
    # `strptime` below is being handed something only assumed to be a string.
    parameters: dict[str, dict[str, float]] = payload["properties"]["parameter"]
    dates: list[str] = sorted(next(iter(parameters.values())))
    columns = [POWER_PARAMETERS[name][0] for name in POWER_PARAMETERS]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *columns])
        for stamp in dates:
            day = _dt.datetime.strptime(stamp, "%Y%m%d").date()
            writer.writerow(
                [day.isoformat()]
                + [parameters[name][stamp] for name in POWER_PARAMETERS]
            )

    longitude, latitude, elevation = payload["geometry"]["coordinates"]
    metadata = {
        "request_url": url,
        "requested_point": {"latitude": latitude_deg, "longitude": longitude_deg},
        "grid_cell_point": {
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
        },
        "downloaded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "row_count": len(dates),
        "columns": {
            POWER_PARAMETERS[name][0]: {
                "power_parameter": name,
                "unit": POWER_PARAMETERS[name][1],
                "description": POWER_PARAMETERS[name][2],
            }
            for name in POWER_PARAMETERS
        },
        "power_header": payload.get("header", {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerRecord:
    """
    One day of NASA POWER output, units as documented in POWER_PARAMETERS.

    Kept distinct from `DailyWeather` because it holds things DailyWeather has
    no business knowing about - three soil wetness depths, wind at two heights
    - and omits nothing that arrived. Conversion to the weather contract is an
    explicit, lossy step (`to_daily_weather`), not an implicit one.
    """

    date: _dt.date
    tmax_c: float
    tmin_c: float
    tmean_c: float
    dewpoint_c: float
    rh_mean_pct: float
    wind_2m_ms: float
    wind_10m_ms: float
    solar_mj: float
    rainfall_mm: float
    wetness_top: float
    wetness_root: float
    wetness_prof: float

    @property
    def day_of_year(self) -> int:
        return self.date.timetuple().tm_yday


def _is_missing(value: float) -> bool:
    """
    POWER's sentinel, matched with tolerance rather than equality.

    Round-tripping through CSV can perturb the last digit, and `-999.0 == x`
    would then quietly pass a fill value through as real data.
    """
    return math.isnan(value) or abs(value - POWER_FILL_VALUE) < 0.5


def load_records(csv_path: Path = DEFAULT_CSV) -> list[PowerRecord]:
    """
    Read the cached CSV, dropping any day with a missing value.

    Days are dropped whole rather than imputed. A day missing radiation cannot
    have a trustworthy ET0, and interpolating it would put a fabricated number
    into a series whose entire purpose is to be the real one. In practice the
    Dubai cell has no gaps; the check exists so that a future refresh which
    *does* have gaps fails visibly.

    Raises:
        FileNotFoundError: If the cache is absent, with the command to create
            it - a missing-file traceback is a bad way to learn that.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NASA POWER cache not found at {csv_path}.\n"
            "It is committed to the repository; if it is genuinely missing, "
            "recreate it with:\n"
            "    python scripts/fetch_nasa_power.py"
        )

    records: list[PowerRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values = {
                key: float(value) for key, value in row.items() if key != "date"
            }
            if any(_is_missing(value) for value in values.values()):
                continue
            records.append(
                PowerRecord(date=_dt.date.fromisoformat(row["date"]), **values)
            )
    return records


def load_metadata(metadata_path: Path = DEFAULT_METADATA) -> dict:
    """Provenance sidecar written by `download`."""
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Conversion to the weather contract
# --------------------------------------------------------------------------
def relative_humidity_extremes(
    tmax_c: float, tmin_c: float, dewpoint_c: float
) -> tuple[float, float]:
    """
    RHmax and RHmin [%] from dewpoint and the temperature extremes.

    POWER reports mean relative humidity, but `DailyWeather` is defined in
    terms of the daily extremes that FAO-56 Eq. 17 uses. Rather than invent
    them, they are derived from the identity RH = 100 * ea / e0(T), holding
    ea fixed across the day - the same assumption FAO-56 makes when it treats
    dewpoint as a daily constant.

    RH is highest when air is coldest, so RHmax pairs with Tmin. The result is
    clamped to [1, 100]: a reanalysis dewpoint can exceed Tmin by a fraction of
    a degree, which is physically supersaturated and would give RH > 100.

    Note this makes the returned extremes *consistent with* ea rather than
    independent evidence about it - which is exactly why `et0_for_day` uses
    the dewpoint directly (Eq. 14) and never these values.
    """
    ea = saturation_vapour_pressure(dewpoint_c)
    rh_max = 100.0 * ea / saturation_vapour_pressure(tmin_c)
    rh_min = 100.0 * ea / saturation_vapour_pressure(tmax_c)
    return (
        min(100.0, max(1.0, rh_max)),
        min(100.0, max(1.0, rh_min)),
    )


def to_daily_weather(record: PowerRecord) -> DailyWeather:
    """
    Project a POWER record onto the `DailyWeather` contract.

    Measured radiation and dewpoint are carried through as the optional fields
    they are, so `et0_for_day` takes the FAO-56 preferred routes and never
    touches `sunshine_hours`. That field is set to NaN rather than to a
    plausible number: there is no sunshine-duration measurement here, and a
    fabricated one would be used silently the moment someone removed the
    radiation column.
    """
    rh_max, rh_min = relative_humidity_extremes(
        record.tmax_c, record.tmin_c, record.dewpoint_c
    )
    return DailyWeather(
        day_of_year=record.day_of_year,
        tmax_c=record.tmax_c,
        tmin_c=record.tmin_c,
        rh_max_pct=rh_max,
        rh_min_pct=rh_min,
        wind_2m_ms=record.wind_2m_ms,
        sunshine_hours=math.nan,
        rainfall_mm=record.rainfall_mm,
        date=record.date,
        solar_radiation_mj=record.solar_mj,
        dewpoint_c=record.dewpoint_c,
    )


def load_weather(csv_path: Path = DEFAULT_CSV) -> list[DailyWeather]:
    """Cached POWER record as a chronological list of `DailyWeather`."""
    return [to_daily_weather(record) for record in load_records(csv_path)]


#: Elevation POWER reports for the Dubai grid cell, kept for reference. The
#: project uses DUBAI_ALTITUDE_M for ET0 because the cell's mean elevation is
#: pulled up by inland terrain, while the irrigated area near the coast is at
#: roughly sea level - and Rso and atmospheric pressure both depend on it.
SITE_ALTITUDE_M = DUBAI_ALTITUDE_M
