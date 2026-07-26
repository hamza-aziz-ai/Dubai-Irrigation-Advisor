"""Refresh the cached NASA POWER record for Dubai.

This is the only script in the repository that touches the network, and it is
run by hand. Its output is committed, so everything else - library, tests,
demo, dashboard - runs offline.

    python scripts/fetch_nasa_power.py

Re-running it overwrites `data/raw/`. Expect small differences in older years
after a NASA reanalysis update; that is why the metadata sidecar records the
download timestamp and the POWER API version.
"""
from __future__ import annotations

import sys

from irrigation.data import nasa_power


def main() -> int:
    url = nasa_power.build_request_url()
    print("Requesting NASA POWER daily data")
    print(f"  {url}\n")

    csv_path = nasa_power.download()
    metadata = nasa_power.load_metadata()
    records = nasa_power.load_records(csv_path)

    cell = metadata["grid_cell_point"]
    print(f"Wrote {csv_path}")
    print(f"  rows written   : {metadata['row_count']}")
    print(f"  rows usable    : {len(records)}  (fill values dropped)")
    print(f"  date range     : {records[0].date} to {records[-1].date}")
    print(
        "  grid cell      : "
        f"{cell['latitude']:.4f} N, {cell['longitude']:.4f} E, "
        f"{cell['elevation_m']:.0f} m"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
