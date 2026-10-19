"""Replace a network's wind / solar availability profiles with measured ones
(decision Q23, TODO Phase 2c).

Every generator whose carrier maps to a column of the profile CSV (written by
``build_tennet_profiles.py``) gets that column as its ``p_max_pu`` series;
capacities (``p_nom``, e.g. the 2030 targets) stay as they are, so production =
measured capacity factor × installed capacity. The series is an upper bound:
every algorithm, like central dispatch, may curtail below it. Other carriers
(run-of-river, thermal, storage) are not touched. Network snapshots are taken
as UTC (PyPSA-Eur convention).

Usage (from the repo root)::

    python energy-scheduling-benchmark/resources/networks/apply_renewable_profiles.py \\
        --network energy-scheduling-benchmark/resources/networks/lower_saxony_2030.nc \\
        --profiles energy-scheduling-benchmark/resources/networks/tennet_2019_capacity_factors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pypsa

#: Network carrier -> profile column.
CARRIER_COLUMN = {
    "solar": "solar",
    "solar-hsat": "solar",
    "onwind": "onwind",
    "offwind-ac": "offwind",
    "offwind-dc": "offwind",
    "offwind-float": "offwind",
}


def apply_profiles(n: pypsa.Network, profiles: pd.DataFrame) -> dict[str, int]:
    """Set ``p_max_pu`` of every mapped generator; returns the count per carrier."""
    snapshots = pd.DatetimeIndex(n.snapshots)
    utc = (
        snapshots.tz_localize("UTC")
        if snapshots.tz is None
        else snapshots.tz_convert("UTC")
    )
    missing = utc.difference(profiles.index)
    if len(missing):
        raise ValueError(f"profiles lack {len(missing)} snapshots, e.g. {missing[0]}")
    aligned = profiles.reindex(utc)
    aligned.index = n.snapshots
    counts: dict[str, int] = {}
    pmpu = n.generators_t.p_max_pu
    for name, carrier in n.generators["carrier"].items():
        column = CARRIER_COLUMN.get(str(carrier))
        if column is None:
            continue
        pmpu[name] = aligned[column].to_numpy()
        counts[str(carrier)] = counts.get(str(carrier), 0) + 1
    n.generators_t.p_max_pu = pmpu
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: overwrite --network."
    )
    args = parser.parse_args(argv)

    profiles = pd.read_csv(args.profiles, index_col=0, parse_dates=True)
    if profiles.index.tz is None:
        profiles.index = profiles.index.tz_localize("UTC")
    n = pypsa.Network(args.network)
    counts = apply_profiles(n, profiles)
    n.export_to_netcdf(args.out or args.network)
    print(f"replaced p_max_pu of {sum(counts.values())} generators: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
