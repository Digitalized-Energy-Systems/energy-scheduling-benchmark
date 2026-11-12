"""Build ``lower_saxony_2025.nc`` (the dataset's raw input) from the unclustered PyPSA-Eur Germany
network, cut to the Lower Saxony buses (NUTS1 DE9), islanded. Profiles come from a named source;
the default keeps the network's own.

Replaces ``lower_saxony_2030`` (supervisor meeting 2026-11-10: no 2030 uplift — with the EEG 2030
targets renewables covered the load in 82 % of the hours). Same data as ``pypsa_eur_germany_100``:
2025 weather + demand, fleet in service at the end of 2024 with renewables topped up to the IRENA
2024 country totals (`from_irenastat`, audit 2026-11-12 Q-A2), 2025 costs, no uplift
(`docs/renewables-predispatch-plan.md` D9).

Inputs (not in git, in ``data/pypsa-eur/``): ``base_s_all_elec_2025.nc`` = the PyPSA-Eur output
``resources/lower-saxony-2025/networks/base_s_all_elec_.nc`` (built with
``config.lower-saxony-2025.yaml``, a copy is next to this script; needs the powerplant bus-mapping
patch) and ``nuts3_shapes.geojson``. Needs ``geopandas`` (ESB extra ``[build]``).

Usage (from the workspace root)::

    python energy-scheduling-benchmark/resources/networks/build_lower_saxony_2025.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pypsa
from mango_energy_environments import keep_buses

DATA = Path(__file__).resolve().parents[3] / "data" / "pypsa-eur"
OUT = Path(__file__).with_name("lower_saxony_2025.nc")
REGION = "DE9"  # NUTS1 Lower Saxony

# Profile sources: set renewable p_max_pu and load p_set. "pypsa-eur" keeps the network's own:
# capacity factors per bus from ERA5 2025 weather (atlite) and 2025 demand per load.
PROFILE_SOURCES = {"pypsa-eur": lambda n: None}


def region_buses(n: pypsa.Network, shapes: Path):
    shape = gpd.read_file(shapes).query("level1 == @REGION").to_crs(4326).union_all()
    points = gpd.points_from_xy(n.buses.x, n.buses.y, crs=4326)
    # small buffer so buses exactly on the border are kept
    return n.buses.index[points.within(shape.buffer(0.01))]


def build(profiles: str = "pypsa-eur") -> pypsa.Network:
    n = pypsa.Network(DATA / "base_s_all_elec_2025.nc")
    keep_buses(n, region_buses(n, DATA / "nuts3_shapes.geojson"))
    # Offshore wind dropped (Jonas, 2026-11-11): 8.8 of Germany's 11.2 GW land in DE9 and serve the
    # national grid, not an islanded Lower Saxony — with it renewables were 1.54× the load (63 %
    # surplus hours, 125 days fully covered); without: 0.74× (27 %, 29 days). Since the Q-A2 rebuild
    # (2026-11-12, PV 3.8 → 10.5 GW, onshore 15.5 → 13.7 GW): 0.81× (34 %, 24 days).
    n.remove(
        "Generator", n.generators.index[n.generators.carrier.str.startswith("offwind")]
    )
    PROFILE_SOURCES[profiles](n)
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", choices=PROFILE_SOURCES, default="pypsa-eur")
    n = build(parser.parse_args().profiles)
    n.export_to_netcdf(OUT)
    print(
        f"{OUT.name}: {len(n.buses)} buses, {len(n.lines)} lines, {len(n.links)} links"
    )
