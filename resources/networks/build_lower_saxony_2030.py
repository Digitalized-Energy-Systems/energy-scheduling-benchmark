"""Build ``lower_saxony_2030.nc`` (the dataset's raw input) from the unclustered PyPSA-Eur Germany
network: 2030 capacity uplift on all of Germany, then cut to the Lower Saxony buses (NUTS1 DE9),
islanded. Profiles come from a named source; the default keeps the network's own.

Inputs (not in git, in ``data/pypsa-eur/``): ``base_s_all_elec_2030.nc`` (PyPSA-Eur build with
``config.lower-saxony-2030.yaml``, see the archived ``docs/lower-saxony-2030-workflow.md``) and
``nuts3_shapes.geojson``. Needs ``geopandas`` (ESB extra ``[build]``).

Usage (from the workspace root)::

    python energy-scheduling-benchmark/resources/networks/build_lower_saxony_2030.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pypsa
from mango_energy_environments import keep_buses, scale_to_targets

DATA = Path(__file__).resolve().parents[3] / "data" / "pypsa-eur"
OUT = Path(__file__).with_name("lower_saxony_2030.nc")
REGION = "DE9"  # NUTS1 Lower Saxony

# 2030 national (whole-Germany) installed capacity [MW], applied before the cut so each carrier's
# increase is spread over all German buses by p_nom_max headroom and Lower Saxony keeps its share.
#   onwind  115 GW — EEG 2023 § 4 Satz 1 Nr. 1 lit. d
#   solar   215 GW — EEG 2023 § 4 Satz 1 Nr. 3 lit. d
#   offwind  30 GW — WindSeeG § 1 Abs. 2 (combined offshore target, applied to offwind-ac)
#   battery  24 GW — no statutory 2030 target; project assumption of an ambitious 2030 fleet
#            (matches NEP 2037/2045 V2023 large-battery figures for 2037, decided 2026-10-16)
TARGETS_2030_MW = {
    "onwind": 115_000,
    "solar": 215_000,
    "offwind-ac": 30_000,
    "battery": 24_000,
}

# Profile sources: set renewable p_max_pu and load p_set. "pypsa-eur" keeps the network's own:
# capacity factors per bus from ERA5 2019 weather (atlite) and 2019 demand per load.
PROFILE_SOURCES = {"pypsa-eur": lambda n: None}


def region_buses(n: pypsa.Network, shapes: Path):
    shape = gpd.read_file(shapes).query("level1 == @REGION").to_crs(4326).union_all()
    points = gpd.points_from_xy(n.buses.x, n.buses.y, crs=4326)
    # small buffer so buses exactly on the border are kept
    return n.buses.index[points.within(shape.buffer(0.01))]


def build(profiles: str = "pypsa-eur") -> pypsa.Network:
    n = pypsa.Network(DATA / "base_s_all_elec_2030.nc")
    scale_to_targets(n, TARGETS_2030_MW)
    keep_buses(n, region_buses(n, DATA / "nuts3_shapes.geojson"))
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
