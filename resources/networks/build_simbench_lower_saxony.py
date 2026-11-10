"""Build ``simbench_lower_saxony.json`` (the dataset's raw input): SimBench ``1-EHV-mixed--0-sw``
cut to the Lower Saxony buses (NUTS1 DE9, same shapes as lower_saxony_2025), islanded, one bus
per substation, marginal costs from the former lower_saxony_2030 (decided 2026-11-09, TODO Next #8b).

- buses joined by closed bus–bus switches or transformers are fused (one agent per substation);
  substations cut off from the rest by the region border are dropped;
- c₁ per SimBench type = the median c₁ of that carrier in lower_saxony_2030 (PyPSA-Eur 2030),
  written to pandapower's ``poly_cost``; c₂ and the band come from ``esb-prepare``;
- profiles: hourly means of SimBench's 15-min values, real 2016 dates; stored in the JSON as
  ``profiles_load`` / ``profiles_sgen`` (time × element index, MW).

Inputs: the ``simbench`` package (MEE dependency) and ``data/pypsa-eur/nuts3_shapes.geojson``
(not in git). Needs ``geopandas`` (ESB extra ``[build]``).

Usage (from the workspace root)::

    python energy-scheduling-benchmark/resources/networks/build_simbench_lower_saxony.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandapower as pp
import pandapower.toolbox as tb
from mango_energy_environments import load_simbench

SHAPES = (
    Path(__file__).resolve().parents[3] / "data" / "pypsa-eur" / "nuts3_shapes.geojson"
)
OUT = Path(__file__).with_name("simbench_lower_saxony.json")
CODE, REGION = "1-EHV-mixed--0-sw", "DE9"
# median c₁ (€/MWh) per carrier in lower_saxony_2030 (computed 2026-11-09); gas → CCGT
C1 = {
    ("gen", "gas"): 54.6,
    ("gen", "hard coal"): 27.3,
    ("gen", "lignite"): 28.2,
    ("gen", "oil"): 132.7,
    ("gen", "waste"): 35.4,
    ("sgen", "biomass"): 20.0,
}  # other sgen types (wind, "mixed" 586 MW: no fuel given) are renewables with c₁ = 0


def region_buses(net) -> list[int]:
    xy = net.bus.geo.map(lambda s: json.loads(s)["coordinates"])
    points = gpd.GeoSeries(
        gpd.points_from_xy(xy.str[0], xy.str[1]), net.bus.index, crs=4326
    )
    shape = gpd.read_file(SHAPES).query("level1 == @REGION").to_crs(4326).union_all()
    # small buffer so buses exactly on the border are kept (as in build_lower_saxony_2025.py)
    return list(net.bus.index[points.within(shape.buffer(0.01))])


def fuse_substations(net) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)
    sw = net.switch[(net.switch.et == "b") & net.switch.closed]
    graph.add_edges_from(zip(sw.bus, sw.element))
    graph.add_edges_from(zip(net.trafo.hv_bus, net.trafo.lv_bus))
    for group in nx.connected_components(graph):
        first, *rest = sorted(group)
        if rest:
            tb.fuse_buses(net, first, rest)


def main_grid(net) -> list[int]:
    """Buses of the largest line-connected part: border substations whose lines all leave the
    region are dropped (decided 2026-11-09; two load-only substations)."""
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)
    graph.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
    return sorted(max(nx.connected_components(graph), key=len))


def build():
    full, profiles = load_simbench(CODE)
    net = tb.select_subnet(full, region_buses(full))
    fuse_substations(net)
    keep = main_grid(net)
    dropped = net.bus.index.difference(keep)
    # only load may be lost here; a cut-off generator would change the fleet
    if net.gen.bus.isin(dropped).any() or net.sgen.bus.isin(dropped).any():
        raise ValueError("a cut-off part holds generators")
    net = tb.select_subnet(net, keep)
    for table in ("gen", "sgen"):
        for idx, kind in net[table].type.items():
            if (table, kind) in C1:
                pp.create_poly_cost(net, idx, table, cp1_eur_per_mw=C1[table, kind])
            elif table == "gen":
                raise KeyError(f"no c₁ for gen type {kind!r}")
    for table in ("load", "sgen"):
        net[f"profiles_{table}"] = profiles[table][list(net[table].index)]
    return net


if __name__ == "__main__":
    net = build()
    pp.to_json(net, OUT)
    print(
        f"{OUT.name}: {len(net.bus)} buses, {len(net.line)} lines, {len(net.gen)} gen, "
        f"{len(net.sgen)} sgen, {len(net.load)} loads, {len(net.ext_grid)} ext_grid"
    )
