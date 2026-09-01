"""Slice a Lower Saxony (NUTS1 = DE9) sub-network out of an unclustered
PyPSA-Eur Germany extract and write it as a benchmark input ``.nc``.

PyPSA-Eur cannot scope the model below country level, so the upstream build
(``config/config.lower-saxony-2030.yaml``) produces the whole of Germany with
no clustering. This script does the geographic cut:

  * keep every bus whose location falls inside the dissolved DE9* NUTS shape
  * keep lines / links / transformers with *both* ends inside
  * keep every generator / load / storage unit / store attached to a kept bus,
    with all their time series
  * optionally scale wind / solar / battery capacity up to 2030 national
    targets (applied to all of Germany before the cut, allocated per bus by
    land-availability ``p_nom_max``, so Lower Saxony keeps its geographic share)
  * optionally reduce to the largest connected component
  * border handling: islanded (default) or one aggregated import/export node

Usage (from the repo root, .venv active):

  python energy-scheduling-benchmark/resources/networks/extract_lower_saxony.py \
      --src pypsa-eur/resources/lower-saxony-2030/networks/base_s_all_elec_2030.nc \
      --shapes pypsa-eur/resources/lower-saxony-2030/nuts3_shapes.geojson \
      --out energy-scheduling-benchmark/resources/networks/lower_saxony_2030.nc \
      --uplift-2030 --largest-component

Then:  ./run_all_scenarios.sh lower_saxony_2030.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from shapely.geometry import Point

# NUTS1 code for Lower Saxony. Its NUTS2 children are DE91..DE94, NUTS3 DE911..
REGION_PREFIX = "DE9"

# 2030 *national* (whole-Germany) installed-capacity targets [MW].
# Defaults follow EEG 2023 / WindSeeG headline figures — edit to your scenario.
# The uplift runs on the full German network *before* the Lower Saxony cut, so
# each carrier's national increment is spread over all German buses in
# proportion to their land-availability ``p_nom_max`` headroom; Lower Saxony
# then keeps its geographic share. Set a value to None to leave a carrier alone.
TARGETS_2030_MW = {
    "onwind": 115_000,
    "offwind-ac": 30_000,  # combined offshore target; split across ac/dc as built
    "offwind-dc": None,
    "solar": 215_000,
    "battery": 24_000,  # storage-unit power rating (max_hours from config)
}

BRANCH_COMPONENTS = {"Line", "Link", "Transformer"}
NODAL_COMPONENTS = ["Generator", "Load", "StorageUnit", "Store", "ShuntImpedance"]


def _region_shape(shapes_path: Path):
    gdf = gpd.read_file(shapes_path)
    # the code lives either in the index or in one of these columns
    code = None
    for col in ("index", "nuts3", "id", "NUTS_ID", "name"):
        if col in gdf.columns:
            code = gdf[col].astype(str)
            break
    if code is None:
        code = gdf.index.to_series().astype(str)
    mask = code.str.startswith(REGION_PREFIX)
    if not mask.any():
        raise SystemExit(
            f"No shapes matching {REGION_PREFIX!r} in {shapes_path}. "
            f"Columns: {list(gdf.columns)}"
        )
    geom = gdf.loc[mask.values].to_crs(4326).union_all()
    return geom


def _buses_in_region(n: pypsa.Network, geom) -> pd.Index:
    pts = gpd.GeoSeries(
        [Point(xy) for xy in zip(n.buses.x, n.buses.y)],
        index=n.buses.index,
        crs=4326,
    )
    # small positive buffer so buses exactly on the border are kept
    inside = pts.within(geom.buffer(0.01))
    return n.buses.index[inside.values]


def _copy_component(
    src: pypsa.Network, dst: pypsa.Network, cls: str, keep: pd.Index
) -> None:
    sc = src.components[cls]
    sdf = sc.static.loc[sc.static.index.intersection(keep)]
    if sdf.empty:
        return
    dst.add(cls, sdf.index, **{c: sdf[c] for c in sdf.columns})
    dc = dst.components[cls]
    for attr, tdf in sc.dynamic.items():
        cols = tdf.columns.intersection(sdf.index)
        if len(cols):
            dc.dynamic[attr] = tdf.loc[:, cols].copy()


def _uplift_2030(n: pypsa.Network) -> None:
    """Scale renewable / battery capacity to national 2030 targets, spreading
    the delta over buses in proportion to their remaining ``p_nom_max`` room."""
    for carrier, target in TARGETS_2030_MW.items():
        if target is None:
            continue
        if carrier == "battery":
            df = n.storage_units
        else:
            df = n.generators
        sel = df.index[df.carrier == carrier]
        if len(sel) == 0:
            print(f"  uplift: no {carrier} units — skipped")
            continue
        current = df.loc[sel, "p_nom"].sum()
        if current >= target:
            print(
                f"  uplift: {carrier} already {current:.0f} >= target {target} MW — skipped"
            )
            continue
        headroom = (df.loc[sel, "p_nom_max"] - df.loc[sel, "p_nom"]).clip(lower=0)
        if headroom.sum() == 0 or not np.isfinite(headroom.sum()):
            weights = pd.Series(1.0 / len(sel), index=sel)  # even split fallback
        else:
            weights = headroom / headroom.sum()
        add = (target - current) * weights
        df.loc[sel, "p_nom"] = df.loc[sel, "p_nom"] + add
        print(f"  uplift: {carrier} {current:.0f} -> {target} MW (+{add.sum():.0f})")


def _largest_component(n: pypsa.Network) -> None:
    n.determine_network_topology()
    if n.buses.sub_network.nunique() <= 1:
        return
    biggest = n.buses.sub_network.value_counts().idxmax()
    drop = n.buses.index[n.buses.sub_network != biggest]
    print(f"  largest component: dropping {len(drop)} buses in minor islands")
    for cls in NODAL_COMPONENTS:
        df = n.components[cls].static
        n.remove(cls, df.index[df.bus.isin(drop)])
    for cls in BRANCH_COMPONENTS:
        df = n.components[cls].static
        n.remove(cls, df.index[df.bus0.isin(drop) | df.bus1.isin(drop)])
    n.remove("Bus", drop)


def _add_exchange_node(
    sub: pypsa.Network,
    full: pypsa.Network,
    ls_buses: pd.Index,
    import_price: float,
    export_price: float,
) -> None:
    """Collapse everything outside Lower Saxony into a single node connected by
    the severed tie-lines, with a price-taking import generator and an
    export sink. Requires the upstream build to include neighbouring buses
    (i.e. the real German grid beyond DE9)."""
    cut = []
    for cls in ("Line", "Link"):
        df = full.components[cls].static
        one_in = df.bus0.isin(ls_buses) ^ df.bus1.isin(ls_buses)
        for name, row in df.loc[one_in].iterrows():
            inner = row.bus0 if row.bus0 in ls_buses else row.bus1
            cap = row.get("s_nom", row.get("p_nom", 0.0)) or 0.0
            cut.append((inner, float(cap)))
    if not cut:
        print("  border: no severed tie-lines found — nothing to do")
        return
    sub.add(
        "Bus", "DE_rest", x=float(sub.buses.x.mean()), y=float(sub.buses.y.max()) + 1.0
    )
    total_cap = sum(c for _, c in cut)
    for i, (inner, cap) in enumerate(cut):
        sub.add(
            "Line",
            f"tie_{i}_{inner}",
            bus0=inner,
            bus1="DE_rest",
            s_nom=cap,
            x=0.1,
            r=0.01,
            carrier="AC",
        )
    sub.add(
        "Generator",
        "import_DE_rest",
        bus="DE_rest",
        carrier="import",
        p_nom=total_cap,
        marginal_cost=import_price,
    )
    sub.add("Load", "export_DE_rest", bus="DE_rest", p_set=0.0)
    sub.add(
        "Generator",
        "export_DE_rest_sink",
        bus="DE_rest",
        carrier="export",
        p_nom=total_cap,
        p_max_pu=0.0,
        p_min_pu=-1.0,
        marginal_cost=-export_price,
    )
    print(f"  border: DE_rest node via {len(cut)} tie-lines, {total_cap:.0f} MW")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="unclustered PyPSA-Eur DE extract (base_s_all_elec_*.nc)",
    )
    ap.add_argument(
        "--shapes",
        required=True,
        type=Path,
        help="nuts3_shapes.geojson from the same PyPSA-Eur run",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--uplift-2030",
        action="store_true",
        help="scale wind/solar/battery to 2030 national targets (TARGETS_2030_MW), "
        "distributed by p_nom_max across all German buses before the LS cut",
    )
    ap.add_argument(
        "--largest-component",
        action="store_true",
        help="keep only the largest connected sub-network",
    )
    ap.add_argument("--border", choices=["islanded", "exchange"], default="islanded")
    ap.add_argument("--import-price", type=float, default=80.0, help="EUR/MWh")
    ap.add_argument("--export-price", type=float, default=5.0, help="EUR/MWh")
    args = ap.parse_args()

    print(f"loading {args.src}")
    n = pypsa.Network(str(args.src))
    print(
        f"  full network: {len(n.buses)} buses, {len(n.generators)} generators, "
        f"{len(n.loads)} loads, {len(n.snapshots)} snapshots"
    )

    geom = _region_shape(args.shapes)
    ls_buses = _buses_in_region(n, geom)
    if len(ls_buses) == 0:
        raise SystemExit(
            "no buses fell inside the DE9 shape — check --src / --shapes CRS"
        )
    print(f"  Lower Saxony: {len(ls_buses)} buses")

    if args.uplift_2030:
        print("2030 capacity uplift (national, pre-cut):")
        _uplift_2030(n)
        ls_gen = n.generators[n.generators.bus.isin(ls_buses)]
        ls_su = n.storage_units[n.storage_units.bus.isin(ls_buses)]
        for c in ("onwind", "offwind-ac", "solar"):
            print(
                f"    -> Lower Saxony {c}: "
                f"{ls_gen.loc[ls_gen.carrier == c, 'p_nom'].sum():.0f} MW"
            )
        print(f"    -> Lower Saxony battery: {ls_su.p_nom.sum():.0f} MW")

    sub = pypsa.Network()
    sub.set_snapshots(n.snapshots)
    sub.snapshot_weightings.loc[:, :] = n.snapshot_weightings.values
    _copy_component(n, sub, "Carrier", n.carriers.index)
    _copy_component(n, sub, "Bus", ls_buses)

    for cls in BRANCH_COMPONENTS:
        df = n.components[cls].static
        both_in = df.index[df.bus0.isin(ls_buses) & df.bus1.isin(ls_buses)]
        _copy_component(n, sub, cls, both_in)

    for cls in NODAL_COMPONENTS:
        df = n.components[cls].static
        on_bus = df.index[df.bus.isin(ls_buses)]
        _copy_component(n, sub, cls, on_bus)

    print(
        f"  extract: {len(sub.buses)} buses, {len(sub.lines)} lines, "
        f"{len(sub.links)} links, {len(sub.generators)} generators, "
        f"{len(sub.storage_units)} storage units, {len(sub.loads)} loads"
    )

    if args.largest_component:
        print("connectivity:")
        _largest_component(sub)

    if args.border == "exchange":
        print("border treatment:")
        _add_exchange_node(sub, n, ls_buses, args.import_price, args.export_price)

    gen_by_carrier = sub.generators.groupby("carrier").p_nom.sum().round(0)
    print("\ninstalled capacity by carrier [MW]:")
    print(gen_by_carrier.to_string())
    ann_load = (sub.loads_t.p_set.sum(axis=1) * sub.snapshot_weightings.objective).sum()
    print(f"annual demand: {ann_load / 1e6:.1f} TWh  (Lower Saxony ~55-60 TWh)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.export_to_netcdf(str(args.out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
