"""Benchmark networks: raw PyPSA input → MEE ``Network`` → the benchmark's preparation recipe.

Prepare once, run many times: ``esb-prepare --network NAME`` writes the prepared network to
``resources/prepared/NAME.nc`` (recipe stored in the file); ``esb-run`` runs from that file. The
raw files in ``resources/networks`` stay untouched. The prepared network does not depend on the
environment that later runs it.

Usage::

    esb-prepare --network lower_saxony_2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
import pypsa
from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    STORAGE,
    Network,
    add_quadratic_cost,
    drop_unusable_units,
    from_pandapower,
    from_pypsa,
    save_network,
    set_storage_defaults,
)

# shortcut: path relative to the source checkout (editable install); package data when distributed.
RESOURCES = Path(__file__).resolve().parents[2] / "resources"
NETWORKS_DIR = RESOURCES / "networks"
PREPARED_DIR = RESOURCES / "prepared"
NETWORKS = {
    "base_s_1_elec_2019": NETWORKS_DIR / "base_s_1_elec_2019.nc",
    # built by resources/networks/build_lower_saxony_2025.py (replaces lower_saxony_2030, 2026-11-11)
    "lower_saxony_2025": NETWORKS_DIR / "lower_saxony_2025.nc",
    # built by resources/networks/build_synthetic.py (balanced bus sensitivities)
    "synthetic_10": NETWORKS_DIR / "synthetic_10.nc",
    "synthetic_50": NETWORKS_DIR / "synthetic_50.nc",
    # PyPSA-Eur run with resources/networks/config.germany-100-2025.yaml (copy of the config in
    # the pypsa-eur checkout), output copied unchanged
    "pypsa_eur_germany_100": NETWORKS_DIR / "pypsa_eur_germany_100.nc",
    # pandapower JSON, built by resources/networks/build_simbench_lower_saxony.py
    "simbench_lower_saxony": NETWORKS_DIR / "simbench_lower_saxony.json",
}

#: Quadratic cost factor k in c₂ = k·c₁/p_nom on thermal units (old decision Q17). No network
#: provides quadratic coefficients, so they are derived from the marginal cost c₁: k is the mean
#: of c·P_max/b over the three units of Wood & Wollenberg, *Power Generation, Operation and
#: Control*, Example 4E (0.091, 0.129, 0.123; data via the GAMS model library ``dispatch``),
#: rounded to 0.11. The marginal cost then rises from c₁ at zero output to (1 + 2k)·c₁ ≈ 1.22·c₁
#: at full output. ``k = 0`` is the linear cost model.
QUADRATIC_K = 0.11
#: Curvature band (€/MWh) for **storage only**: c₂ = band/(2·p_nom), so its marginal cost rises by
#: the band from zero to full output (decision 2026-11-04, `docs/consensus-plan.md` Q2; since
#: 2026-11-10 only storage — renewables and free thermal units are pre-dispatched, S1, and keep
#: c₂ = 0). Why: price-responding algorithms (consensus + innovations, Kar & Hug 2012 / Hug, Kar &
#: Wu 2015) need a strictly convex cost — with c₂ = 0 the answer to a price is all-or-nothing and
#: flips every iteration. It is in the problem itself (central and every algorithm), so no
#: algorithm gets a method-made gap. It is a modelling device, not a physical cost (deviation P1);
#: the physical cost of storage is :data:`STORAGE_COST`. ``band = 0`` (with ``k = 0``) is the
#: linear cost model again.
CURVATURE_BAND = 1.0
BAND_KINDS = (STORAGE,)
#: Storage ageing cost c₁ (€/MWh **discharged**) per carrier (decision 2026-11-10, S2,
#: `docs/storage-cost-plan.md`). Battery: Xu, Zhao, Zheng, Litvinov & Kirschen 2018, "Factoring
#: the Cycle Aging Cost of Batteries Participating in Electricity Markets", IEEE TPWRS 33(2),
#: eq. 5 with one segment: c = R·Φ(1)/η_dis — ageing only on discharge, R = cell replacement
#: cost, Φ(1) = 1/N (life loss of one full cycle). Values from the Danish Energy Agency, "Technology
#: Data for Energy Storage", data sheet v0011, chapter 180 Lithium-ion battery (utility-scale),
#: 2025 (EUR 2020): R = energy component (battery pack) 0.065 M€/MWh, N = lifetime in total number
#: of cycles 6 000 (LFP/NMC; taken as full cycles), η_dis = discharge efficiency 97 %
#: → 11.17 €/MWh. PHS: 0.22 €/MWh discharged = variable O&M of PHS, Zakeri & Syri 2015, "Electrical
#: energy storage systems: A comparative life cycle cost analysis", RSER 42, 569–596, Appendix A
#: Table A1 (median of the reviewed sources; IQR 0.20–0.79, range 0.19–0.84; euro inflation-
#: adjusted to ~2013, footnote 2; unchanged by the 2016 corrigendum) — replaces the placeholder 0
#: (DEA ch. 160), decided 2026-11-10/11 (Jonas). Their VOM is per MWh discharged (eq. 5), as our c₁.
STORAGE_COST = {"battery": 65_000 / (6_000 * 0.97), "PHS": 0.22}
#: Storage energy capacity in hours of p_nom per carrier (old Q25); others keep the network's.
STORAGE_MAX_HOURS = {"PHS": 6.0, "battery": 2.0}
#: Initial state of charge as a fraction of the energy capacity (old Q25).
SOC_INITIAL = 0.5
#: Storage carriers removed: natural inflow is not modelled.
DROP_STORAGE_CARRIERS = ("hydro",)


def load(
    name: str,
    *,
    k: float = QUADRATIC_K,
    band: float = CURVATURE_BAND,
    storage_max_hours: dict[str, float] | None = None,
    soc_initial: float = SOC_INITIAL,
) -> Network:
    """Load network *name* (``"toy"`` or a key of :data:`NETWORKS`), convert it, prepare it."""
    if name == "toy":
        net = from_pypsa(toy_network())
    elif NETWORKS[name].suffix == ".json":
        net = _from_pandapower_json(NETWORKS[name])
    else:
        net = from_pypsa(pypsa.Network(NETWORKS[name]))
    drop_unusable_units(net, DROP_STORAGE_CARRIERS)
    add_quadratic_cost(net, k, band, BAND_KINDS)
    set_storage_defaults(
        net, storage_max_hours or STORAGE_MAX_HOURS, soc_initial, STORAGE_COST
    )
    return net


def prepare(
    name: str,
    *,
    k: float = QUADRATIC_K,
    band: float = CURVATURE_BAND,
    out: Path | None = None,
) -> Path:
    """Load + prepare *name* and save it (default ``resources/prepared/<name>.nc``)."""
    out = Path(out or PREPARED_DIR / f"{name}.nc")
    out.parent.mkdir(parents=True, exist_ok=True)
    recipe = {
        "dataset": name,
        "source": str(NETWORKS[name].name) if name in NETWORKS else "toy_network()",
        "quadratic_k": k,
        "curvature_band": band,
        "band_kinds": list(BAND_KINDS),
        "storage_cost": STORAGE_COST,
        "storage_max_hours": STORAGE_MAX_HOURS,
        "soc_initial": SOC_INITIAL,
        "drop_storage_carriers": list(DROP_STORAGE_CARRIERS),
    }
    save_network(load(name, k=k, band=band), out, meta=recipe)
    return out


#: Meteorological seasons (months) for the standard and tuning days.
SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def typical_days(network: Network) -> dict[str, tuple[str, str]]:
    """Per season (standard day, tuning day): the full days whose residual load (load −
    available renewables, MWh) is closest / second-closest to the season's median day
    (decisions 2026-11-09). The dates live in the README and ``run_all.sh``; regenerate with
    ``typical_days(load_network(prepared_path(name)))``."""
    comps = network.components.values()
    residual = sum(c.p_max for c in comps if c.kind == LOAD) - sum(
        c.p_max for c in comps if c.kind == RENEWABLE
    )
    hourly = pd.Series(residual, index=network.time_index)
    daily = hourly.resample("D").sum()[hourly.resample("D").count() == 24]
    days = {}
    for season, months in SEASONS.items():
        d = daily[daily.index.month.isin(months)]
        standard, tuning = (d - d.median()).abs().sort_values(kind="stable").index[:2]
        days[season] = (str(standard.date()), str(tuning.date()))
    return days


def prepared_path(network: str) -> Path:
    """*network* is a prepared ``.nc`` file or a dataset name (→ ``resources/prepared/``)."""
    path = Path(network) if network.endswith(".nc") else PREPARED_DIR / f"{network}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with: esb-prepare --network {Path(network).stem}"
        )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a benchmark network once.")
    parser.add_argument("--network", choices=["toy", *NETWORKS], required=True)
    parser.add_argument("--k", type=float, default=QUADRATIC_K, help="quadratic cost k")
    parser.add_argument(
        "--band",
        type=float,
        default=CURVATURE_BAND,
        help="curvature band €/MWh for renewables/storage (0 = linear)",
    )
    parser.add_argument(
        "--out", help="output .nc (default resources/prepared/<name>.nc)"
    )
    args = parser.parse_args(argv)
    print(prepare(args.network, k=args.k, band=args.band, out=args.out))


def _from_pandapower_json(path: Path) -> Network:
    """pandapower JSON with ``profiles_<table>`` frames (see build_simbench_lower_saxony.py)."""
    pp_net = pp.from_json(path)
    profiles = {}
    for table in ("load", "sgen"):
        df = pp_net[f"profiles_{table}"]
        # the JSON round trip stores the DatetimeIndex as epoch milliseconds
        profiles[table] = df.set_axis(pd.to_datetime(df.index, unit="ms"))
    return from_pandapower(pp_net, profiles)


def toy_network(hours: int = 48) -> pypsa.Network:
    """Tiny 4-bus chain: gas + coal (thermal), wind (hourly profile), two loads, one battery."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=hours, freq="h"))
    for i in range(4):
        n.add("Bus", f"bus{i}")
    for i in range(3):
        n.add("Line", f"line{i}", bus0=f"bus{i}", bus1=f"bus{i + 1}", x=0.1, s_nom=1000)
    hour = np.arange(hours) % 24
    n.add("Generator", "gas", bus="bus0", carrier="gas", p_nom=100, marginal_cost=30)
    n.add("Generator", "coal", bus="bus1", carrier="coal", p_nom=60, marginal_cost=50)
    n.add(
        "Generator",
        "wind",
        bus="bus2",
        carrier="onwind",
        p_nom=40,
        p_max_pu=0.3 + 0.4 * np.abs(hour - 12) / 12,
    )
    n.add("Load", "load0", bus="bus3", p_set=45 + 10 * np.cos(2 * np.pi * hour / 24))
    n.add("Load", "load1", bus="bus1", p_set=30 + 6 * np.cos(2 * np.pi * hour / 24 + 1))
    # max_hours comes from the recipe (STORAGE_MAX_HOURS)
    n.add("StorageUnit", "battery", bus="bus0", carrier="battery", p_nom=20)
    return n
