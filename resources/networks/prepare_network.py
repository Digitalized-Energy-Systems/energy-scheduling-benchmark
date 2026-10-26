"""Clean a PyPSA network once so the scenarios can use it as it is.

Applied to every bundled ``.nc``; run it on any network you add:

  * drop hydro storage units (natural inflow is not modelled) and generators /
    storage units with zero or non-finite ``p_nom``
  * write ``marginal_cost_quadratic`` = c₂ = 0.11·c₁/p_nom on thermal units
    (decision Q17), which ``--cost-model quadratic`` (the default) reads
  * storage: ``max_hours`` PHS 6 h, battery 2 h; initial SOC 50 %

Unprepared networks still load; the scenarios' own fallbacks then apply.

Usage (from the repo root)::

    python energy-scheduling-benchmark/resources/networks/prepare_network.py \\
        --network energy-scheduling-benchmark/resources/networks/base_s_1_elec_2019.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pypsa
from mango_energy_environments import DEFAULT_RENEWABLE_CARRIERS

from energy_scheduling_benchmark.scenarios._model import QUADRATIC_K

MAX_HOURS = {"PHS": 6.0, "battery": 2.0}


def prepare(n: pypsa.Network) -> None:
    su = n.storage_units
    n.remove("StorageUnit", su.index[(su.carrier == "hydro") | ~(su.p_nom > 0)])
    g = n.generators
    n.remove("Generator", g.index[~np.isfinite(g.p_nom) | ~(g.p_nom > 0)])

    g = n.generators
    thermal = ~g.carrier.str.lower().isin(DEFAULT_RENEWABLE_CARRIERS)
    g["marginal_cost_quadratic"] = np.where(
        thermal, QUADRATIC_K * g.marginal_cost.clip(lower=0) / g.p_nom, 0.0
    )

    su = n.storage_units
    su["max_hours"] = su.carrier.map(MAX_HOURS).fillna(su.max_hours)
    su["state_of_charge_initial"] = 0.5 * su.p_nom * su.max_hours


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: overwrite --network."
    )
    args = parser.parse_args(argv)

    n = pypsa.Network(args.network)
    prepare(n)
    n.export_to_netcdf(args.out or args.network)
    print(
        f"{args.network.name}: {len(n.generators)} generators, "
        f"{len(n.storage_units)} storage units"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
