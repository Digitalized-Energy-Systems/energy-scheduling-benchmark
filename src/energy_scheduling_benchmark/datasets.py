"""Benchmark networks: raw PyPSA input → MEE ``Network`` → the benchmark's preparation recipe.

Preparation runs in memory on every load; the raw files in ``resources/networks`` stay untouched.
The prepared network does not depend on the environment that later runs it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from mango_energy_environments import (
    Network,
    add_quadratic_cost,
    drop_unusable_units,
    from_pypsa,
    set_storage_defaults,
)

# shortcut: path relative to the source checkout (editable install); package data when distributed.
NETWORKS_DIR = Path(__file__).resolve().parents[2] / "resources" / "networks"
NETWORKS = {
    "base_s_1_elec_2019": NETWORKS_DIR / "base_s_1_elec_2019.nc",
    # built by resources/networks/build_lower_saxony_2030.py
    "lower_saxony_2030": NETWORKS_DIR / "lower_saxony_2030.nc",
}

#: Quadratic cost factor k in c₂ = k·c₁/p_nom on thermal units (old decision Q17). No network
#: provides quadratic coefficients, so they are derived from the marginal cost c₁: k is the mean
#: of c·P_max/b over the three units of Wood & Wollenberg, *Power Generation, Operation and
#: Control*, Example 4E (0.091, 0.129, 0.123; data via the GAMS model library ``dispatch``),
#: rounded to 0.11. The marginal cost then rises from c₁ at zero output to (1 + 2k)·c₁ ≈ 1.22·c₁
#: at full output. ``k = 0`` is the linear cost model.
QUADRATIC_K = 0.11
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
    storage_max_hours: dict[str, float] | None = None,
    soc_initial: float = SOC_INITIAL,
) -> Network:
    """Load network *name* (``"toy"`` or a key of :data:`NETWORKS`), convert it, prepare it."""
    net = from_pypsa(toy_network() if name == "toy" else pypsa.Network(NETWORKS[name]))
    drop_unusable_units(net, DROP_STORAGE_CARRIERS)
    add_quadratic_cost(net, k)
    set_storage_defaults(net, storage_max_hours or STORAGE_MAX_HOURS, soc_initial)
    return net


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
