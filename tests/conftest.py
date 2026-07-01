"""Shared fixtures for energy-scheduling-benchmark tests."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fix mango import resolution when pytest is run from the repo root.
#
# The repo root contains a bare `mango/` directory (the local editable
# source).  When pytest's CWD is the repo root, Python's PathFinder discovers
# `mango/` as a namespace package (no __init__.py at the top level) BEFORE
# the editable-install finder gets a chance to map it to the real inner
# `mango/mango/` package.  Moving the editable finder ahead of PathFinder
# ensures the correct package is loaded regardless of CWD.
# ---------------------------------------------------------------------------
_path_finder_idx = next(
    (i for i, f in enumerate(sys.meta_path) if getattr(f, "__name__", "") == "PathFinder"),
    None,
)
for _ef in list(sys.meta_path):
    if getattr(_ef, "__name__", "") != "_EditableFinder":
        continue
    _mod = importlib.import_module(_ef.__module__)
    if "mango" not in getattr(_mod, "MAPPING", {}):
        continue
    sys.meta_path.remove(_ef)
    sys.meta_path.insert(_path_finder_idx if _path_finder_idx is not None else 0, _ef)
    break


from energy_scheduling_benchmark.environment import (
    ComponentRef,
)


@pytest.fixture
def simple_pypsa_net():
    """Two-bus network: one thermal generator, one load."""
    import pypsa

    net = pypsa.Network()
    net.set_snapshots(pd.date_range("2024-01-01", periods=1, freq="h"))
    net.add("Bus", "b0", v_nom=20.0)
    net.add("Bus", "b1", v_nom=20.0)
    net.add(
        "Generator",
        "g0",
        bus="b0",
        carrier="gas",
        p_nom=10.0,
        p_set=5.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=30.0,
    )
    net.add("Load", "l0", bus="b1", p_set=4.0)
    return net, "g0", "l0"


@pytest.fixture
def pypsa_net_with_timeseries(simple_pypsa_net):
    net, g0, l0 = simple_pypsa_net
    # Widen to 24 h so we have a longer timeseries.
    net.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
    start = datetime(2024, 1, 1)
    ts = pd.Series(
        [5.0 + i * 0.2 for i in range(24)],
        index=pd.date_range(start, periods=24, freq="h"),
    )
    timeseries = {ComponentRef("thermal", g0): ts}
    return net, g0, l0, timeseries, start


@pytest.fixture
def five_bus_pypsa_net():
    from energy_scheduling_benchmark.networks import build_toy_network

    result = build_toy_network(periods=72)
    return result.net, result.timeseries, result.start
