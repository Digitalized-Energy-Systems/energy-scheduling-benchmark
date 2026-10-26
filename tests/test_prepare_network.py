"""``resources/networks/prepare_network.py``: one-off clean-up of a network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios._model import (
    QUADRATIC_K,
    quadratic_cost_coefficient,
)

NETWORKS = Path(__file__).resolve().parents[1] / "resources" / "networks"
_spec = importlib.util.spec_from_file_location(
    "prepare_network", NETWORKS / "prepare_network.py"
)
prepare_network = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_network)


def test_prepare_cleans_units_and_writes_costs_and_storage():
    n = build_toy_network(periods=4).net
    n.add("StorageUnit", "hydro0", bus="bus1", carrier="hydro", p_nom=10.0)
    n.add("Generator", "empty", bus="bus1", carrier="gas", p_nom=0.0)
    n.storage_units.loc["batt0", "carrier"] = "battery"

    prepare_network.prepare(n)

    assert "hydro0" not in n.storage_units.index
    assert "empty" not in n.generators.index
    g = n.generators.marginal_cost_quadratic
    assert g["thermal0"] == pytest.approx(QUADRATIC_K * 30.0 / 100.0)
    assert g["wind0"] == 0.0
    batt = n.storage_units.loc["batt0"]
    assert batt.max_hours == 2.0
    assert batt.state_of_charge_initial == pytest.approx(0.5 * 20.0 * 2.0)


def test_c2_from_the_network_wins_over_the_formula():
    statics = {"marginal_cost": 30.0, "p_nom": 100.0, "marginal_cost_quadratic": 0.5}
    assert quadratic_cost_coefficient(statics, "quadratic") == 0.5
    assert quadratic_cost_coefficient(statics, "linear") == 0.0
