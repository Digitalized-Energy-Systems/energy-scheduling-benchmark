import networkx as nx
import pytest
from mango_energy_environments import connection_graph

from energy_scheduling_benchmark import datasets


def test_toy_through_pipeline():
    net = datasets.load("toy")
    kinds = {cid: c.kind for cid, c in net.components.items()}
    assert kinds == {
        "gas": "gen",
        "coal": "gen",
        "wind": "sgen",
        "load0": "load",
        "load1": "load",
        "battery": "storage",
    }
    assert len(net.time_index) == 48
    assert nx.is_connected(connection_graph(net))
    assert net.components["gas"].cost == pytest.approx((30.0, 0.11 * 30 / 100))
    assert net.components["wind"].cost[1] == 0.0
    # the recipe overrides the toy's 4 h
    assert net.components["battery"].params["max_hours"] == 2.0
    assert net.components["battery"].params["state_of_charge_initial"] == 20.0


def test_recipe_overrides():
    net = datasets.load(
        "toy", k=0.0, storage_max_hours={"battery": 3.0}, soc_initial=0.0
    )
    assert net.components["gas"].cost[1] == 0.0
    assert net.components["battery"].params["max_hours"] == 3.0
    assert net.components["battery"].params["state_of_charge_initial"] == 0.0


def test_base_s_1_elec_2019():
    net = datasets.load("base_s_1_elec_2019")
    c = net.components
    assert len(net.time_index) == 8760
    assert "DE0 0AC hydro" not in c  # no inflow model
    assert "DE0 0AC 0 offwind-dc" not in c  # p_nom = 0
    assert c["DE0 0AC biomass"].kind == "gen" and c["DE0 0AC biomass"].cost[1] > 0
    assert c["DE0 0AC PHS"].params["max_hours"] == 6.0
