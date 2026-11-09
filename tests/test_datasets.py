import networkx as nx
import numpy as np
import pytest
from mango_energy_environments import connection_graph, load_network

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
    wind = net.components["wind"]
    assert wind.cost[1] == pytest.approx(
        datasets.CURVATURE_BAND / (2 * wind.params["p_nom"])
    )
    # the recipe overrides the toy's 4 h
    assert net.components["battery"].params["max_hours"] == 2.0
    assert net.components["battery"].params["state_of_charge_initial"] == 20.0


def test_recipe_overrides():
    net = datasets.load(
        "toy", k=0.0, band=0.0, storage_max_hours={"battery": 3.0}, soc_initial=0.0
    )
    assert all(c.cost[1] == 0.0 for c in net.components.values())  # linear model
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


def test_lower_saxony_2030():
    net = datasets.load("lower_saxony_2030")
    kinds = [c.kind for c in net.components.values()]
    assert len(net.time_index) == 8760
    assert kinds.count("load") == 67
    assert not any(c.params.get("carrier") == "hydro" for c in net.components.values())
    assert nx.is_connected(connection_graph(net))


@pytest.mark.parametrize("name", ["toy", *datasets.NETWORKS])
def test_curvature_band_keeps_merit_order(name):
    """Renewables/storage at full output stay cheaper than the cheapest priced thermal unit."""
    comps = datasets.load(name).components.values()
    cheapest = min(c.cost[0] for c in comps if c.kind == "gen" and c.cost[0] > 0)
    top = max(
        c.cost[0] + datasets.CURVATURE_BAND
        for c in comps
        if c.kind in ("sgen", "storage")
    )
    assert top < cheapest


@pytest.mark.parametrize("name", ["toy", *datasets.NETWORKS])
def test_prepared_file_matches_recipe(name):
    """Committed resources/prepared/<name>.nc = today's recipe; else rerun esb-prepare."""
    saved = load_network(datasets.prepared_path(name))
    fresh = datasets.load(name)
    stale = f"resources/prepared/{name}.nc is stale: esb-prepare --network {name}"
    assert saved.time_index.equals(fresh.time_index), stale
    assert saved.lines == fresh.lines, stale
    assert list(saved.components) == list(fresh.components), stale
    for cid, f in fresh.components.items():
        s = saved.components[cid]
        for field in ("kind", "bus", "cost", "params"):
            assert getattr(s, field) == getattr(f, field), stale
        assert np.array_equal(s.p_min, f.p_min, equal_nan=True), stale
        assert np.array_equal(s.p_max, f.p_max, equal_nan=True), stale


@pytest.mark.parametrize("name", ["synthetic_10", "synthetic_50"])
def test_synthetic_balanced_and_varied(name):
    """Pick rule (TODO Next #8a): bus Σ 1/(2c₂) within 2× with storage on and off; all mixes."""
    comps = load_network(datasets.prepared_path(name)).components.values()
    for kinds in (("gen", "sgen", "storage"), ("gen", "sgen")):
        sens = {}
        for c in comps:
            if c.kind in kinds:
                sens[c.bus] = sens.get(c.bus, 0.0) + 1 / (2 * c.cost[1])
        assert max(sens.values()) / min(sens.values()) < 2
    assert {c.kind for c in comps} == {"gen", "sgen", "storage", "load"}
    assert len({c.bus for c in comps if c.kind == "load"}) < len(
        sens
    )  # load-free buses
