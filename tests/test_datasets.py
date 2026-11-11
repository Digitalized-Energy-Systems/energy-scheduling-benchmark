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
    # band only on storage (S1: wind is pre-dispatched, c₂ = 0); battery c₁ = ageing cost (S2)
    assert net.components["wind"].cost[1] == 0.0
    bat = net.components["battery"]
    assert bat.cost == pytest.approx(
        (65_000 / (6_000 * 0.97), datasets.CURVATURE_BAND / (2 * bat.params["p_nom"]))
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
    assert c["DE0 0AC PHS"].cost[0] == 0.22  # Zakeri & Syri 2015 Table A1 variable O&M


def test_lower_saxony_2025():
    net = datasets.load("lower_saxony_2025")
    kinds = [c.kind for c in net.components.values()]
    carriers = {c.params.get("carrier") for c in net.components.values()}
    assert len(net.time_index) == 8760 and net.time_index[0].year == 2025
    assert kinds.count("load") == 67
    assert "hydro" not in carriers
    assert not any(
        str(x).startswith("offwind") for x in carriers
    )  # dropped in the build
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
    """Pick rule (TODO Next #8a): bus Σ 1/(2c₂) within 2× with storage on and off; all mixes.

    Checked as at the pick (build_synthetic.py: renewables p_nom/band). Since S1 renewables are
    pre-dispatched (c₂ = 0) and the dispatchable buses are no longer balanced (max/min 10 / 201);
    the networks stay frozen (Jonas, 2026-11-10), S3 per-agent steps handle the spread."""
    comps = load_network(datasets.prepared_path(name)).components.values()
    for kinds in (("gen", "sgen", "storage"), ("gen", "sgen")):
        sens = {}
        for c in comps:
            if c.kind in kinds:
                s = (
                    c.params["p_nom"] / datasets.CURVATURE_BAND
                    if c.kind == "sgen"
                    else 1 / (2 * c.cost[1])
                )
                sens[c.bus] = sens.get(c.bus, 0.0) + s
        assert max(sens.values()) / min(sens.values()) < 2
    assert {c.kind for c in comps} == {"gen", "sgen", "storage", "load"}
    unit_buses = {c.bus for c in comps if c.kind != "load"}
    load_buses = {c.bus for c in comps if c.kind == "load"}
    assert len(load_buses) < len(unit_buses)  # load-free buses


def test_simbench_lower_saxony():
    """Regression: SimBench's local-time labels (DST gap) once gave NaN hours that dropped every
    renewable unit; load_simbench now uses a regular 15-min index."""
    net = datasets.load("simbench_lower_saxony")
    comps = net.components.values()
    assert len(net.time_index) == 8784  # 2016, hourly
    # 29 sgens: 27 wind/mixed renewables + 2 biomass, which are thermal (MEE, S1 plan D10)
    assert sum(c.kind == "sgen" for c in comps) == 27
    assert not any(np.isnan(c.p_max).any() for c in comps)


def test_typical_days():
    """Standard day = closest to the season median, tuning day = second-closest (README)."""
    days = datasets.typical_days(load_network(datasets.prepared_path("synthetic_10")))
    assert days["winter"] == ("2019-01-16", "2019-01-09")
    assert all(standard != tuning for standard, tuning in days.values())
