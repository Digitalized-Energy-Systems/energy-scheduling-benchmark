import networkx as nx
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.world import build_world, leader, topology_graph

BUSES = {"bus0", "bus1", "bus2", "bus3"}  # bus3: load only


@pytest.fixture(scope="module")
def toy():
    return datasets.load("toy")


def test_grid(toy):
    g = topology_graph(toy, "grid")
    assert (
        set(g.nodes) == BUSES
    )  # one agent per bus, incl. load-only bus3; no leader node (Q-B7)
    assert set(g.edges) == set(toy.lines)


def test_ring(toy):
    g = topology_graph(toy, "ring")
    assert set(g.nodes) == BUSES  # no slack (audit Q-A6), no leader node (Q-B7)
    assert all(d == 2 for _, d in g.degree)  # the buses form the ring


def test_ring_one_bus_has_no_self_loop(toy):
    from mango_energy_environments import THERMAL, Component, Network

    one = [1.0]
    net = Network(toy.time_index[:1], {"g": Component(THERMAL, "a", one, one)}, [])
    assert nx.number_of_selfloops(topology_graph(net, "ring")) == 0


def test_complete_and_unknown(toy):
    assert topology_graph(toy, "complete").number_of_edges() == 6  # K4 over the buses
    with pytest.raises(ValueError):
        topology_graph(toy, "star")


def test_grid_disconnected(toy):
    from mango_energy_environments import THERMAL, Component, Network

    one = [1.0]
    net = Network(
        time_index=toy.time_index[:1],
        components={
            "g0": Component(THERMAL, "a", one, one),
            "g1": Component(THERMAL, "b", one, one),
        },
        lines=[],
    )
    with pytest.raises(ValueError, match="disconnected"):
        topology_graph(net, "grid")


async def test_build_world_and_loss_count(toy):
    world, behavior, agents, comm, routes = build_world(toy, "ring", loss_rate=1.0)
    assert (
        leader(agents) is agents["bus0"]
    )  # the first bus carries the leader role (Q-B7)
    assert behavior.observe(agents["bus1"].aid, "components") == ["coal", "load1"]
    assert {a.aid for a in agents["bus0"].neighbors()} == {"bus1", "bus3"}

    async with world:
        await world.send_message(
            "hi", agents["bus1"].addr, sender_id=agents["bus0"].aid
        )
        await world.step_until(1.0)
    assert (comm.sent, comm.lost) == (1, 1)


async def test_settle_time_for_multi_hop_reports():
    from energy_scheduling_benchmark import datasets
    from energy_scheduling_benchmark.run import window
    from energy_scheduling_benchmark.world import build_world, settle_s

    toy = window(datasets.load("toy"), None, 2)
    for topology, expected in [("complete", 0.0), ("ring", 2 * 0.01)]:
        _, _, agents, _, routes = build_world(toy, topology, 0.0, 0.01)
        buses = [a.aid for a in agents.values()]
        # complete: every bus 1 hop → no wait; ring of 4 with the leader role on bus0: bus2 is 2 hops
        assert settle_s(routes, buses, agents["bus0"].aid, 0.01) == expected
