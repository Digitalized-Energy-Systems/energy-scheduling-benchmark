import networkx as nx
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.world import LEADER, SLACK, build_world, topology_graph


@pytest.fixture(scope="module")
def toy():
    return datasets.load("toy")


def test_grid(toy):
    g = topology_graph(toy, "grid")
    assert set(g.nodes) == {"gas", "coal", "wind", "battery", LEADER}  # no loads
    assert nx.is_connected(g)
    assert set(g[LEADER]) == {"gas", "battery"}  # agents on the first bus (bus0)


def test_ring_with_slack(toy):
    g = topology_graph(toy, "ring", slack=True)
    assert all(d == 2 for _, d in g.degree)
    assert set(g[LEADER]) == {SLACK, "wind"}  # position 0, ring ordered by id
    assert set(g[SLACK]) == {LEADER, "battery"}


def test_complete_and_unknown(toy):
    assert topology_graph(toy, "complete").number_of_edges() == 10  # 5 nodes
    with pytest.raises(ValueError):
        topology_graph(toy, "star")


async def test_build_world_and_loss_count(toy):
    world, behavior, agents, comm = build_world(toy, "ring", loss_rate=1.0)
    assert behavior.observe(agents["gas"].aid, "statics")["kind"] == "gen"
    # ring: leader, battery, coal, gas, wind
    assert {a.aid for a in agents[LEADER].neighbors()} == {"battery", "wind"}

    async with world:
        await world.send_message("hi", agents["gas"].addr, sender_id=agents[LEADER].aid)
        await world.step_until(1.0)
    assert (comm.sent, comm.lost) == (1, 1)
