"""The benchmark's consensus agents, unchanged, on real mango TCP containers (localhost) instead of
the simulated world (audit 2026-11-12 Q-A8): toy, grid topology (routed messages), lossless.
Wall-clock time; participants iterate until the stop arrives, so the result matches the
simulated run up to the stop tolerances, not bit for bit."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket

import numpy as np
import pytest
from distributed_resource_optimization import (
    CoordinatorRole,
    StartCoordinatedDistributedOptimization,
    json_codec,
    next_hops,
)
from mango import (
    RoleAgent,
    activate,
    auto_assign,
    create_tcp_container,
    graph_topology,
    topology_to_aid_graph,
)
from mango_energy_environments import LOAD, CopperPlateBehavior

from energy_scheduling_benchmark import consensus, datasets
from energy_scheduling_benchmark.outputs import write_run
from energy_scheduling_benchmark.parameters import parameters
from energy_scheduling_benchmark.run import build_parser, run, window
from energy_scheduling_benchmark.world import leader, topology_graph


def _free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


async def test_toy_consensus_on_tcp_containers(tmp_path):
    hours = 6
    argv = [
        "--algorithm",
        "consensus",
        "--hours",
        str(hours),
        "--out",
        str(tmp_path / "sim"),
    ]
    out = await run(build_parser().parse_args(argv))
    simulated = json.loads((out / "meta.json").read_text())

    network = window(datasets.load("toy"), None, hours)
    behavior = CopperPlateBehavior(network)
    graph = topology_graph(network, "grid")
    containers = [
        create_tcp_container(("127.0.0.1", p), codec=json_codec())
        for p in _free_ports(3)
    ]
    agents = {}
    for i, node in enumerate(graph.nodes):  # spread round-robin over the containers
        agents[node] = containers[i % 3].register(RoleAgent(), suggested_aid=str(node))
        behavior.install(agents[node], bus=node)  # a simulated world does this itself
    topo = graph_topology(graph)
    auto_assign(topo, list(agents.values()))
    routes = next_hops(topology_to_aid_graph(topo))
    params = parameters(
        "consensus", "toy", "grid"
    )  # toy's manual set, as the simulated run
    args = argparse.Namespace(delay_s=0.01, loss_rate=0.0)
    outcome = consensus.setup(params, network, network, args)(behavior, agents, routes)
    host = leader(agents)
    role = next(r for r in host.roles if isinstance(r, CoordinatorRole))
    units = [cid for cid, c in network.components.items() if c.kind != LOAD]

    async with activate(*containers):
        host.handle_message(StartCoordinatedDistributedOptimization(None), {})
        await asyncio.wait_for(role.wait_done(), timeout=120)
        for _ in range(200):  # the stop still travels to every bus agent
            if set(units) <= set(behavior.schedules):
                break
            await asyncio.sleep(0.05)
    result = outcome()
    assert result.stop_reason == "converged"
    write_run(
        tmp_path / "tcp",
        behavior,
        messages={},
        inputs={},
        network_file=None,
        algorithm="consensus",
        iterations=result.iterations,
        stop_reason=result.stop_reason,
        wall_time_s=0.0,
    )
    tcp = json.loads((tmp_path / "tcp" / "meta.json").read_text())
    assert tcp["total_cost"] == pytest.approx(simulated["total_cost"], rel=1e-4)
    assert tcp["imbalance"]["shortfall_mwh"] == pytest.approx(
        simulated["imbalance"]["shortfall_mwh"], abs=0.5
    )
    assert np.isfinite(result.sim_time_s) and result.sim_time_s > 0  # wall-clock here
