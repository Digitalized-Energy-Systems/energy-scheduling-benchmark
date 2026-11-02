"""mango world for a benchmark run: copper-plate behaviour, communication topology, lossy links.

Agents: one per network component except loads (copper plate: the leader knows the total demand),
plus a ``leader`` and, if enabled, a ``slack`` agent, both placed at the first bus / position 0.
"""

from __future__ import annotations

import random

import networkx as nx
from distributed_resource_optimization.carrier.mango_carrier import next_hops
from mango import RoleAgent
from mango.express.topology import AGENT_NODE_KEY, graph_topology
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world
from mango_energy_environments import (
    CopperPlateBehavior,
    CopperPlateNetwork,
    connection_graph,
)
from mango_energy_environments.environments.scheduling import LOAD

LEADER, SLACK = "leader", "slack"
#: Slack import price (€/MWh), export is free (Q3: SDAC harmonised maximum clearing price).
SLACK_IMPORT_PRICE = 4000.0
TOPOLOGIES = ("grid", "complete", "ring")


class CountingCommunication(SimpleCommunicationSimulation):
    """mango's loss/delay simulation that also counts sent and lost messages (per hop)."""

    sent = 0
    lost = 0

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        self.sent += len(messages)
        self.lost += sum(not r.reached for r in result.package_results)
        return result


def topology_graph(
    network: CopperPlateNetwork, kind: str, slack: bool = False
) -> nx.Graph:
    """Agent graph: component ids (no loads) + leader (+ slack) at the first bus / position 0."""
    components = network.components
    agents = sorted(cid for cid, c in components.items() if c.kind != LOAD)
    extra = [LEADER, SLACK] if slack else [LEADER]
    if kind == "complete":
        return nx.complete_graph(extra + agents)
    if kind == "ring":
        return nx.cycle_graph(extra + agents)
    if kind != "grid":
        raise ValueError(f"unknown topology {kind!r}, expected one of {TOPOLOGIES}")
    # shortcut: dropping load nodes can split the graph at a bus that hosts only loads; none of
    # the current networks has one in the middle — contract such buses when a network needs it.
    graph = connection_graph(network)
    graph.remove_nodes_from(cid for cid, c in components.items() if c.kind == LOAD)
    if not nx.is_connected(graph):
        # unroutable agents would be reached directly, bypassing the topology
        raise ValueError(
            "grid topology is disconnected (e.g. split at a load-only bus)"
        )
    first_bus = min(components[cid].bus for cid in agents)
    anchors = [cid for cid in agents if components[cid].bus == first_bus]
    graph.add_edges_from((e, a) for e in extra for a in anchors)
    if slack:
        graph.add_edge(LEADER, SLACK)
    return graph


def build_world(
    network: CopperPlateNetwork,
    topology: str = "grid",
    loss_rate: float = 0.0,
    delay_s: float = 0.01,
    seed: int = 0,
    slack: bool = False,
):
    """Create the world with one ``RoleAgent`` per topology node; callers add the roles.

    Returns ``(world, behavior, agents, communication, routes)``; ``agents`` maps node id →
    agent, ``routes`` maps aid → its routing table for the DRO roles (``routes=routes[aid]``).
    """
    behavior = CopperPlateBehavior(network)
    communication = CountingCommunication(
        loss_percent=loss_rate, default_delay_s=delay_s, rng=random.Random(seed)
    )
    world = create_world(
        start_time=0.0,
        communication_sim=communication,
        environment=DefaultEnvironment(behavior=behavior),
    )
    graph = topology_graph(network, topology, slack)
    agents = {}
    for node in graph.nodes:
        bind = {} if node in (LEADER, SLACK) else {"id": node}
        agents[node] = world.register(RoleAgent(), suggested_aid=str(node), **bind)
    topo = graph_topology(graph)
    for node, agent in agents.items():
        topo.graph.nodes[node][AGENT_NODE_KEY].add(agent)
    topo.inject()
    routes = next_hops(nx.relabel_nodes(graph, {n: a.aid for n, a in agents.items()}))
    return world, behavior, agents, communication, routes
