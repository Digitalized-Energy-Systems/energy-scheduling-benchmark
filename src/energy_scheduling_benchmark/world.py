"""mango world for a benchmark run: copper-plate behaviour, communication topology, lossy links.

Agents: one per bus, owning all components at it incl. its load (KH12 §III-C), plus a ``leader``
and, if enabled, a ``slack`` agent: leaves on the first bus (grid, ring) or at position 0 of the
complete graph. The leader knows
no demand; it only learns it from the bus agents' messages.
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
from mango_energy_environments import CopperPlateBehavior, Network

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


def topology_graph(network: Network, kind: str, slack: bool = False) -> nx.Graph:
    """Agent graph over the buses; leader (+ slack) at position 0 (complete) or as leaves on the
    first bus (grid, ring), so the bus agents' own graph is the named topology."""
    buses = sorted({c.bus for c in network.components.values()})
    extra = [LEADER, SLACK] if slack else [LEADER]
    if kind == "complete":
        return nx.complete_graph(extra + buses)
    if kind == "ring":
        graph = nx.cycle_graph(buses)
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))  # a one-bus network
    elif kind == "grid":
        graph = nx.Graph()
        graph.add_nodes_from(buses)
        graph.add_edges_from(network.lines)
        if not nx.is_connected(graph) or len(graph) != len(buses):
            # unroutable agents would be reached directly, bypassing the topology
            raise ValueError(
                "grid topology is disconnected or has buses without components"
            )
    else:
        raise ValueError(f"unknown topology {kind!r}, expected one of {TOPOLOGIES}")
    graph.add_edges_from((e, buses[0]) for e in extra)
    if slack:
        graph.add_edge(LEADER, SLACK)
    return graph


def build_world(
    network: Network,
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
        # no per-message log (memory grew ~4 MB/iteration on PyPSA-Eur);
        # world.messages_delivered_count is kept either way
        record_messages=False,
    )
    graph = topology_graph(network, topology, slack)
    agents = {}
    for node in graph.nodes:
        bind = {} if node in (LEADER, SLACK) else {"bus": node}
        agents[node] = world.register(RoleAgent(), suggested_aid=str(node), **bind)
    topo = graph_topology(graph)
    for node, agent in agents.items():
        topo.graph.nodes[node][AGENT_NODE_KEY].add(agent)
    topo.inject()
    routes = next_hops(nx.relabel_nodes(graph, {n: a.aid for n, a in agents.items()}))
    return world, behavior, agents, communication, routes
