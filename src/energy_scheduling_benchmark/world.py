"""mango world for a benchmark run: copper-plate behaviour, communication topology, lossy links.

Agents: one per bus, owning all components at it incl. its load (KH12 §III-C), plus a ``leader``
and, if enabled, a ``slack`` agent: leaves on the first bus (grid, ring) or at position 0 of the
complete graph. The leader knows
no demand; it only learns it from the bus agents' messages.
"""

from __future__ import annotations

import random

import networkx as nx
from distributed_resource_optimization import ControlMessage, next_hops, unwrap
from mango import RoleAgent, auto_assign, graph_topology, topology_to_aid_graph
from mango.simulation.communication import PackageResult, SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world
from mango_energy_environments import CopperPlateBehavior, Network

LEADER, SLACK = "leader", "slack"
TOPOLOGIES = ("grid", "complete", "ring")


class CountingCommunication(SimpleCommunicationSimulation):
    """mango's loss/delay simulation that also counts sent and lost messages (per hop), and never
    loses a DRO ``ControlMessage`` (start, stop, converged reports; also inside a routed
    envelope). Only the algorithms' own messages (consensus λ, dummy offers) are lossy — the
    benchmark measures the algorithm under lossy links (KH12 §IV), not our self-made protocol (S5
    Q-L2, Jonas 2026-11-11; deviation from M4 "all messages lossy", deviations.md C8). The DRO
    marker replaced a hard-coded list of message classes (audit 2026-11-12 Q-A4).

    Kept (decision 2026-11-09, TODO 14d): mango counts only delivered messages
    (``world.messages_delivered_count``); sent/lost exist only as trace events, and tracing
    is a debugging option (``esb-run --trace``), too slow for benchmark runs.
    """

    sent = 0
    lost = 0

    def calculate_communication(self, current_time, messages):
        result = super().calculate_communication(current_time, messages)
        for i, (message, r) in enumerate(zip(messages, result.package_results)):
            content = unwrap(
                message.content[0]
            )  # MessagePackage.content = (content, meta)
            if not r.reached and isinstance(content, ControlMessage):
                result.package_results[i] = PackageResult(True, r.delay_s)
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
    trace=None,
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
        trace=trace,  # mango tracing (a trace file path) for debugging, off by default
    )
    graph = topology_graph(network, topology, slack)
    agents = {}
    for node in graph.nodes:
        bind = {} if node in (LEADER, SLACK) else {"bus": node}
        agents[node] = world.register(RoleAgent(), suggested_aid=str(node), **bind)
    topo = graph_topology(graph)
    # agents are in graph-node order: one per node
    auto_assign(topo, list(agents.values()))
    routes = next_hops(topology_to_aid_graph(topo))
    return world, behavior, agents, communication, routes


def longest_hops(routes: dict, buses: list[str], leader: str) -> int:
    """Most hops between the leader and a bus agent (start messages and reports travel them)."""

    def hops(aid: str) -> int:
        n = 0
        while aid != leader:
            aid, n = routes[aid][leader], n + 1
        return n

    return max((hops(b) for b in buses), default=1)


def settle_s(routes: dict, buses: list[str], leader: str, delay_s: float) -> float:
    """Longest report delay to the leader (hops × *delay_s*) for a coordinator's stop; 0 when
    every bus is a neighbour (equal delays). Without it a late "not converged" report on a longer
    route could be overtaken by a near "converged" one and the leader would stop on a stale
    snapshot (audit 2026-11-11; LS2025 ring: up to 34 hops)."""
    longest = longest_hops(routes, buses, leader)
    return 0.0 if longest <= 1 else longest * delay_s
