"""Scenario loading, mango world construction, communication topology and loss settings."""

from __future__ import annotations

import asyncio
import logging
import random
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import Any

import networkx as nx
from mango import auto_assign, complete_topology, graph_topology
from mango_energy_environments.environments.scheduling.topology import (
    TOPOLOGIES,
    TopologyCommunicationSimulation,
    build_communication_graph,
)

from energy_scheduling_benchmark import (
    ComponentRef,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    build_toy_network,
    load_scenario,
)

logger = logging.getLogger(__name__)


def resolve_scenario(network: str, *, simulate_days: int) -> ScenarioData:
    """Resolve the CLI ``--network`` argument into a :class:`ScenarioData` bundle.

    ``"toy"`` builds the built-in 5-bus fixture sized to the simulated window;
    anything else (PyPSA example name, ``.nc``/``.h5``/``.xlsx``/CSV path, or
    PYPOWER case) goes through :func:`load_scenario`.

    :param network: Value of the ``--network`` CLI argument.
    :param simulate_days: Number of simulated days, used to size the toy network.
    """
    if network == "toy":
        return build_toy_network(periods=simulate_days * 24)
    return load_scenario(network)


def resolve_loss_rate(loss_rate: float = 0.0) -> float:
    """Return the message-loss probability (a fraction in ``[0, 1]``).

    :raises ValueError: If the rate lies outside ``[0, 1]``.
    """
    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError(f"loss_rate must be a fraction in [0, 1], got {loss_rate!r}.")
    return float(loss_rate)


#: Per world: agent id -> the component it was registered for (see
#: :func:`register_component_agent`); :func:`install_topology` reads it.
_REFS_BY_WORLD: weakref.WeakKeyDictionary[Any, dict[str, ComponentRef]] = (
    weakref.WeakKeyDictionary()
)


def register_component_agent(world: Any, agent: Any, ref: ComponentRef) -> Any:
    """Register *agent* as ``ref.component_id`` and bind it to *ref* in the environment.

    mango's ``world.register`` forwards extra keyword arguments to the
    environment's ``install`` hook, so a single call both registers the agent
    and installs it for *ref* (a separate ``world.environment.install`` would
    run the hook a second time, the first time without ``id``).
    """
    registered = world.register(agent, suggested_aid=ref.component_id, id=ref)
    _REFS_BY_WORLD.setdefault(world, {})[agent.aid] = ref
    return registered


@dataclass(frozen=True)
class CommunicationTopology:
    """A scenario's communication graph, built once by :func:`participant_graph`.

    ``graph`` holds the algorithm participants only (diameter / degrees for
    the stop rules, DEED-ADMM's β, exact diffusion's Perron scaling);
    ``full`` adds every other component that talks (loads, units outside the
    algorithm) and is what :func:`install_topology` routes messages over.
    ``full`` is ``None`` for ``complete`` (every pair is one hop).
    """

    kind: str
    graph: nx.Graph
    full: nx.Graph | None
    participants: tuple[ComponentRef, ...]
    diameter: int
    max_degree: int

    @property
    def degrees(self) -> dict[ComponentRef, int]:
        return dict(self.graph.degree())

    def hops_from(self, ref: ComponentRef) -> int:
        """Largest number of hops from *ref* to any component (1 on ``complete``)."""
        if self.full is None or ref not in self.full:
            return 1
        lengths = nx.single_source_shortest_path_length(self.full, ref)
        return max(1, max(lengths.values(), default=1))


def participant_graph(
    scenario: ScenarioData,
    kind: str,
    participants: Sequence[ComponentRef],
    attached: Sequence[ComponentRef] = (),
) -> CommunicationTopology:
    """Build the communication graph of *participants* (and *attached*
    components, which connect via their bus) under topology *kind*, before any
    agent exists. Pass the result to :func:`install_topology`."""
    participants = list(participants)
    attached = [r for r in attached if r not in set(participants)]
    if kind not in TOPOLOGIES:
        raise ValueError(f"unknown topology {kind!r}; choose from {TOPOLOGIES}")
    if kind == "complete":
        graph = nx.complete_graph(participants)
        full = None
    else:
        full = build_communication_graph(kind, scenario.net, participants, attached)
        graph = full.subgraph(participants).copy()
    diameter = nx.diameter(graph) if graph.number_of_nodes() > 1 else 1
    max_degree = max((d for _, d in graph.degree()), default=0)
    return CommunicationTopology(
        kind=kind,
        graph=graph,
        full=full,
        participants=tuple(participants),
        diameter=max(1, diameter),
        max_degree=max_degree,
    )


def install_topology(
    world: Any,
    topology: CommunicationTopology,
    participants: Sequence[Any],
    *,
    logical: Sequence[Any] | None = None,
) -> None:
    """Connect the algorithm agents along *topology* and route every message.

    *participants* are the agents (registered with
    :func:`register_component_agent`) of ``topology.participants``. The
    algorithm's neighbours are their edges, and the world's communication
    simulation delivers every message (also leader <-> agent) along the
    shortest path of ``topology.full``: hops x delay, lost with
    ``1 - (1 - loss)^hops``. With ``complete`` every pair is one hop and the
    run is unchanged.

    :param logical: agents that instead get a complete mango topology among
        themselves (ADMM: the coordinator must reach every participant); the
        messages still travel over ``topology.full``.
    """
    algorithm_agents = list(logical if logical is not None else participants)
    if topology.full is None:
        if algorithm_agents:
            auto_assign(complete_topology(len(algorithm_agents)), algorithm_agents)
        return
    refs = _REFS_BY_WORLD.get(world, {})
    part_refs = [refs[a.aid] for a in participants]
    if set(part_refs) != set(topology.participants):
        raise ValueError(
            "install_topology: the agents do not match topology.participants"
        )
    unknown = [r for r in refs.values() if r not in topology.full]
    if unknown:
        logger.warning(
            "Topology %s: %d registered components are not in the graph (one hop): %s",
            topology.kind,
            len(unknown),
            unknown[:5],
        )
    aid_by_ref = {ref: aid for aid, ref in refs.items()}
    world.communication_sim.set_graph(
        nx.relabel_nodes(topology.full, aid_by_ref, copy=True)
    )
    logger.info(
        "Topology %s: %d participants (diameter %d), %d components, %d links",
        topology.kind,
        len(part_refs),
        topology.diameter,
        topology.full.number_of_nodes(),
        topology.full.number_of_edges(),
    )
    if not algorithm_agents:
        return
    if logical is not None:
        auto_assign(complete_topology(len(algorithm_agents)), algorithm_agents)
        return
    index = {ref: i for i, ref in enumerate(part_refs)}
    logical_graph = nx.Graph()
    logical_graph.add_nodes_from(range(len(part_refs)))
    logical_graph.add_edges_from((index[u], index[v]) for u, v in topology.graph.edges)
    auto_assign(graph_topology(logical_graph), algorithm_agents)


class AgentError(RuntimeError):
    """An agent raised during the simulation; the run was aborted."""


class _AgentErrorWatch(logging.Handler):
    """Cancels the running simulation on the first exception mango logs.

    mango only logs exceptions from agents: a failed scheduled task
    ("got exception in scheduled event") or a failed message handler, which
    also ends that agent's inbox loop so its peers wait for it forever. Either
    way the world would keep stepping until the end of the horizon (or never
    finish a distributed run). Only records that carry an exception count.
    """

    def __init__(self, task: asyncio.Future) -> None:
        super().__init__(level=logging.ERROR)
        self.task = task
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info is None or record.exc_info[0] is None:
            return
        if issubclass(record.exc_info[0], asyncio.CancelledError):
            return
        self.records.append(record)
        if len(self.records) == 1:
            self.task.cancel()


async def run_simulation(world: Any, seconds: float) -> None:
    """``discrete_step_until(world, seconds)`` that fails fast.

    Steps the world exactly like :func:`mango.simulation.world.discrete_step_until`
    (results are unchanged) but raises :class:`AgentError` with the first
    agent exception as soon as mango logs one, instead of simulating on.
    """
    from mango.simulation.world import discrete_step_until

    # The stepping runs in a child task, which is what the watch cancels: the
    # caller's own task is never cancelled, so no cancellation can be left
    # pending after the error has been turned into AgentError.
    stepping = asyncio.ensure_future(discrete_step_until(world, seconds))
    watch = _AgentErrorWatch(stepping)
    mango_logger = logging.getLogger("mango")
    mango_logger.addHandler(watch)
    try:
        await stepping
    except asyncio.CancelledError:
        if not watch.records or not stepping.cancelled():
            raise
    finally:
        mango_logger.removeHandler(watch)
    if watch.records:
        first = watch.records[0]
        exc = first.exc_info[1] if first.exc_info else None
        raise AgentError(
            f"agent exception during the simulation (t = "
            f"{getattr(getattr(world, 'clock', None), 'time', '?')} s): "
            f"{first.getMessage()}: {exc!r}"
        ) from exc


def build_world(
    scenario: ScenarioData, *, delay_s: float, loss_rate: float = 0.0, seed: int = 0
) -> tuple[Any, PyPSABehavior]:
    """Build the mango simulation ``world`` + environment behavior for *scenario*.

    Every scenario wires the same environment/communication-sim/world triple;
    this is that wiring in one place.

    :param scenario: Scenario bundle to drive, already clipped to the
        simulated window (see :func:`_clip_scenario`).
    :param delay_s: Comms delay (seconds) for :class:`SimpleCommunicationSimulation`.
    :param loss_rate: Probability in ``[0, 1]`` that a message is lost. mango's
        ``SimpleCommunicationSimulation`` names this argument ``loss_percent``
        but treats it as a fraction.
    :param seed: Seed for the loss draws, so lossy runs are reproducible.
    """
    from mango.simulation.environment import DefaultEnvironment
    from mango.simulation.world import create_world

    loss_rate = resolve_loss_rate(loss_rate)
    logger.info(
        "Communication: delay %.4g s, loss rate %.4g, seed %d", delay_s, loss_rate, seed
    )
    behavior = PyPSABehavior.from_scenario(scenario)
    # Without a graph (topology "complete") this equals mango's
    # SimpleCommunicationSimulation draw for draw; install_topology sets one.
    com_sim = TopologyCommunicationSimulation(
        delay_s=delay_s, loss_rate=loss_rate, rng=random.Random(seed)
    )
    # No scenario reads world.recorded_messages, and keeping every message's
    # content for the whole run dominates memory (OOM on lower_saxony_2030).
    world = create_world(
        start_time=0.0,
        communication_sim=com_sim,
        environment=DefaultEnvironment(behavior=behavior),
        record_messages=False,
    )
    return world, behavior


@dataclass(frozen=True)
class LossTolerance:
    """Timeouts that let a scenario run through lost messages.

    All in simulation seconds.  Without loss none of them fires: rounds and
    requests complete after one or two message delays, well before their
    deadline.
    """

    #: Peer-to-peer round deadline / ADMM per-round answer timeout. ``None``
    #: in lossless runs: rounds wait for every neighbour, as in the papers.
    round_timeout: float | None
    #: Deadline for applying an hour whose load reports are incomplete.
    load_timeout: float
    #: Timeout of one leader request (kick-off, schedule/report poll).
    request_timeout: float
    #: Interval between the leader's polls for missing results. ``None`` in
    #: lossless runs: nothing can go missing, and a long run must not be
    #: given up after ``result_retries`` polls.
    result_poll_s: float | None
    #: Polls before a missing result is given up (degraded run).
    result_retries: int
    #: Consecutive rounds a participant may miss before the ADMM coordinator
    #: stops the run with ``converged = False`` (``None``: never).
    max_missing_rounds: int | None = None
    #: Whether round / load timeouts were given explicitly (not scaled by
    #: :meth:`for_hops`).
    explicit_round_timeout: bool = False
    explicit_load_timeout: bool = False

    def for_hops(self, hops: int, *, coordinator: bool = False) -> LossTolerance:
        """Deadlines for messages that travel up to *hops* hops each way.

        On a sparse topology a request from the leader and its reply take
        ``2 x hops`` message delays, so the default load / request deadlines
        (and, for a *coordinator* that waits for every participant, the round
        deadline) are multiplied by *hops*. Peer-to-peer rounds only wait for
        neighbours one hop away and keep their deadline. Explicitly given
        timeouts are left as they are.
        """
        if hops <= 1:
            return self
        round_timeout = self.round_timeout
        if (
            coordinator
            and round_timeout is not None
            and not self.explicit_round_timeout
        ):
            round_timeout = round_timeout * hops
        return _dataclass_replace(
            self,
            round_timeout=round_timeout,
            load_timeout=(
                self.load_timeout
                if self.explicit_load_timeout
                else self.load_timeout * hops
            ),
            request_timeout=self.request_timeout * hops,
        )


def loss_tolerance(
    delay_s: float,
    *,
    loss_rate: float = 0.0,
    round_timeout: float | None = None,
    result_poll_s: float = 300.0,
    result_retries: int = 12,
    load_timeout: float | None = None,
    max_missing_rounds: int | None = 20,
) -> LossTolerance:
    """Default :class:`LossTolerance` for a comms delay of *delay_s*.

    Round, load and request timeouts default to three message delays (a
    request plus its reply take two); at least 1 ms, so a zero-delay world
    still gets a positive deadline. Lossless runs (``loss_rate == 0``) get no
    round deadline unless *round_timeout* is given: the algorithms then run
    the papers' synchronous rounds and plain neighbour mixing (the running-sum
    flows are only needed under loss), and no result polls (a run that takes
    longer than ``result_retries x result_poll_s`` would otherwise be given
    up and zero-filled). *load_timeout* overrides the default
    load deadline; *max_missing_rounds* is passed to the ADMM coordinator.
    """
    base = max(float(delay_s), 1e-3)
    if round_timeout is not None:
        resolved_round_timeout: float | None = float(round_timeout)
    elif loss_rate > 0.0:
        resolved_round_timeout = 3.0 * base
    else:
        resolved_round_timeout = None
    return LossTolerance(
        round_timeout=resolved_round_timeout,
        load_timeout=3.0 * base if load_timeout is None else float(load_timeout),
        request_timeout=3.0 * base,
        result_poll_s=float(result_poll_s) if loss_rate > 0.0 else None,
        result_retries=int(result_retries),
        max_missing_rounds=(
            None if max_missing_rounds is None else int(max_missing_rounds)
        ),
        explicit_round_timeout=round_timeout is not None,
        explicit_load_timeout=load_timeout is not None,
    )


def _clip_scenario(scenario: ScenarioData, simulate_days: int) -> ScenarioData:
    """Return a copy of *scenario* with timeseries trimmed to *simulate_days* × 24 hours.

    Limits both the number of tasks scheduled by PyPSABehavior.initialize() and
    the optimization horizon used by all scenario algorithms, keeping runtimes
    proportional to the simulated window regardless of how long the underlying
    network's snapshot range is.
    """
    n = simulate_days * 24
    clipped = {k: v.iloc[:n] for k, v in scenario.timeseries.items()}
    return _dataclass_replace(scenario, timeseries=clipped)
