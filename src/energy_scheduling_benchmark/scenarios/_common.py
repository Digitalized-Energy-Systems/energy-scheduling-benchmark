"""Shared boilerplate for the distributed-optimization scenarios.

Consensus, ADMM, diffusion, FDGDM, and DEED-ADMM all follow the same shape:
load agents push per-timestep demand to a leader, the leader kicks off a
distributed optimization run exactly once, and once every generator reports
back it applies the resulting per-timestep schedule as the simulation clock
advances. This module holds that shared shape so each scenario only needs to
supply the algorithm-specific pieces (actor construction, start-message
payload, and the attribute path to read the finished schedule from).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import subprocess
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
    RoleLocalMessage,
)
from mango import Role, auto_assign, complete_topology, graph_topology, sender_addr
from mango_energy_environments.environments.scheduling.topology import (
    TOPOLOGIES,
    TopologyCommunicationSimulation,
    build_communication_graph,
)

from energy_scheduling_benchmark import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PowerUpdateInfo,
    PyPSABehavior,
    tuned_params,
)
from energy_scheduling_benchmark.dispatch import SlackSpec
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.plotting import (
    cost_over_time,
    generation_vs_demand,
    order_carriers,
    per_unit_small_multiples,
    resolve_carrier_colors,
    stacked_area,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_toy_network",
    "load_scenario",
    "resolve_scenario",
    "build_behavior",
    "build_world",
    "register_component_agent",
    "resolve_loss_rate",
    "_clip_scenario",
    "_lookup_ts",
    "compute_overall_cost",
    "compute_slack",
    "SlackSpec",
    "build_group_map",
    "carrier_style",
    "build_scenario_argparser",
    "LossTolerance",
    "loss_tolerance",
    "generator_agent",
    "filter_and_cache_statics",
    "collect_generator_refs",
    "build_demand_horizon",
    "DemandHorizon",
    "build_p_max_vec",
    "capacity_scaled_epsilon",
    "EpsilonScaling",
    "StorageParams",
    "write_scenario_outputs",
    "run_scenario_main",
    "PowerLoadInfo",
    "OptimizationFinishedInfo",
    "ScheduleRequest",
    "KickOff",
    "KickOffAck",
    "PowerLoadMonitoring",
    "PowerLoadAggregator",
    "GeneratorResultRole",
    "make_finish_callback",
]


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


def build_behavior(scenario: ScenarioData) -> PyPSABehavior:
    """Return the environment behavior driving *scenario*'s PyPSA network.

    :param scenario: Scenario bundle from :func:`resolve_scenario`/:func:`load_scenario`.
    """
    return PyPSABehavior.from_scenario(scenario)


#: Cost models of ``--cost-model`` (decision Q2).
COST_MODELS = ("linear", "quadratic")

#: Quadratic cost factor k of ``c₂ = k · c₁ / p_nom`` (decision Q17): a unit's
#: marginal cost rises from c₁ at 0 MW to (1 + 2k)·c₁ at full output. 0.11 is
#: the mean of c·P_max/b over the three units of Wood & Wollenberg, *Power
#: Generation, Operation and Control*, Example 4E (0.091, 0.129, 0.123).
QUADRATIC_K = 0.11


def quadratic_cost_coefficient(statics: dict, cost_model: str) -> float:
    """``c₂`` (€/MW²h) of one generator for *cost_model* (0 for ``linear``).

    ``c₂ = QUADRATIC_K · max(c₁, 0) / p_nom``; zero-cost units (renewables) and
    units without ``p_nom`` stay linear. Storage is not passed here (its cost
    stays linear, decision Q15).
    """
    if cost_model not in COST_MODELS:
        raise ValueError(f"cost_model must be one of {COST_MODELS}, got {cost_model!r}")
    if cost_model == "linear":
        return 0.0
    c1 = max(float(statics.get("marginal_cost", 0.0)), 0.0)
    p_nom = float(statics.get("p_nom", 0.0))
    return QUADRATIC_K * c1 / p_nom if p_nom > 0 else 0.0


def thermal_cost_quad(
    gen_refs: Sequence[ComponentRef], statics_by_ref: dict, cost_model: str
) -> dict[str, float]:
    """``{component_id: c₂}`` of the thermal units with a quadratic term.

    Empty for ``cost_model="linear"``. Only thermal units get c₂ (heat-rate
    curves, decision Q17); renewables and storage stay linear everywhere.
    """
    out: dict[str, float] = {}
    for ref in gen_refs:
        if ref.element_type != THERMAL:
            continue
        c2 = quadratic_cost_coefficient(statics_by_ref[ref], cost_model)
        if c2 > 0:
            out[ref.component_id] = c2
    return out


def apply_cost_model(
    eps: EpsilonScaling,
    gen_refs: Sequence[ComponentRef],
    statics_by_ref: dict,
    cost_model: str,
) -> tuple[EpsilonScaling, dict[str, float]]:
    """Price-response ε for *cost_model* plus the c₂ table for the cost metric.

    With quadratic costs a thermal unit's ε is its true curvature 2c₂ (Jian et
    al. eq. 23: P = (λ − b)/(2a)) instead of the capacity-scaled proxy; all
    other units keep the proxy. Linear: returns *eps* unchanged and ``{}``.
    """
    cost_quad = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    if not cost_quad:
        return eps, {}
    eps_by_aid = dict(eps.eps_by_aid)
    for aid, c2 in cost_quad.items():
        eps_by_aid[aid] = 2.0 * c2
    return _dataclass_replace(eps, eps_by_aid=eps_by_aid), cost_quad


def price_sensitivity(
    eps: EpsilonScaling, gen_refs: Sequence[ComponentRef], statics_by_ref: dict
) -> float:
    """Σ 1/εᵢ (MW per €/MWh) over *gen_refs*; storage uses band / p_nom."""
    total = 0.0
    for ref in gen_refs:
        e = eps.eps_by_aid.get(ref.component_id)
        if e is None:
            e = eps.target_band / max(float(statics_by_ref[ref].get("p_nom", 0.0)), 1.0)
        total += 1.0 / e
    return total


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
    behavior = build_behavior(scenario)
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


def filter_and_cache_statics(
    behavior: PyPSABehavior, refs: Sequence[ComponentRef]
) -> tuple[list[ComponentRef], dict[ComponentRef, dict]]:
    """Drop components with zero or non-finite nominal power.

    Returns the surviving refs plus a ``{ref: statics}`` cache. ``get_statics``
    rebuilds the component's full observer registry on every call, so scenarios
    that need a component's static row more than once should fetch it here
    once and reuse the returned dict instead of calling ``get_statics`` again.

    :param behavior: Environment behavior to query, from :func:`build_behavior`.
    :param refs: Candidate component refs (e.g. from ``get_components_by_type``).
    """
    statics_by_ref: dict[ComponentRef, dict] = {}
    kept: list[ComponentRef] = []
    for ref in refs:
        statics = behavior.get_statics(ref)
        p_nom = statics.get("p_nom", 0.0)
        if not math.isfinite(p_nom) or p_nom == 0.0:
            continue
        statics_by_ref[ref] = statics
        kept.append(ref)
    return kept, statics_by_ref


def collect_generator_refs(
    behavior: PyPSABehavior, *, exclude_hydro: bool = True
) -> tuple[list[ComponentRef], dict[ComponentRef, dict]]:
    """Collect thermal/renewable/storage refs, filter, and cache their statics.

    *exclude_hydro* drops any component whose id contains ``"hydro"`` — hydro
    dispatch is driven by natural inflow, which none of the distributed
    scenarios' storage/generator actors model (they assume freely schedulable
    charge/discharge or a pure cost curve), so hydro units cannot participate
    meaningfully. This is a substring match on ``component_id``, not a
    carrier/type check, so a plant merely named e.g. ``"...hydrogen..."``
    would also be excluded — a known sharp edge, not a deliberate rule.

    Also drops zero/non-finite ``p_nom`` components via
    :func:`filter_and_cache_statics`.

    :param behavior: Environment behavior to query.
    :param exclude_hydro: Whether to apply the hydro name filter.
    """
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    if exclude_hydro:
        kept = [ref for ref in gen_refs if "hydro" not in ref.component_id]
        n_dropped = len(gen_refs) - len(kept)
        if n_dropped:
            logger.info(
                "collect_generator_refs: excluded %d component(s) by hydro name match",
                n_dropped,
            )
        gen_refs = kept
    return filter_and_cache_statics(behavior, gen_refs)


@dataclass
class DemandHorizon:
    """Aggregated demand vector + time bookkeeping shared by every scenario."""

    time_index: pd.Index
    #: Number of timesteps in the horizon (``len(time_index)``).
    horizon: int
    #: Aggregated (summed across loads) demand per timestep, MW.
    target_series: np.ndarray
    #: Maps rounded mango simulation seconds → index into ``target_series``.
    time_to_index: dict[float, int]
    start_dt: Any


def build_demand_horizon(
    behavior: PyPSABehavior,
    scenario: ScenarioData,
    load_refs: Sequence[ComponentRef],
    *,
    simulate_days: int | None,
) -> DemandHorizon:
    """Build the aggregated demand vector and simulation-time → index map.

    Sums each load's timeseries (reindexed onto the first load's index) into
    a single per-timestep demand vector, and maps mango's simulation-clock
    seconds to an index into that vector (both keyed by
    ``round(seconds, 6)`` — the two must agree on that rounding or a
    timestep's dispatch is silently dropped, see
    :meth:`PowerLoadAggregator._handle_load_info`).

    :param load_refs: Load component refs, e.g. from
        ``behavior.get_components_by_type([LOAD])``.
    :param simulate_days: If given, ``time_index`` is truncated to
        ``simulate_days * 24`` hours (matches the window ``_clip_scenario``
        already trimmed the timeseries to). Pass ``None`` to use the full
        length of the first load's timeseries as-is.
    :raises RuntimeError: If the first load's timeseries, or any load's
        timeseries, cannot be found in *scenario*.
    """
    load_series_0 = _lookup_ts(scenario, load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    if simulate_days is not None:
        time_index = time_index[: simulate_days * 24]
    horizon = len(time_index)

    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(scenario, ref)
        if s is None:
            raise RuntimeError(f"Load timeseries missing for {ref}.")
        target_series += np.asarray(s.reindex(time_index), dtype=float)

    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

    return DemandHorizon(time_index, horizon, target_series, time_to_index, start_dt)


def build_p_max_vec(
    scenario: ScenarioData,
    ref: ComponentRef,
    statics: dict,
    time_index: pd.Index,
    horizon: int,
) -> np.ndarray:
    """Build a per-timestep max-power vector for a generator, aligned to *time_index*.

    Falls back to the static ``p_max_pu * p_nom`` (e.g. nuclear at 0.871) when
    *ref* has no timeseries; a timeseries overrides it, as in PyPSA.
    The timeseries is a per-unit availability (``generators_t.p_max_pu``,
    PyPSA convention) and is scaled by ``p_nom`` — for thermal units too
    (e.g. scigrid-de's wind units, classified thermal by carrier) — except
    where ``extract_timeseries`` fell back to ``generators_t.p_set``, in MW.
    """
    p_nom = float(statics.get("p_nom", 0.0))
    ts = _lookup_ts(scenario, ref)
    if ts is None:
        return np.full(horizon, p_nom * float(statics.get("p_max_pu", 1.0)))
    values = np.asarray(ts.reindex(time_index), dtype=float)
    gens_t = getattr(scenario.net, "generators_t", None)
    cid = ref.component_id
    in_mw = (
        gens_t is not None
        and cid in getattr(gens_t, "p_set", pd.DataFrame()).columns
        and cid not in getattr(gens_t, "p_max_pu", pd.DataFrame()).columns
    )
    return values if in_mw else values * p_nom


@dataclass
class EpsilonScaling:
    """Result of :func:`capacity_scaled_epsilon`."""

    #: Per-generator price-response epsilon, keyed by ``component_id``
    #: (non-storage refs only).
    eps_by_aid: dict[str, float]
    #: The shared price-response band width (epsilon * p_nom) all epsilons
    #: were derived from.
    target_band: float
    #: ``max(costs) - min(costs)`` across non-storage refs (``0.0`` if fewer
    #: than 2).
    cost_range: float
    #: Marginal costs of the non-storage refs the band was computed from
    #: (empty if fewer than 2).
    costs_all: list[float]


def capacity_scaled_epsilon(
    gen_refs: Sequence[ComponentRef],
    statics_by_ref: dict[ComponentRef, dict],
    *,
    default_epsilon: float = 0.1,
    cost_fraction: float = 0.1,
) -> EpsilonScaling:
    """Per-generator epsilon (capacity-scaled), shared by consensus/diffusion.

    A single shared epsilon gives every generator the same price-response
    band width (epsilon * p_nom) above its own marginal cost. For merit
    order to hold, that band must be small relative to the spread of
    marginal costs across generators — otherwise many generators are still
    in their partial "ramp" region at the clearing price simultaneously, so
    power gets spread roughly by capacity rather than sorted by cost. On
    real PyPSA-Eur networks generator capacities span orders of magnitude
    (tens to tens-of-thousands of MW); scaling the band off typical/mean
    capacity makes it far wider than the cost spread, breaking merit order.
    Scale epsilon inversely with p_nom so every generator's band is a small,
    fixed fraction of the cost spread, independent of its capacity.

    :param gen_refs: Generator refs (storage refs are excluded from the
        band computation and from the returned ``eps_by_aid``).
    :param statics_by_ref: Cached statics, e.g. from
        :func:`filter_and_cache_statics`/:func:`collect_generator_refs`.
    :param default_epsilon: Floor on the price-response band width.
    :param cost_fraction: Fraction of the cost spread the band should target.
    """
    nonstorage_refs = [ref for ref in gen_refs if ref.element_type != STORAGE]
    eps_by_aid: dict[str, float] = {}
    costs_all: list[float] = []
    cost_range = 0.0
    if len(nonstorage_refs) >= 2:
        costs_all = [
            float(statics_by_ref[r].get("marginal_cost", 0.0)) for r in nonstorage_refs
        ]
        cost_range = max(costs_all) - min(costs_all)
        target_band = max(default_epsilon, cost_fraction * cost_range)
    else:
        target_band = default_epsilon
    for ref in nonstorage_refs:
        p_nom_ref = float(statics_by_ref[ref].get("p_nom", 0.0))
        eps_by_aid[ref.component_id] = target_band / max(p_nom_ref, 1.0)
    return EpsilonScaling(eps_by_aid, target_band, cost_range, costs_all)


@dataclass
class StorageParams:
    """Storage actor parameters derived from PyPSA statics.

    Shared by every scenario, so all of them model the same storage (Q15).
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    #: Floored to at least ``1e-6`` (never exactly 0, to keep it usable as a
    #: divisor in the actor's dynamics).
    eta_charge: float
    eta_discharge: float
    #: Initial state of charge, as a fraction of ``e_max`` in ``[0, 1]``.
    e_initial: float

    @classmethod
    def from_statics(
        cls,
        statics: dict,
        p_nom: float,
        *,
        default_max_hours: float = 100.0,
        default_efficiency: float = 0.95,
    ) -> StorageParams:
        p_min_pu = float(statics.get("p_min_pu", -1.0))
        p_max_pu = float(statics.get("p_max_pu", 1.0))
        p_charge_max = max(0.0, -p_min_pu * p_nom)  # p_min_pu >= 0: cannot charge
        p_discharge_max = max(0.0, p_max_pu * p_nom)

        max_hours = float(statics.get("max_hours", default_max_hours))
        e_max = max(1e-6, p_nom * max_hours)

        eta_charge = float(
            statics.get(
                "efficiency_store", statics.get("efficiency_charge", default_efficiency)
            )
        )
        eta_discharge = float(
            statics.get(
                "efficiency_dispatch",
                statics.get("efficiency_discharge", default_efficiency),
            )
        )

        soc_initial_raw = statics.get("state_of_charge_initial", np.nan)
        if np.isfinite(soc_initial_raw) and float(soc_initial_raw) > 1e-9:
            soc_initial_abs = float(soc_initial_raw)
        else:
            # PyPSA default is often 0.0; treat that as "unspecified" here so
            # storage can participate in the benchmark without extra config.
            soc_initial_abs = 0.5 * e_max
        e_initial = float(np.clip(soc_initial_abs / e_max, 0.0, 1.0))

        return cls(
            e_max=e_max,
            p_charge_max=p_charge_max,
            p_discharge_max=p_discharge_max,
            eta_charge=max(1e-6, eta_charge),
            eta_discharge=max(1e-6, eta_discharge),
            e_initial=e_initial,
        )


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


def _lookup_ts(scenario: ScenarioData, ref: Any) -> Any | None:
    """Return the timeseries for *ref*, or ``None`` if it has none."""
    return scenario.timeseries.get(ref)


def compute_overall_cost(
    cost_by_aid: dict[str, float],
    t_P: Sequence[float],
    Y_P: np.ndarray,
    labels_P: Sequence[str],
    cost_quad_by_aid: dict[str, float] | None = None,
) -> tuple[float, pd.Series]:
    """Total cost of an hourly dispatch: Σ_t Σ_i cost_by_aid[aid_i] · P_i(t).

    *t_P* / *Y_P* hold one row per hour (see
    :meth:`PowerLoadAggregator.hourly_dispatch`). Negative power (storage
    charging) is clipped to zero before weighting: charging is not credited
    as negative cost, and storage pays its cost on discharge only (Q15).

    Returns ``(total, per_step)`` where *per_step* is a ``"cost:total"``
    series indexed by *t_P*, merged into the CSV by
    :func:`write_scenario_outputs`.
    """
    t_arr = np.asarray(t_P, dtype=float)
    cost_vec = np.array([cost_by_aid.get(aid, 0.0) for aid in labels_P])
    dispatched = np.maximum(Y_P, 0.0)
    step_cost = (dispatched * cost_vec).sum(axis=1)
    if cost_quad_by_aid:
        # --cost-model quadratic: + Σ c₂ · P² per step (Q17).
        quad_vec = np.array([cost_quad_by_aid.get(aid, 0.0) for aid in labels_P])
        step_cost = step_cost + (dispatched**2 * quad_vec).sum(axis=1)
    per_step = pd.Series(step_cost, index=t_arr, name="cost:total")
    # NB: energy cost only (Σ marginal cost · MW, + c₂·P²) — no startup /
    # no-load / commitment cost.
    return float(per_step.sum()), per_step


def compute_slack(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a balance residual (demand − Σ dispatch, MW) into slack flows.

    Returns ``(import, export)``, both non-negative: a positive residual is a
    shortfall the slack supplies, a negative one a surplus it absorbs.
    """
    r = np.asarray(residual, dtype=float)
    return np.maximum(r, 0.0), np.maximum(-r, 0.0)


def _slack_columns(
    aggregator: PowerLoadAggregator,
    t_index: pd.Index,
    slack: SlackSpec,
    step_s: float = 3600.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slack import/export and their cost on the hourly rows *t_index*.

    From the aggregator's applied schedules (storage charging included).
    """
    by_bucket = aggregator.slack_by_bucket(step_s)
    rows = [by_bucket.get(int(t // step_s), (0.0, 0.0)) for t in t_index]
    imp = np.array([r[0] for r in rows], dtype=float)
    exp = np.array([r[1] for r in rows], dtype=float)
    frame = pd.DataFrame({"slack:import": imp, "slack:export": exp}, index=t_index)
    cost = pd.Series(
        imp * slack.import_price + exp * slack.export_price,
        index=t_index,
        name="cost:total",
    )
    return frame, cost


# ---------------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------------


def build_group_map(
    refs: Sequence[ComponentRef],
    *,
    statics_by_ref: dict[ComponentRef, dict] | None = None,
    behavior: Any | None = None,
) -> dict[str, str]:
    """Map ``component_id -> carrier`` for the stacked / per-unit plots.

    The carrier comes from the cached statics row when *statics_by_ref* is
    given, else from ``behavior.get_statics(ref)``.  Components without a
    carrier fall back to their ``element_type`` (``"storage"`` etc.), then to
    ``"other"``.  ``LOAD`` refs are skipped (loads never enter the ``"P"``
    recording).  Keys are ``str(component_id)`` so they match the agent AIDs
    used as plot labels.
    """
    out: dict[str, str] = {}
    for ref in refs:
        if getattr(ref, "element_type", None) == LOAD:
            continue
        statics: dict | None = None
        if statics_by_ref is not None:
            statics = statics_by_ref.get(ref)
        if statics is None and behavior is not None:
            statics = behavior.get_statics(ref)
        carrier = (statics or {}).get("carrier")
        if carrier is not None and not pd.isna(carrier) and str(carrier).strip():
            out[str(ref.component_id)] = str(carrier)
        else:
            out[str(ref.component_id)] = str(
                getattr(ref, "element_type", None) or "other"
            )
    return out


def carrier_style(net: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``({carrier: colour}, {carrier: nice_name})`` from ``net.carriers``.

    Both dicts are empty when the network carries no carrier metadata (e.g. the
    toy network).
    """
    colors: dict[str, str] = {}
    names: dict[str, str] = {}
    carriers = getattr(net, "carriers", None)
    if carriers is None or len(carriers) == 0:
        return colors, names
    for carrier, row in carriers.iterrows():
        col = row.get("color") if hasattr(row, "get") else None
        if isinstance(col, str) and col.strip():
            colors[str(carrier)] = col
        nice = row.get("nice_name") if hasattr(row, "get") else None
        if isinstance(nice, str) and nice.strip():
            names[str(carrier)] = nice
    return colors, names


def _non_negative_int(text: str) -> int:
    """argparse ``type=`` for an integer ``>= 0``."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _bounded_float(lo: float, hi: float) -> Callable[[str], float]:
    """argparse ``type=`` that parses a float and rejects values outside ``[lo, hi]``."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"must be in [{lo:g}, {hi:g}], got {value:g}"
            )
        return value

    return parse


def loss_rate_from_args(args: argparse.Namespace) -> float:
    """The ``--loss-rate`` fraction (``0`` when not given)."""
    return 0.0 if args.loss_rate is None else args.loss_rate


def build_scenario_argparser(
    description: str | None,
    *,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    with_balance_tol: bool = False,
) -> argparse.ArgumentParser:
    """Build the CLI parser shared by all scenario entry points.

    *extra_args*, if given, is called with the parser to add algorithm-specific
    flags (e.g. DEED-ADMM's ``--gamma``/``--max-iter``).

    :param with_balance_tol: If true, add ``--balance-tol`` (the power-balance
        tolerance passed to :class:`PowerLoadAggregator`) and ``--strict``
        (fail the run instead of writing outputs when that check fails).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--network",
        type=str,
        default="toy",
        help=(
            "Network source. Either 'toy' (the built-in 5-bus fixture), "
            "a PyPSA example name "
            f"({', '.join(available_examples())}), or a path to a "
            ".nc/.h5/.xlsx file or CSV folder."
        ),
    )
    parser.add_argument("--delay-s", type=float, default=0.02, help="Comms delay (s).")
    parser.add_argument(
        "--loss-rate",
        type=_bounded_float(0.0, 1.0),
        default=None,
        help=(
            "Probability that a message is lost, as a fraction in [0, 1] "
            "(e.g. 0.05 for 5%%; default 0)."
        ),
    )
    parser.add_argument(
        "--tuned-params",
        type=Path,
        default=None,
        help=(
            "TOML file with the tuned algorithm parameters (iteration budgets, rho, "
            "alpha, ...) per scenario, topology and network; default: the packaged "
            "tuned_parameters.toml. Explicit flags override it."
        ),
    )
    parser.add_argument(
        "--topology",
        choices=TOPOLOGIES,
        default="complete",
        help=(
            "Communication graph: 'complete' (default, everyone talks to everyone) "
            "or 'ring' (algorithm agents in a cycle ordered by id). Loads attach "
            "via their bus; every message travels the shortest path (hops x "
            "--delay-s, loss per hop)."
        ),
    )
    parser.add_argument(
        "--cost-model",
        choices=COST_MODELS,
        default="linear",
        help=(
            "Generator cost: 'linear' (PyPSA marginal cost, default) or 'quadratic' "
            "(+ c2·P² with c2 = 0.11·marginal_cost/p_nom, decision Q17)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the message-loss draws (default 0), so lossy runs are reproducible.",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=None,
        help=(
            "Seconds (simulation time) a round waits for missing messages before "
            "advancing with the last received values (default: 3 x --delay-s under "
            "loss; none without loss, so lossless rounds wait for every neighbour as "
            "in the papers)."
        ),
    )
    parser.add_argument(
        "--result-poll-s",
        type=float,
        default=300.0,
        help=(
            "Interval (simulation seconds) between the leader's requests for "
            "results that never arrived (default 300)."
        ),
    )
    parser.add_argument(
        "--result-retries",
        type=int,
        default=12,
        help=(
            "Requests for a missing result before the run continues without it "
            "and is marked degraded (default 12)."
        ),
    )
    parser.add_argument(
        "--load-timeout",
        type=_bounded_float(0.0, float("inf")),
        default=None,
        help=(
            "Seconds (simulation time) after an hour starts before it is applied "
            "with incomplete load reports (missing loads count with their forecast; "
            "default 3 x --delay-s)."
        ),
    )
    parser.add_argument(
        "--max-missing-rounds",
        type=_non_negative_int,
        default=20,
        help=(
            "ADMM only: consecutive rounds a participant may miss before the "
            "coordinator stops the run as not converged (default 20; 0 disables)."
        ),
    )
    parser.add_argument("--name-base", type=str, default=default_name_base)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for the CSV/PDF outputs (created if missing).",
    )
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--slack",
        action="store_true",
        help=(
            "Add a slack node (the paper's point of common coupling) that "
            "imports any shortfall and exports any surplus, priced into "
            "cost:total and written as slack:import/slack:export columns. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--slack-import-price",
        type=float,
        default=SlackSpec().import_price,
        help="Price per MWh the slack supplies, €/MWh (default: %(default)s).",
    )
    parser.add_argument(
        "--slack-export-price",
        type=float,
        default=SlackSpec().export_price,
        help="Price per MWh the slack absorbs, €/MWh (default: %(default)s).",
    )
    if with_balance_tol:
        # Dests "balance_tol"/"strict" — keep in sync with the matching
        # names hardcoded into run_scenario_main's `with_balance_tol` branch
        # below, which forwards them to execute_test_case.
        parser.add_argument(
            "--balance-tol",
            type=float,
            default=0.01,
            help=(
                "Max allowed per-timestep |generation - demand| / demand "
                "before a finished run is flagged as not converged (default: "
                "0.01, i.e. 1%%)."
            ),
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help=(
                "Exit with an error instead of writing CSV/PDF outputs if "
                "the power-balance check fails (default: log a warning and "
                "write the outputs anyway)."
            ),
        )
    if extra_args is not None:
        extra_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PowerLoadInfo:
    """Current load power forwarded from a load agent to the aggregator."""

    power_load: float
    time: float


@dataclass
class OptimizationFinishedInfo:
    """A generator's finished schedule, sent to the leader.

    Sent when the generator's run is done, and again as the reply to a
    :class:`ScheduleRequest` (so a lost one can be recovered).
    """

    aid: str
    schedule: np.ndarray | None = None
    #: Rounds the unit's algorithm ran and how it stopped (for the meta sidecar).
    iterations: int | None = None
    converged: bool | None = None


class ScheduleRequest(RoleLocalMessage):
    """Leader → generator: send your finished schedule (answered by
    :class:`GeneratorResultRole`, never seen by the algorithm)."""


@dataclass
class KickOff(RoleLocalMessage):
    """Leader → the trigger generator: start the distributed run with
    *start_message* (acknowledged, so a lost kick-off is retried). *run*
    numbers the runs, so retries of one run start it once while a later run
    (e.g. FDGDM's outer storage rounds) starts again."""

    start_message: Any
    run: int = 1


@dataclass
class KickOffAck:
    """Reply to :class:`KickOff`."""

    aid: str


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class PowerLoadMonitoring(Role):
    """Observes ``max_active_power`` and forwards it to *target* each timestep."""

    def __init__(self, behavior: Any, target: Any) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        power = self._behavior.observe(self.context.aid, "max_active_power")
        self.context.schedule_instant_message(
            PowerLoadInfo(power_load=float(power), time=self.context.current_timestamp),
            self._target,
        )


class GeneratorResultRole(Role):
    """Generator-side companion of the :class:`DistributedOptimizationRole`.

    Remembers the finished schedule (set by :func:`make_finish_callback`) and
    answers the leader's :class:`ScheduleRequest` with it, so a lost
    "finished" message can be recovered.  On :class:`KickOff` it starts the
    local algorithm once and acknowledges every request, so a lost kick-off is
    retried without ever restarting a run.
    """

    def __init__(self) -> None:
        super().__init__()
        #: The finished schedule, ``None`` while the run is in progress.
        self.schedule: np.ndarray | None = None
        self.iterations: int | None = None
        self.converged: bool | None = None
        self._started_run = 0

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._on_schedule_request, lambda c, m: isinstance(c, ScheduleRequest)
        )
        self.context.subscribe_message(
            self, self._on_kick_off, lambda c, m: isinstance(c, KickOff)
        )

    def _reply(self, content: Any, meta: dict) -> None:
        self.context.schedule_instant_task(self.context.reply_to(content, meta))

    def _on_schedule_request(self, content: ScheduleRequest, meta: dict) -> None:
        if self.schedule is None:
            return  # still running: the leader asks again later
        self._reply(
            OptimizationFinishedInfo(
                aid=self.context.aid,
                schedule=self.schedule,
                iterations=self.iterations,
                converged=self.converged,
            ),
            meta,
        )

    def _on_kick_off(self, content: KickOff, meta: dict) -> None:
        if content.run > self._started_run:
            self._started_run = content.run
            self.schedule = None  # a poll must not return the previous run's result
            self.context.get_role(DistributedOptimizationRole).start(
                content.start_message
            )
        self._reply(KickOffAck(aid=self.context.aid), meta)


def generator_agent(opt_role: DistributedOptimizationRole) -> Any:
    """A generator agent: the optimization role plus its :class:`GeneratorResultRole`."""
    from mango import agent_composed_of

    return agent_composed_of(opt_role, GeneratorResultRole())


class PowerLoadAggregator(Role):
    """Leader role: fires a distributed optimization run once, then dispatches
    the resulting schedule.

    Load agents stream ``PowerLoadInfo`` messages each hour. Once all loads
    report for a given timestep, the aggregator applies the pre-computed
    set-points for that hour. Timesteps that arrive before the optimization
    run is done are held in ``_pending`` and flushed immediately afterwards.

    Message loss (all optional; without loss none of it changes a run):

    * *load_timeout*: every hour is applied at the latest *load_timeout*
      seconds after it starts, even if load reports are missing (counted in
      :attr:`hours_with_missing_loads`).
    * *trigger_addr_is_generator*: the kick-off is a :class:`KickOff` request
      to the trigger generator's :class:`GeneratorResultRole`, retried (mango
      ``gather``) until acknowledged; *start_locally* instead starts a run on
      this agent without any message (e.g. an ADMM coordinator role).
    * *generator_addrs* + *result_poll_s*: every *result_poll_s* seconds the
      leader requests the schedules still missing with ``gather``; after *result_retries*
      polls a missing generator gets an all-zero schedule and the run is
      marked degraded (:attr:`degraded_generators`), which the balance check
      then reports.
    """

    def __init__(
        self,
        *,
        behavior: Any,
        number_loads: int,
        generator_aids: list[str],
        trigger: Any,
        time_to_index: dict[float, int],
        schedule_by_aid: dict[str, np.ndarray],
        finished_message_type: type,
        build_start_message: Callable[[], Any],
        n_finished_required: int | None = None,
        demand_target: np.ndarray | None = None,
        balance_label: str = "",
        balance_tol: float = 0.01,
        slack: SlackSpec | None = None,
        start_locally: Callable[[Any], None] | None = None,
        generator_addrs: dict[str, Any] | None = None,
        load_timeout: float | None = None,
        request_timeout: float = 1.0,
        result_poll_s: float | None = None,
        result_retries: int = 12,
        kick_off_retries: int = 20,
        on_run_finished: Callable[[], Any | None] | None = None,
    ) -> None:
        super().__init__()
        self._slack = slack
        # Called when every required unit reported. Returning a start message
        # launches another distributed run (e.g. FDGDM's outer storage rounds);
        # returning None finishes. Finished messages carry no run id, so under
        # heavy loss a late result poll could deliver a previous run's schedule.
        self._on_run_finished = on_run_finished
        #: Distributed runs started (1 unless *on_run_finished* asks for more).
        self.runs: int = 0
        #: Per unit: rounds run and stop flag from the finished messages.
        self.iterations_by_aid: dict[str, int | None] = {}
        self.converged_by_aid: dict[str, bool | None] = {}
        #: Coordinator-level statistics (e.g. ADMM iterations / missed answers).
        self.algorithm_stats: dict[str, Any] = {}
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = trigger
        self._time_to_index = time_to_index
        self._schedule_by_aid = schedule_by_aid
        self._finished_message_type = finished_message_type
        self._build_start_message = build_start_message
        self._n_finished_required = (
            n_finished_required
            if n_finished_required is not None
            else len(generator_aids)
        )
        self._demand_target = (
            None if demand_target is None else np.asarray(demand_target, dtype=float)
        )
        self._balance_label = balance_label
        self._balance_tol = balance_tol
        self._start_locally = start_locally
        # Kept by reference: a scenario may fill it after registering the units.
        self._generator_addrs = generator_addrs if generator_addrs is not None else {}
        self._load_timeout = load_timeout
        self._request_timeout = request_timeout
        self._result_poll_s = result_poll_s
        self._result_retries = result_retries
        self._kick_off_retries = kick_off_retries

        self._ready: bool = False
        self._finished_aids: set[str] = set()
        self._pending: set[int] = set()

        #: Per hour (simulation seconds): the loads that reported it.
        self.demand_map: dict[Any, set[Any]] = {}
        self._closed_times: set[Any] = set()
        #: Largest per-timestep |Σ schedules − demand| / demand after the
        #: optimization finished; ``None`` until verified (or if no
        #: ``demand_target`` was supplied).
        self.balance_max_rel_gap: float | None = None
        #: Generators that never delivered a schedule and got the zero default.
        self.degraded_generators: list[str] = []
        #: Hours applied at their deadline with at least one load missing.
        self.hours_with_missing_loads: int = 0

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._handle_load_info, lambda c, m: isinstance(c, PowerLoadInfo)
        )
        self.context.subscribe_message(
            self,
            self._handle_finished,
            lambda c, m: isinstance(c, self._finished_message_type),
        )

    def on_ready(self) -> None:
        self._arm_hour_deadlines()
        if self._trigger is None or self._n_finished_required == 0:
            # All schedules are pre-filled; nothing to run. Still verify the
            # balance here — schedule_by_aid was already populated by the
            # scenario's own pre-scheduling before the world started, so the
            # check is meaningful even though no distributed round ever ran.
            # Skipping this left balance_max_rel_gap permanently None on this
            # path (e.g. FDGDM with 0-1 thermal generators), which silently
            # defeated the --strict CLI flag.
            self._ready = True
            self._verify_balance()
            return
        self._start_run(self._build_start_message())

    def _start_run(self, start_message: Any) -> None:
        self.runs += 1
        if self._start_locally is not None:
            self._start_locally(start_message)
        elif self._generator_addrs:
            self.context.schedule_instant_task(self._kick_off(start_message))
        else:
            self.context.schedule_instant_message(start_message, self._trigger)
        self._schedule_result_poll(1)

    def _arm_hour_deadlines(self) -> None:
        """Apply every hour at its deadline, whether or not any report arrived."""
        if self._load_timeout is None:
            return
        for time in self._time_to_index:
            self.context.schedule_timestamp_task(
                self._close_hour_at_deadline(time), float(time) + self._load_timeout
            )

    # -- kick-off and result collection ---------------------------------------

    async def _kick_off(self, start_message: Any) -> None:
        """Send :class:`KickOff` until acknowledged.

        The first run starts at the trigger generator (the others join on its
        round-0 messages). A later run kicks off every participant: after a
        finished run, participants ignore non-initial messages.
        """
        request = KickOff(start_message=start_message, run=self.runs)
        pending = (
            [self._trigger] if self.runs == 1 else list(self._generator_addrs.values())
        )
        for _ in range(self._kick_off_retries):
            replies = await self.context.gather(
                request,
                pending,
                timeout=self._request_timeout,
                reply_type=KickOffAck,
            )
            acked = {reply.aid for reply in replies.values()}
            pending = [addr for addr in pending if addr.aid not in acked]
            if not pending:
                return
        logger.error(
            "%s: the kick-off was never acknowledged after %d attempts; the "
            "distributed run did not start.",
            self._balance_label or type(self).__name__,
            self._kick_off_retries,
        )

    def _schedule_result_poll(self, attempt: int) -> None:
        if self._result_poll_s is None or not self._generator_addrs:
            return
        self.context.schedule_timestamp_task(
            self._poll_results(attempt, self.runs),
            self.context.current_timestamp + self._result_poll_s,
        )

    async def _poll_results(self, attempt: int, run: int) -> None:
        """Ask the generators whose schedule is still missing for it.

        A chain of polls belongs to one *run*; it stops once a later run
        (FDGDM's outer storage rounds) has started its own chain.
        """
        if self._ready or run != self.runs:
            return
        missing = [
            aid for aid in self._generator_addrs if aid not in self._finished_aids
        ]
        if missing:
            replies = await self.context.gather(
                self._result_request(),
                [self._generator_addrs[aid] for aid in missing],
                timeout=self._request_timeout,
                reply_type=self._finished_message_type,
            )
            for reply in replies.values():
                self._handle_finished(reply, {})
        if self._ready or run != self.runs:
            return
        if attempt < self._result_retries:
            self._schedule_result_poll(attempt + 1)
            return
        still_missing = [
            aid for aid in self._generator_addrs if aid not in self._finished_aids
        ]
        self.degraded_generators = still_missing
        logger.error(
            "%s: no answer from %d unit(s) after %d polls (%s); continuing "
            "without them. Treat this run as degraded.",
            self._balance_label or type(self).__name__,
            len(still_missing),
            attempt,
            ", ".join(still_missing),
        )
        self._give_up(still_missing)

    def _result_request(self) -> Any:
        """The request sent to units whose result is still missing."""
        return ScheduleRequest()

    def _give_up(self, missing: list[str]) -> None:
        """Units in *missing* never answered: give them all-zero schedules."""
        horizon = len(self._demand_target) if self._demand_target is not None else 0
        for aid in missing:
            self._handle_finished(
                OptimizationFinishedInfo(aid=aid, schedule=np.zeros(horizon)), {}
            )

    def accept_schedules(self, schedules: dict[str, np.ndarray]) -> None:
        """Take schedules that are already on this agent (e.g. from a local
        coordinator) as if every generator had reported them."""
        for aid, schedule in schedules.items():
            self._handle_finished(
                OptimizationFinishedInfo(aid=aid, schedule=schedule), {}
            )

    # -- applying the schedule -------------------------------------------------

    def _apply_schedule_index(self, idx: int) -> None:
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            p_schedule = np.asarray(schedule).ravel()
            if idx < p_schedule.size:
                self._behavior.act(aid, "regulate", float(p_schedule[idx]))

    def _handle_finished(self, message: Any, meta: dict) -> None:
        if message.aid in self._finished_aids:
            return  # duplicate (e.g. the reply to a poll after all)
        schedule = getattr(message, "schedule", None)
        if schedule is not None:
            self._schedule_by_aid[message.aid] = np.asarray(
                schedule, dtype=float
            ).copy()
        self._finished_aids.add(message.aid)
        self.iterations_by_aid[message.aid] = getattr(message, "iterations", None)
        self.converged_by_aid[message.aid] = getattr(message, "converged", None)
        if len(self._finished_aids) == self._n_finished_required:
            if self._on_run_finished is not None:
                next_start = self._on_run_finished()
                if next_start is not None:
                    self._finished_aids = set()
                    self._start_run(next_start)
                    return
            self._ready = True
            self._verify_balance()
            for idx in sorted(self._pending):
                self._apply_schedule_index(idx)
            self._pending.clear()

    @property
    def ready(self) -> bool:
        """Whether every required generator reported and the schedule is being applied."""
        return self._ready

    def missing_generators(self) -> list[str]:
        """Generators whose schedule never arrived; empty once the run is ready."""
        if self._ready:
            return []
        return [
            aid
            for aid in self._generator_aids
            if aid not in self._finished_aids and aid not in self._schedule_by_aid
        ]

    def _verify_balance(self) -> None:
        """Check the finished schedules against the known demand target.

        Ces et al. 2025 verify power balance before accepting a distributed
        solution; without this an unconverged run silently dispatches an
        infeasible schedule whose "cost" is meaningless (unserved demand looks
        cheap, over-generation looks expensive).
        """
        gap = self.schedule_residual()
        if gap is None:
            return
        rel = np.abs(gap) / np.maximum(np.abs(self._demand_target), 1e-9)
        self.balance_max_rel_gap = float(rel.max()) if rel.size else 0.0
        label = self._balance_label or type(self).__name__
        if self._slack is not None:
            imp, exp = compute_slack(gap)
            demand_total = max(float(np.sum(self._demand_target)), 1e-9)
            logger.info(
                "%s: slack closes the balance — import %.1f MWh (%.2f%% of "
                "demand), export %.1f MWh (%.2f%%).",
                label,
                float(imp.sum()),
                100.0 * float(imp.sum()) / demand_total,
                float(exp.sum()),
                100.0 * float(exp.sum()) / demand_total,
            )
        elif self.balance_max_rel_gap > self._balance_tol:
            logger.error(
                "%s: schedules violate power balance — max per-timestep "
                "|generation − demand| is %.1f%% of demand (tolerance %.1f%%). "
                "The optimization did not converge; treat this run's cost as "
                "invalid.",
                label,
                100.0 * self.balance_max_rel_gap,
                100.0 * self._balance_tol,
            )
        else:
            logger.info(
                "%s: power balance verified — max per-timestep gap %.3f%% of demand.",
                label,
                100.0 * self.balance_max_rel_gap,
            )

    def schedule_residual(self) -> np.ndarray | None:
        """Per-timestep ``demand − Σ schedules`` (MW) of the current schedules.

        ``None`` without a demand target. Missing or short schedules count as
        zero output.
        """
        if self._demand_target is None:
            return None
        m = len(self._demand_target)
        total = np.zeros(m, dtype=float)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            arr = np.asarray(schedule, dtype=float).ravel()
            n = min(m, arr.size)
            total[:n] += arr[:n]
        return self._demand_target - total

    def hourly_dispatch(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """``(seconds, P, aids, demand)`` per hour from the final schedules.

        Not from the simulation's recordings: an hour that starts before the
        distributed run finishes is applied late, in the same instant as the
        others, so a recording never shows its set-point. Missing or short
        schedules count as zero output.
        """
        times = sorted(self._time_to_index, key=self._time_to_index.get)
        idx = np.array([self._time_to_index[t] for t in times], dtype=int)
        aids = list(self._generator_aids)
        P = np.zeros((idx.size, len(aids)))
        for j, aid in enumerate(aids):
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            arr = np.asarray(schedule, dtype=float).ravel()
            ok = idx < arr.size
            P[ok, j] = arr[idx[ok]]
        demand = (
            np.zeros(idx.size)
            if self._demand_target is None
            else self._demand_target[idx]
        )
        return np.asarray(times, dtype=float), P, aids, demand

    def slack_by_bucket(self, step_s: float = 3600.0) -> dict[int, tuple[float, float]]:
        """Slack ``(import, export)`` MW per simulation-time bucket.

        Keyed like the hourly CSV rows (``int(sim_seconds // step_s)``).
        """
        gap = self.schedule_residual()
        if gap is None:
            return {}
        imp, exp = compute_slack(gap)
        return {
            int(seconds // step_s): (float(imp[idx]), float(exp[idx]))
            for seconds, idx in self._time_to_index.items()
            if idx < gap.size
        }

    # -- load reports -----------------------------------------------------------

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        sender = sender_addr(meta).aid if meta else None
        key = sender if sender is not None else object()  # unknown sender: own slot
        time = round(float(message.time), 6)
        if time in self._closed_times:
            return  # hour already applied (at its deadline)
        reported = self.demand_map.setdefault(time, set())
        reported.add(key)
        if len(reported) != self.number_loads:
            return  # wait until all loads report for this timestep (or the deadline)
        self._close_hour(time)

    async def _close_hour_at_deadline(self, time: Any) -> None:
        if time in self._closed_times:
            return
        if len(self.demand_map.get(time, ())) >= self.number_loads:
            return
        self.hours_with_missing_loads += 1
        self._close_hour(time)

    def _close_hour(self, time: Any) -> None:
        self._closed_times.add(time)
        idx = self._time_to_index.get(time)
        if idx is None:
            return
        if self._ready:
            self._apply_schedule_index(idx)
        else:
            self._pending.add(idx)  # applied once the schedule is available


def make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    finished_message_type: type,
    algorithm_label: str,
    schedule_attr: str = "actor.P",
) -> Any:
    """Return the ``(algorithm, carrier) -> None`` hook shared by the
    carrier-based algorithms (consensus, diffusion, FDGDM, DEED-ADMM).

    *schedule_attr* is a dotted attribute path read off *algorithm* to obtain
    the finished per-timestep schedule (e.g. ``"actor.P"`` or ``"P"``).  The
    schedule travels to the leader in the finished message and stays on the
    agent's :class:`GeneratorResultRole` for later requests.
    """

    def handle_finished(algorithm: Any, carrier: Any) -> None:
        # `carrier._parent` is a distributed_resource_optimization-internal
        # attribute (the mango Role wrapping this carrier); there is no
        # public accessor for it today.
        role = carrier._parent
        aid = role.context.aid

        obj = algorithm
        try:
            for part in schedule_attr.split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(
                f"{algorithm_label}: finished algorithm for {aid!r} has no "
                f"attribute path {schedule_attr!r} ({exc}); the run may have "
                "aborted before producing a schedule."
            ) from exc
        schedule = np.asarray(obj, dtype=float).copy()
        converged = getattr(algorithm, "converged", None)
        iterations = getattr(algorithm, "iterations", None)
        result_role = role.context.get_role(GeneratorResultRole)
        if result_role is not None:
            result_role.schedule = schedule
            result_role.iterations = iterations
            result_role.converged = converged
        if converged is False:
            logger.warning(
                "%s did not converge for %s after %s iteration(s); schedule may "
                "be suboptimal.",
                algorithm_label,
                aid,
                iterations if iterations is not None else "an unknown number of",
            )

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            role.context.schedule_instant_message(
                finished_message_type(
                    aid=aid,
                    schedule=schedule,
                    iterations=iterations,
                    converged=converged,
                ),
                leader_addr,
            )
        logger.info(
            "%s finished for %s (schedule len=%s%s)",
            algorithm_label,
            aid,
            schedule.size,
            f", iterations={iterations}" if iterations is not None else "",
        )

    return handle_finished


def _run_meta(world: Any, aggregator: Any, total_cost: float) -> dict[str, Any]:
    """Outcome part of ``<name>-meta.json`` (the CLI adds dimensions, git, wall time)."""
    meta: dict[str, Any] = {
        "total_cost": total_cost,
        "messages_delivered": getattr(world, "messages_delivered_count", None),
        "simulated_seconds": float(
            getattr(getattr(world, "clock", None), "time", 0.0) or 0.0
        ),
    }
    if aggregator is None:
        return meta
    iterations = [i for i in aggregator.iterations_by_aid.values() if i is not None]
    converged = [c for c in aggregator.converged_by_aid.values() if c is not None]
    stats = dict(getattr(aggregator, "algorithm_stats", {}) or {})
    if not iterations and stats.get("iterations") is not None:
        iterations = [stats["iterations"]]
    if not converged and stats.get("converged") is not None:
        converged = [stats["converged"]]
    if converged:
        stop_reason = "converged" if all(converged) else "max_iter"
    else:
        stop_reason = "fixed_budget" if iterations else "n/a"
    for key in ("solver_status", "objective"):
        value = getattr(aggregator, key, None)
        if value is not None:
            stats[key] = value
    meta.update(
        {
            "balance_max_rel_gap": aggregator.balance_max_rel_gap,
            "iterations_max": max(iterations) if iterations else None,
            "iterations_min": min(iterations) if iterations else None,
            "iterations_mean": float(np.mean(iterations)) if iterations else None,
            "stop_reason": stop_reason,
            "outer_runs": getattr(aggregator, "runs", None),
            "degraded_generators": list(aggregator.degraded_generators),
            "hours_with_missing_loads": aggregator.hours_with_missing_loads,
            "algorithm_stats": stats,
        }
    )
    return meta


def file_sha256(path: Any) -> str | None:
    """SHA-256 of the file *path* (``None`` for built-in networks such as ``toy``)."""
    import hashlib

    p = Path(str(path))
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(module: Any) -> dict[str, Any]:
    """HEAD SHA and dirty flag of the git checkout containing *module* (None if none)."""
    try:
        here = str(Path(module.__file__).resolve().parent)
        sha = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError, AttributeError, TypeError):
        return {"sha": None, "dirty": None}


def write_scenario_outputs(
    world: Any,
    *,
    name_base: str,
    cost_by_aid: dict[str, float],
    stacked_title: str = "Stacked power",
    cost_title: str = "Cost per timestep",
    aggregator: PowerLoadAggregator,
    balance_tol: float | None = None,
    strict: bool = False,
    group_map: dict[str, str] | None = None,
    net: Any | None = None,
    label: str = "",
    slack: SlackSpec | None = None,
    cost_quad_by_aid: dict[str, float] | None = None,
) -> float:
    """Write the standard scenario result files and return the total cost.

    Shared by every scenario: take one row per hour from the schedules
    *aggregator* applied (:meth:`PowerLoadAggregator.hourly_dispatch`),
    compute the overall cost, then emit
    ``<name_base>-{df.csv,stacked.pdf,observation.pdf,balance.pdf,cost.pdf}``:

    * ``df.csv`` — ``target:``/``P:`` columns plus cost (and slack), one row
      per hourly snapshot (consumed by ``compare_costs.py``).
    * ``stacked.pdf`` — generation stacked by carrier, with the demand target.
    * ``observation.pdf`` — per-unit power as a grid of per-carrier subplots.
    * ``balance.pdf`` — total generation vs demand + residual.
    * ``cost.pdf`` — dispatch cost per timestep.

    :param aggregator: The scenario's leader role; its schedules are the
        output. With *strict*, raise instead of writing outputs if
        ``aggregator.balance_max_rel_gap`` exceeds *balance_tol* -- or was
        never computed at all (the schedule never finished), which is just
        as much a failure as an out-of-tolerance gap.
    :param balance_tol: Tolerance to compare *aggregator*'s gap against;
        required (and otherwise ignored) with *strict*.
    :param group_map: ``component_id -> carrier`` (see
        :func:`build_group_map`) used to aggregate the plots into carrier
        bands. Without it every component is its own group.
    :param net: PyPSA network whose ``carriers`` table supplies band colours
        and display names (see :func:`carrier_style`); optional.
    :param label: Appended to the per-unit and balance plot titles.
    :param slack: If given, a slack node closes the balance: per hour it imports
        ``max(0, demand − Σ schedules)`` and exports ``max(0, Σ schedules −
        demand)`` (from *aggregator*'s applied schedules), adds both at their
        prices to ``cost:total`` and writes ``slack:import``/``slack:export``
        (MW) to the CSV. Cannot be combined with *strict*.
    :raises RuntimeError: If *strict* is set and the balance check failed or
        never ran.
    :raises ValueError: If *slack* is given with *strict*.
    :returns: The total cost (same value logged and used in the annotation).
    """
    if slack is not None and strict:
        raise ValueError("slack cannot be combined with strict.")
    if not aggregator.ready:
        # Without this a run whose optimization never finished (e.g. a
        # participant's last messages were lost) wrote outputs silently: no
        # schedule was ever applied and the target column stayed at 0.
        missing = aggregator.missing_generators()
        logger.error(
            "%s: the distributed schedule never finished, so no set-points were "
            "applied; %d generator(s) never reported: %s. Treat this run's "
            "outputs as invalid.",
            name_base,
            len(missing),
            ", ".join(missing) or "(unknown)",
        )
    if strict:
        gap = aggregator.balance_max_rel_gap
        if gap is None:
            raise RuntimeError(
                f"{name_base}: power balance was never verified (the "
                "distributed schedule never finished; missing: "
                f"{', '.join(aggregator.missing_generators()) or 'none'}); "
                "refusing to write outputs because --strict was given."
            )
        if gap > balance_tol:
            raise RuntimeError(
                f"{name_base}: power balance check failed (max relative gap "
                f"{gap:.4f} > tolerance {balance_tol}); refusing to write "
                "outputs because --strict was given."
            )

    t_P, Y_P, labels_P, target_series = aggregator.hourly_dispatch()

    total_cost, cost_series = compute_overall_cost(
        cost_by_aid, t_P, Y_P, labels_P, cost_quad_by_aid=cost_quad_by_aid
    )
    extra: pd.Series | pd.DataFrame = cost_series
    slack_note = ""
    if slack is not None:
        slack_frame, slack_cost = _slack_columns(aggregator, cost_series.index, slack)
        cost_series = cost_series + slack_cost
        total_cost = float(cost_series.sum())
        extra = pd.concat([cost_series, slack_frame], axis=1)
        imp_mwh = float(slack_frame["slack:import"].sum())
        exp_mwh = float(slack_frame["slack:export"].sum())
        logger.info(
            "%s: slack import %.1f MWh (%.2f €), export %.1f MWh (%.2f €)",
            name_base,
            imp_mwh,
            imp_mwh * slack.import_price,
            exp_mwh,
            exp_mwh * slack.export_price,
        )
        slack_note = f"\nSlack: import {imp_mwh:,.0f} MWh, export {exp_mwh:,.0f} MWh"
    logger.info("%s: overall cost = %.2f", name_base, total_cost)
    annotation = f"Total cost: {total_cost:,.2f}{slack_note}"
    suffix = f" – {label}" if label else ""

    df = pd.DataFrame(Y_P, index=t_P, columns=[f"P:{aid}" for aid in labels_P])
    df.insert(0, f"target:{aggregator.context.aid}", target_series)
    df = df.join(extra)
    df.index = (df.index // 3600.0).astype(int)
    df.index.name = "time"
    df.to_csv(f"{name_base}-df.csv")
    with open(f"{name_base}-meta.json", "w", encoding="utf-8") as f:
        json.dump(_run_meta(world, aggregator, total_cost), f, indent=2, default=str)

    if group_map is None:
        group_map = {str(lbl): str(lbl) for lbl in labels_P}
    style_colors, nice_names = carrier_style(net) if net is not None else ({}, {})
    carriers = sorted(set(group_map.values()))
    band_colors = resolve_carrier_colors(carriers, style_colors)
    band_order = order_carriers(carriers)
    hours = np.asarray(t_P, dtype=float) / 3600.0

    stacked_area(
        hours,
        Y_P,
        labels_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title=stacked_title,
        annotation=annotation,
        groups=group_map,
        colors=band_colors,
        order=band_order,
        write_to=f"{name_base}-stacked.pdf",
    )

    per_unit_small_multiples(
        hours,
        Y_P,
        labels_P,
        group_map,
        xlabel="Hour",
        ylabel="P in MW",
        title=f"Per-unit power{suffix}",
        annotation=annotation,
        carrier_names=nice_names,
        write_to=f"{name_base}-observation.pdf",
    )

    generation_vs_demand(
        hours,
        Y_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title=f"Generation vs demand{suffix}",
        annotation=annotation,
        write_to=f"{name_base}-balance.pdf",
    )

    cost_over_time(
        np.asarray(cost_series.index, dtype=float) / 3600.0,
        cost_series.to_numpy(),
        title=cost_title,
        annotation=annotation,
        write_to=f"{name_base}-cost.pdf",
    )

    return total_cost


def run_scenario_main(
    execute_test_case: Callable[..., Any],
    *,
    doc: str | None,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_kwargs: Sequence[str] = (),
    with_balance_tol: bool = False,
    argv: list[str] | None = None,
) -> None:
    """Shared ``main()`` body: parse CLI args, resolve the scenario, run it.

    *extra_kwargs* names CLI args (added via *extra_args*, or ``"balance_tol"``/
    ``"strict"`` when *with_balance_tol* is set) to forward to
    *execute_test_case* verbatim, e.g. ``("gamma", "max_iter")`` for
    DEED-ADMM's ``--gamma``/``--max-iter``.
    """
    parser = build_scenario_argparser(
        doc,
        default_name_base=default_name_base,
        extra_args=extra_args,
        with_balance_tol=with_balance_tol,
    )
    # Tuned parameters: a first parse finds the network/topology, the file's
    # values become the parser defaults, and the second parse lets explicit
    # flags win.
    first = parser.parse_args(argv)
    config = tuned_params.load(first.tuned_params)
    tuned, tuned_from = tuned_params.resolve(
        config, default_name_base, first.network, first.topology
    )
    source = first.tuned_params or tuned_params.DEFAULT_PATH
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(tuned) - set(actions))
    if unknown:
        parser.error(
            f"{source}: unknown parameter(s) {unknown} for {default_name_base}"
        )
    # set_defaults() skips argparse's own checks: apply each flag's type and
    # choices to the file's values here.
    for key, value in list(tuned.items()):
        action = actions[key]
        if action.type is not None:
            try:
                value = action.type(str(value))
            except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
                parser.error(f"{source}: {key} = {tuned[key]!r}: {exc}")
        elif isinstance(action.const, bool) and not isinstance(value, bool):
            parser.error(f"{source}: {key} must be true or false, got {value!r}")
        if action.choices is not None and value not in action.choices:
            parser.error(
                f"{source}: {key} = {value!r} is not one of {list(action.choices)}"
            )
        tuned[key] = value
    parser.set_defaults(**tuned)
    args = parser.parse_args(argv)
    # Flags given on the command line: parse once more with every default
    # suppressed, so only explicit flags appear.
    saved = [(action, action.default) for action in parser._actions]
    saved_parser_defaults = dict(parser._defaults)
    parser._defaults.clear()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    try:
        explicit = set(vars(parser.parse_args(argv)))
    finally:
        for action, default in saved:
            action.default = default
        parser._defaults.update(saved_parser_defaults)
    args.tuned_from = {key: tuned_from[key] for key in tuned if key not in explicit}

    slack = None
    if args.slack:
        if getattr(args, "strict", False):
            parser.error(
                "--strict cannot be combined with --slack: the slack closes "
                "every power-balance gap, so there is nothing left to enforce."
            )
        slack = SlackSpec(
            import_price=args.slack_import_price,
            export_price=args.slack_export_price,
        )

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenario = resolve_scenario(args.network, simulate_days=args.simulate_days)

    # "balance_tol"/"strict" here must match the --balance-tol/--strict
    # `dest`s added in build_scenario_argparser's with_balance_tol branch.
    names = (
        (*extra_kwargs, "balance_tol", "strict") if with_balance_tol else extra_kwargs
    )
    kwargs = {name: getattr(args, name) for name in names}
    # Every output path is derived from name_base, so prefixing it with the
    # output directory routes all CSV/PDF writes there.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=args.delay_s,
            loss_rate=loss_rate_from_args(args),
            seed=args.seed,
            round_timeout=args.round_timeout,
            result_poll_s=args.result_poll_s,
            result_retries=args.result_retries,
            load_timeout=args.load_timeout,
            max_missing_rounds=args.max_missing_rounds or None,
            name_base=str(args.out_dir / args.name_base),
            simulate_days=args.simulate_days,
            cost_model=args.cost_model,
            topology=args.topology,
            slack=slack,
            **kwargs,
        )
    )
    _complete_meta(
        args.out_dir / f"{args.name_base}-meta.json",
        scenario=default_name_base,
        args=args,
        wall_time_s=perf_counter() - started,
    )


def _complete_meta(
    path: Path, *, scenario: str, args: argparse.Namespace, wall_time_s: float
) -> None:
    """Add run dimensions, git SHAs and wall time to a written meta sidecar."""
    if not path.exists():
        return  # the run failed before writing outputs (e.g. --strict)
    import distributed_resource_optimization
    import mango
    import mango_energy_environments

    import energy_scheduling_benchmark

    meta = json.loads(path.read_text(encoding="utf-8"))
    dims = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    dims.update(
        {
            "scenario": scenario,
            "loss_rate": loss_rate_from_args(args),
        }
    )
    meta = {
        "dimensions": dims,
        **meta,
        "wall_time_s": wall_time_s,
        "inputs": {"network_sha256": file_sha256(args.network)},
        "git": {
            m.__name__: _git_state(m)
            for m in (
                energy_scheduling_benchmark,
                distributed_resource_optimization,
                mango_energy_environments,
                mango,
            )
        },
    }
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
