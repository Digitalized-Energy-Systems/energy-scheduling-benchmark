"""Distributed economic-dispatch scenario using sharing ADMM.

Scenario flow
-------------
1. All generator agents (thermal, renewable, storage) and a single leader
   agent are registered in a mango discrete-event simulation world.
2. At simulation start the leader fires one sharing-ADMM optimisation run:
   - Thermal and renewable units are given box-constrained economic-dispatch
     actors and participate directly in merit-order clearing.
   - Storage units are pre-scheduled separately (see below) and participate
     as fixed-schedule actors so their SOC dynamics do not distort the
     clearing prices seen by thermals.
3. Once every generator reports back, the leader holds a full per-timestep
   schedule for every unit.  As simulation time advances and each load
   timestep fires, the leader applies the corresponding set-points.

Storage pre-scheduling (2-pass)
--------------------------------
``LinearCostEconomicDispatchADMMFlexActor`` responds only to the global
consensus vector *z*, not to the per-participant correction *v*.  The
sharing-ADMM coordinator therefore uses a single-iteration shortcut when
generator specs are provided: it analytically computes the merit-order
clearing price via bisection and pre-sets *z = clearing_price / (ρ·N)*, so
every actor receives exactly the right price signal in one shot.

Storage cannot be part of those specs because its effective per-timestep
bounds depend on SOC from previous timesteps — the bisection would assume
static bounds, thermals would under-dispatch by the amount attributed to
storage, and the schedule would have a systematic gap.

Instead, storage is scheduled in two passes:
  Pass 1 – compute clearing prices from thermals/wind only on the original
            demand target.  Derive each storage unit's SOC-aware schedule
            from those prices via the bias-bisection algorithm.
  Pass 2 – adjust the thermal target: ``adjusted = demand − storage_net``.
            Run ADMM with this adjusted target so thermals cover exactly what
            storage does not.  Storage agents carry a ``FixedScheduleActor``
            that replays the pass-1 schedule regardless of the (adjusted)
            price signal, breaking the circular dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from distributed_resource_optimization import (
    create_admm_economic_dispatch_actor,
    create_admm_sharing_data,
    create_admm_storage_actor,
    create_sharing_target_distance_admm_coordinator,
)
from distributed_resource_optimization.algorithm.admm.core import ADMMAnswer, ADMMMessage
from distributed_resource_optimization.algorithm.admm.economic_dispatch import (
    _storage_schedule_from_price,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import (
    ADMMGeneratorSpec,
    _z_from_clearing_prices,
    create_admm_start,
)
from distributed_resource_optimization.carrier.mango import (
    CoordinatorRole,
    DistributedOptimizationRole,
    OptimizationFinishedMessage,
    StartCoordinatedDistributedOptimization,
)
from mango import Role, RoleAgent, agent_composed_of, auto_assign, complete_topology
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until, record_agent_having

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    PowerUpdateInfo,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    stacked_area,
    visualize_results,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PowerLoadInfo:
    """Current load power forwarded from a load agent to the aggregator."""

    power_load: float
    time: float


@dataclass
class ADMMFinishedInfo:
    """Sent by a generator agent once its ADMM actor has produced a schedule."""

    aid: str


# ---------------------------------------------------------------------------
# Fixed-schedule actor
# ---------------------------------------------------------------------------


class FixedScheduleActor:
    """ADMM participant that always returns a pre-computed schedule.

    Storage units are pre-scheduled from initial (undistorted) clearing
    prices.  This actor lets storage participate in the ADMM round –
    so the coordinator receives the expected number of replies – while
    guaranteeing that the schedule applied in simulation is the one
    derived from pass-1 clearing prices, not the (distorted) pass-2 signal.
    """

    def __init__(self, schedule: np.ndarray) -> None:
        self.x = np.asarray(schedule, dtype=float).copy()

    async def on_exchange_message(self, carrier: Any, message_data: Any, meta: Any) -> None:
        if isinstance(message_data, ADMMMessage):
            carrier.reply_to_other(ADMMAnswer(x=self.x), meta)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class ADMMGeneratorRole(DistributedOptimizationRole):
    """Generator participant that notifies the leader when its schedule is ready."""

    def __init__(self, algorithm: Any, finish_callback: Any) -> None:
        super().__init__(algorithm)
        self._finish_callback = finish_callback

    def setup(self) -> None:
        super().setup()
        self.context.subscribe_message(
            self,
            self._handle_optimization_finished,
            lambda c, m: isinstance(c, OptimizationFinishedMessage),
        )

    def _handle_optimization_finished(
        self, message: OptimizationFinishedMessage, meta: dict
    ) -> None:
        self._finish_callback(self.algorithm, self, self.context.aid)


class PowerLoadMonitoring(Role):
    """Reads the current load power and pushes it to the aggregator each timestep."""

    def __init__(self, behavior: PyPSABehavior, target: Any) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        power = self._behavior.observe(self.context.aid, "max_active_power")
        asyncio.create_task(
            self.context.send_message(
                PowerLoadInfo(power_load=float(power), time=self.context.current_timestamp),
                self._target,
            )
        )


class PowerLoadAggregator(Role):
    """Leader role: fires ADMM once at startup, then dispatches the schedule.

    Load agents stream ``PowerLoadInfo`` messages each hour.  Once all loads
    report for a given timestep, the aggregator applies the pre-computed
    set-points for that hour.  Timesteps that arrive before ADMM is done are
    held in ``_pending`` and flushed immediately afterwards.
    """

    def __init__(
        self,
        *,
        behavior: PyPSABehavior,
        number_loads: int,
        generator_aids: list[str],
        admm_trigger: Any,
        time_to_index: dict[float, int],
        target_series: np.ndarray,
        schedule_by_aid: dict[str, np.ndarray],
        generator_specs: list[ADMMGeneratorSpec],
        dispatch_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = admm_trigger
        self._time_to_index = time_to_index
        self._target_series = target_series
        self._schedule_by_aid = schedule_by_aid
        self._generator_specs = generator_specs
        self._dispatch_epsilon = dispatch_epsilon

        self._admm_ready: bool = False
        self._admm_finished_aids: set[str] = set()
        self._pending: dict[int, float] = {}

        self.demand_map: dict[Any, list[float]] = {}
        self.target: float = 0.0  # recorded by the world for plotting

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._handle_load_info, lambda c, m: isinstance(c, PowerLoadInfo)
        )
        self.context.subscribe_message(
            self, self._handle_admm_finished, lambda c, m: isinstance(c, ADMMFinishedInfo)
        )

    def on_ready(self) -> None:
        data = create_admm_sharing_data(
            target=np.asarray(self._target_series, dtype=float),
            priorities=np.ones(len(self._target_series)),
            generators=self._generator_specs,
            epsilon=self._dispatch_epsilon,
        )
        asyncio.create_task(
            self.context.send_message(
                StartCoordinatedDistributedOptimization(input=create_admm_start(data)),
                self._trigger,
            )
        )

    def _apply_schedule_index(self, idx: int, target_total: float) -> None:
        self.target = float(target_total)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            p_schedule = np.asarray(schedule).ravel()
            if idx < p_schedule.size:
                self._behavior.act(aid, "regulate", float(p_schedule[idx]))

    def _handle_admm_finished(self, message: ADMMFinishedInfo, meta: dict) -> None:
        self._admm_finished_aids.add(message.aid)
        if len(self._admm_finished_aids) == len(self._generator_aids):
            self._admm_ready = True
            for idx in sorted(self._pending):
                self._apply_schedule_index(idx, self._pending[idx])
            self._pending.clear()

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        bucket = self.demand_map.setdefault(message.time, [])
        bucket.append(float(message.power_load))
        if len(bucket) != self.number_loads:
            return  # wait until all loads report for this timestep

        total = float(sum(bucket))
        idx = self._time_to_index.get(round(float(message.time), 6))
        if idx is None:
            return

        if self._admm_ready:
            self._apply_schedule_index(idx, total)
        else:
            self._pending[idx] = total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lookup_ts(scenario: ScenarioData, ref: Any) -> Any | None:
    """Return the timeseries for *ref*, falling back to attribute-based lookup.

    ComponentRef equality may fail across subclasses, so we also try matching
    on ``element_type`` and ``component_id`` directly.
    """
    ts = scenario.timeseries.get(ref)
    if ts is not None:
        return ts
    for k, v in scenario.timeseries.items():
        if (
            getattr(k, "element_type", None) == getattr(ref, "element_type", None)
            and getattr(k, "component_id", None) == getattr(ref, "component_id", None)
        ):
            return v
    return None


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
) -> Any:
    """Return a callback that stores a generator's schedule and notifies the leader."""

    def handle_admm_finished(algorithm: Any, role: Any, aid: str) -> None:
        schedule_by_aid[aid] = np.asarray(algorithm.x, dtype=float).copy()
        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(ADMMFinishedInfo(aid=aid), leader_addr)
            )
        logger.info("ADMM finished for %s (schedule len=%s)", aid, schedule_by_aid[aid].size)

    return handle_admm_finished


def _scalar(v: Any) -> float:
    arr = np.asarray(v).ravel()
    return float(arr[0]) if arr.size else 0.0


def _write_agent_recordings_csv(world: Any, path: str) -> None:
    """Write all per-agent recordings to a single wide CSV."""
    frames: list[pd.DataFrame] = []
    for key, rec in world.data_agent_collections.items():
        if not rec.timeseries:
            continue
        length = min([len(rec.time)] + [len(v) for v in rec.timeseries.values()])
        data = {
            f"{key}:{aid}": [_scalar(v) for v in values[:length]]
            for aid, values in rec.timeseries.items()
        }
        data["time"] = rec.time[:length]
        frames.append(pd.DataFrame(data).set_index("time"))
    if frames:
        pd.concat(frames, axis=1).to_csv(path)
    else:
        pd.DataFrame().to_csv(path)


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.00005,
    name_base: str = "admm",
    simulate_days: int = 3,
) -> None:
    """Run the ADMM benchmark once and write CSV + PDF outputs."""

    # --- World setup ---
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    behavior = PyPSABehavior.from_scenario(scenario)
    com_sim = SimpleCommunicationSimulation(default_delay_s=delay_s, loss_percent=loss_percent)
    world = create_world(
        start_time=0.0,
        communication_sim=com_sim,
        environment=DefaultEnvironment(behavior=behavior),
    )

    # --- Demand target and time index ---
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for ADMM scenario.")

    load_series_0 = _lookup_ts(scenario, load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    horizon = len(time_index)

    # Sum all load timeseries into a single demand target vector.
    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(scenario, ref)
        if s is None:
            raise RuntimeError(f"Load timeseries missing for {ref}.")
        target_series += np.asarray(s.reindex(time_index), dtype=float)

    # Map simulation-seconds → horizon index so set-points can be applied on time.
    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        time_to_index[round(float((dt - start_dt).total_seconds()), 6)] = i

    # --- Generator agent creation ---
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    gen_refs = [g for g in gen_refs if "hydro" not in g.component_id]
    gen_refs = [
        g for g in gen_refs
        if behavior._dataframe_for(g.element_type).loc[g.component_id].get("p_nom", 0.0) != 0.0
    ]

    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]
    dispatch_epsilon = 0.1
    rho = 0.2

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(
        leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid
    )

    gen_specs: list[ADMMGeneratorSpec] = []
    gen_agents: list[RoleAgent] = []
    # Storage actors are built first but registered after pass-1 prices are known.
    storage_refs: list[Any] = []
    storage_precompute_actors: list[Any] = []

    for ref in gen_refs:
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
        p_nom = float(statics.get("p_nom", 0.0))

        ts = _lookup_ts(scenario, ref)
        if ts is None:
            p_max_vec = np.full(horizon, p_nom, dtype=float)
        else:
            values = np.asarray(ts.reindex(time_index), dtype=float)
            p_max_vec = values * p_nom if ref.element_type == RENEWABLE else values

        if ref.element_type == STORAGE:
            # Extract PyPSA storage parameters.
            p_min_pu = float(statics.get("p_min_pu", -1.0))
            p_max_pu = float(statics.get("p_max_pu", 1.0))
            p_charge_max = max(0.0, -p_min_pu * p_nom if p_min_pu < 0.0 else p_nom)
            p_discharge_max = max(0.0, p_max_pu * p_nom)
            e_max = max(1e-6, p_nom * float(statics.get("max_hours", 100.0)))
            eta_charge = float(statics.get("efficiency_store", statics.get("efficiency_charge", 0.95)))
            eta_discharge = float(statics.get("efficiency_dispatch", statics.get("efficiency_discharge", 0.95)))
            soc_initial_raw = statics.get("state_of_charge_initial", np.nan)
            # Fall back to 50 % SOC when the network does not specify an initial SOC.
            soc_initial_abs = (
                float(soc_initial_raw)
                if np.isfinite(soc_initial_raw) and float(soc_initial_raw) > 1e-9
                else 0.5 * e_max
            )
            e_initial = float(np.clip(soc_initial_abs / e_max, 0.0, 1.0))

            storage_refs.append(ref)
            storage_precompute_actors.append(
                create_admm_storage_actor(
                    horizon=horizon,
                    e_max=e_max,
                    p_charge_max=p_charge_max,
                    p_discharge_max=p_discharge_max,
                    eta_charge=max(1e-6, eta_charge),
                    eta_discharge=max(1e-6, eta_discharge),
                    e_initial=e_initial,
                    e_final=e_initial,
                    charge_cost=max(0.0, cost),
                    discharge_cost=max(0.0, cost),
                    epsilon=dispatch_epsilon,
                    n_participants=n_gens,
                )
            )
            # Storage agent registration is deferred to the 2-pass block below.

        else:
            lb_vec = np.zeros(horizon, dtype=float)
            ub_vec = np.asarray(p_max_vec, dtype=float)
            actor = create_admm_economic_dispatch_actor(
                lb=lb_vec, u=ub_vec, cost=cost,
                n_participants=n_gens, epsilon=dispatch_epsilon,
            )
            gen_specs.append(ADMMGeneratorSpec(
                cost=np.full(horizon, cost, dtype=float), lb=lb_vec, ub=ub_vec,
            ))
            agent = agent_composed_of(ADMMGeneratorRole(actor, finish_callback))
            world.register(agent, suggested_aid=ref.component_id)
            world.environment.install(agent, id=ref)
            gen_agents.append(agent)

    # --- Storage 2-pass pre-scheduling ---
    # Pass 1: derive clearing prices from thermals/wind on the original demand.
    # Pass 2: register each storage unit as a FixedScheduleActor and adjust the
    #         thermal target so thermals exactly cover demand minus storage.
    if storage_precompute_actors and gen_specs:
        precompute_data = create_admm_sharing_data(
            target=target_series, generators=gen_specs, epsilon=dispatch_epsilon,
        )
        pi0 = rho * n_gens * _z_from_clearing_prices(precompute_data, rho, n_gens)

        storage_net = np.zeros(horizon, dtype=float)
        for ref, precompute_actor in zip(storage_refs, storage_precompute_actors):
            sched = _storage_schedule_from_price(precompute_actor, pi0)
            storage_net += sched
            agent = agent_composed_of(ADMMGeneratorRole(FixedScheduleActor(sched), finish_callback))
            world.register(agent, suggested_aid=ref.component_id)
            world.environment.install(agent, id=ref)
            gen_agents.append(agent)

        # Thermals must cover demand minus what storage contributes (can be negative
        # when storage charges, which increases the effective thermal requirement).
        adjusted_target = np.clip(target_series - storage_net, 0.0, None)
    else:
        # No thermal specs to compute clearing prices from; register storage actors
        # directly and let them respond to whatever price signal they receive.
        for ref, actor in zip(storage_refs, storage_precompute_actors):
            agent = agent_composed_of(ADMMGeneratorRole(actor, finish_callback))
            world.register(agent, suggested_aid=ref.component_id)
            world.environment.install(agent, id=ref)
            gen_agents.append(agent)
        adjusted_target = target_series

    if not gen_agents:
        raise RuntimeError("No generator agents found for ADMM scenario.")

    # --- Leader agent ---
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    coordinator = create_sharing_target_distance_admm_coordinator()
    coordinator.rho = rho
    leader_agent.add_role(CoordinatorRole(coordinator))
    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            admm_trigger=leader_addr,
            time_to_index=time_to_index,
            target_series=adjusted_target,
            schedule_by_aid=schedule_by_aid,
            generator_specs=gen_specs,
            dispatch_epsilon=dispatch_epsilon,
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    # Additional load agents (load_refs[0] is hosted on the leader agent).
    for ref in load_refs[1:]:
        agent = agent_composed_of(PowerLoadMonitoring(behavior=behavior, target=leader_addr))
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    # All optimisation agents form a complete communication graph.
    all_opt_agents = gen_agents + [leader_agent]
    auto_assign(complete_topology(len(all_opt_agents)), all_opt_agents)

    # --- Recordings ---
    record_agent_having(
        world, "target", PowerLoadAggregator,
        lambda a: next((r.target for r in a.roles if isinstance(r, PowerLoadAggregator)), 0.0),
    )
    record_agent_having(
        world, "P", ADMMGeneratorRole,
        lambda a: float(behavior.observe(a.aid, "active_power") or 0.0),
    )

    # --- Run simulation ---
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # --- Write outputs ---
    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    _, Y_t, _ = agent_recording_as_plottable(world, "target")
    target_recorded = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))

    _write_agent_recordings_csv(world, f"{name_base}-df.csv")
    visualize_results(world, write_to=f"{name_base}-observation.pdf")
    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_recorded,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power – ADMM",
        write_to=f"{name_base}-stacked.pdf",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=str,
        default="toy",
        help=(
            "Network source: 'toy' (built-in 5-bus fixture), "
            f"a PyPSA example name ({', '.join(available_examples())}), "
            "or a path to a .nc/.h5/.xlsx file or CSV folder."
        ),
    )
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--loss-percent", type=float, default=0.00005)
    parser.add_argument("--name-base", type=str, default="admm_withlosses")
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenario = (
        build_toy_network(periods=args.simulate_days * 24)
        if args.network == "toy"
        else load_scenario(args.network)
    )
    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=args.delay_s,
            loss_percent=args.loss_percent,
            name_base=args.name_base,
            simulate_days=args.simulate_days,
        )
    )


if __name__ == "__main__":
    # main() # commeted out for testing

    ###############
    ### Testing ###
    ###############

    simulate_days = 3

    # scenario = build_toy_network(periods=simulate_days * 24) # toy
    # scenario = load_scenario("storage-hvdc")
    scenario = load_scenario("../networks/base_s_1_elec_.nc")

    logging.basicConfig(level=getattr(logging, "INFO", logging.INFO))

    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=0.02,
            loss_percent=0.0,
            name_base="admm_without_losses",
            simulate_days=simulate_days,
        )
    )
