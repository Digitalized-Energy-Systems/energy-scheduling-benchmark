"""Distributed economic-dispatch scenario using sharing ADMM.

Two-pass approach
-----------------
1. Merit-order clearing prices are computed from thermal/renewable specs.
2. Each storage unit is pre-scheduled by a CVXPY LP (:func:`solve_battery_price_schedule`)
   that enforces SOC constraints exactly; its output is subtracted from the
   demand target and replayed via a :class:`FixedScheduleActor` during ADMM.
3. Thermals and renewables respond to the clearing price in one ADMM round via
   :class:`LinearCostEconomicDispatchADMMFlexActor`.
4. As simulation time advances the leader applies per-timestep set-points.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    ADMMAnswer,
    ADMMGeneratorSpec,
    ADMMMessage,
    create_admm_economic_dispatch_actor,
    create_admm_sharing_data,
    create_sharing_target_distance_admm_coordinator,
    solve_battery_price_schedule,
)
from distributed_resource_optimization.algorithm.admm.sharing_admm import (
    _z_from_clearing_prices,
    create_sharing_admm_start,
)
from distributed_resource_optimization.carrier.mango import (
    CoordinatorRole,
    DistributedOptimizationRole,
    OptimizationFinishedMessage,
    StartCoordinatedDistributedOptimization,
)
from mango import RoleAgent, agent_composed_of, auto_assign, complete_topology
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import (
    create_world,
    discrete_step_until,
    record_agent_having,
)

from energy_scheduling_benchmark import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    PyPSABehavior,
)
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    stacked_area,
    visualize_results,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as ADMMFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator as _BasePowerLoadAggregator,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadMonitoring,
    ScenarioData,
    _clip_scenario,
    _lookup_ts,
    _write_agent_recordings_csv,
    build_scenario_argparser,
    build_toy_network,
    load_scenario,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


class FixedScheduleActor:
    """ADMM participant that always replies with a pre-computed schedule.

    Used for storage units whose schedule was computed by the CVXPY LP before
    the ADMM run.  Participating in the ADMM protocol allows the coordinator
    to wait for all generators (including storage) to reply.
    """

    def __init__(self, schedule: np.ndarray) -> None:
        self.x = np.asarray(schedule, dtype=float).copy()

    async def on_exchange_message(
        self, carrier: Any, message_data: Any, meta: dict
    ) -> None:
        if not isinstance(message_data, ADMMMessage):
            return
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


class PowerLoadAggregator(_BasePowerLoadAggregator):
    """Leader role: fires ADMM once at startup, then dispatches the schedule."""

    def __init__(
        self,
        *,
        behavior: PyPSABehavior,
        number_loads: int,
        generator_aids: list[str],
        admm_trigger: Any,
        time_to_index: dict[float, int],
        adjusted_target_series: np.ndarray,
        gen_specs: list[ADMMGeneratorSpec],
        schedule_by_aid: dict[str, np.ndarray],
    ) -> None:
        def build_start_message() -> Any:
            data = create_admm_sharing_data(
                target=np.asarray(adjusted_target_series, dtype=float),
                generators=gen_specs if gen_specs else None,
            )
            return StartCoordinatedDistributedOptimization(
                input=create_sharing_admm_start(data)
            )

        super().__init__(
            behavior=behavior,
            number_loads=number_loads,
            generator_aids=generator_aids,
            trigger=admm_trigger,
            time_to_index=time_to_index,
            schedule_by_aid=schedule_by_aid,
            finished_message_type=ADMMFinishedInfo,
            build_start_message=build_start_message,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
) -> Any:
    """Return a callback that stores a generator's schedule and notifies the leader.

    ADMM's finish notification is fired directly by :class:`ADMMGeneratorRole`
    with signature ``(algorithm, role, aid)`` — unlike the carrier-based
    algorithms (consensus/diffusion/FDGDM/DEED-ADMM) which read the schedule
    off ``carrier._parent``, so this stays a scenario-specific helper rather
    than the shared ``_common.make_finish_callback``.
    """

    def handle_admm_finished(algorithm: Any, role: Any, aid: str) -> None:
        schedule_by_aid[aid] = np.asarray(algorithm.x, dtype=float).copy()
        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(ADMMFinishedInfo(aid=aid), leader_addr)
            )
        logger.info(
            "ADMM finished for %s (schedule len=%s)", aid, schedule_by_aid[aid].size
        )

    return handle_admm_finished


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

    scenario = _clip_scenario(scenario, simulate_days)
    behavior = PyPSABehavior.from_scenario(scenario)
    com_sim = SimpleCommunicationSimulation(
        default_delay_s=delay_s, loss_percent=loss_percent
    )
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

    time_index = load_series_0.index[: simulate_days * 24]
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
        time_to_index[round(float((dt - start_dt).total_seconds()), 6)] = i

    # --- Generator agent creation ---
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    gen_refs = [g for g in gen_refs if "hydro" not in g.component_id]
    gen_refs = [
        g
        for g in gen_refs
        if behavior._dataframe_for(g.element_type).loc[g.component_id].get("p_nom", 0.0)
        != 0.0
    ]

    generator_aids = [ref.component_id for ref in gen_refs]
    n_gens = len(gen_refs)
    rho = 0.2
    dispatch_epsilon = 0.1

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(
        leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid
    )

    gen_agents: list[RoleAgent] = []
    gen_specs: list[ADMMGeneratorSpec] = []  # merit-order specs (thermals + renewables)

    # --- Pass 1: thermals and renewables ---
    storage_refs_and_params: list[tuple[Any, dict]] = []
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
            p_min_pu = float(statics.get("p_min_pu", -1.0))
            p_max_pu = float(statics.get("p_max_pu", 1.0))
            p_charge_max = max(0.0, -p_min_pu * p_nom if p_min_pu < 0.0 else p_nom)
            p_discharge_max = max(0.0, p_max_pu * p_nom)
            e_max = max(1e-6, p_nom * float(statics.get("max_hours", 100.0)))
            eta_charge = float(
                statics.get("efficiency_store", statics.get("efficiency_charge", 0.95))
            )
            eta_discharge = float(
                statics.get(
                    "efficiency_dispatch", statics.get("efficiency_discharge", 0.95)
                )
            )
            soc_initial_raw = statics.get("state_of_charge_initial", np.nan)
            soc_initial_abs = (
                float(soc_initial_raw)
                if np.isfinite(soc_initial_raw) and float(soc_initial_raw) > 1e-9
                else 0.5 * e_max
            )
            e_initial = float(np.clip(soc_initial_abs / e_max, 0.0, 1.0))
            storage_refs_and_params.append(
                (
                    ref,
                    {
                        "p_charge_max": p_charge_max,
                        "p_discharge_max": p_discharge_max,
                        "e_max": e_max,
                        "eta_charge": max(1e-6, eta_charge),
                        "eta_discharge": max(1e-6, eta_discharge),
                        "e_initial": e_initial,
                        "e_final": e_initial,
                        "charge_cost": max(0.0, cost),
                        "discharge_cost": max(0.0, cost),
                    },
                )
            )
            continue  # processed in pass 2

        lb_vec = np.zeros(horizon, dtype=float)
        if ref.element_type == THERMAL:
            p_min_pu = float(statics.get("p_min_pu", 0.0))
            lb_vec = np.full(horizon, max(0.0, p_min_pu * p_nom), dtype=float)
        gen_specs.append(
            ADMMGeneratorSpec(
                cost=np.full(horizon, cost, dtype=float),
                lb=lb_vec,
                ub=np.asarray(p_max_vec, dtype=float),
            )
        )
        actor = create_admm_economic_dispatch_actor(
            lb_vec,
            np.asarray(p_max_vec, dtype=float),
            cost=cost,
            n_participants=n_gens,
            epsilon=dispatch_epsilon,
        )
        agent = agent_composed_of(ADMMGeneratorRole(actor, finish_callback))
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    # --- Compute clearing prices for battery pre-scheduling ---
    adjusted_target = target_series.copy()
    if gen_specs:
        precompute_data = create_admm_sharing_data(
            target=target_series, generators=gen_specs, epsilon=dispatch_epsilon
        )
        pi0 = rho * n_gens * _z_from_clearing_prices(precompute_data, rho, n_gens)
    else:
        pi0 = np.zeros(horizon, dtype=float)

    # --- Pass 2: storage — pre-schedule via LP, then register FixedScheduleActor ---
    for ref, params in storage_refs_and_params:
        battery_sched = solve_battery_price_schedule(horizon=horizon, pi=pi0, **params)
        adjusted_target -= battery_sched
        schedule_by_aid[ref.component_id] = (
            battery_sched  # pre-stored; callback overwrites identically
        )
        actor = FixedScheduleActor(battery_sched)
        agent = agent_composed_of(ADMMGeneratorRole(actor, finish_callback))
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

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
            adjusted_target_series=adjusted_target,
            gen_specs=gen_specs,
            schedule_by_aid=schedule_by_aid,
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    all_opt_agents = gen_agents + [leader_agent]
    auto_assign(complete_topology(len(all_opt_agents)), all_opt_agents)

    # --- Recordings ---
    record_agent_having(
        world,
        "target",
        PowerLoadAggregator,
        lambda a: next(
            (r.target for r in a.roles if isinstance(r, PowerLoadAggregator)), 0.0
        ),
    )
    record_agent_having(
        world,
        "P",
        ADMMGeneratorRole,
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
    parser = build_scenario_argparser(__doc__, default_name_base="admm_withlosses")
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
    main()
