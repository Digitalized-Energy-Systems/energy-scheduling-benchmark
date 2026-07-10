"""Distributed economic-dispatch scenario using iterative (exchange) ADMM.

Mirrors the archived ``admm_old`` prototype's collector/agent structure —
each generator (and storage unit) solves its own local proximal QP each
round; the leader runs the exchange-ADMM ``Σxᵢ = target`` consensus loop
(:func:`~distributed_resource_optimization.create_consensus_target_reach_admm_coordinator`)
until the primal/dual residuals converge, rather than a single-shot
merit-order clearing price. Thermals/renewables use a box-bounded proximal
actor (:func:`~distributed_resource_optimization.create_admm_flex_actor_box_bounded`);
storage uses a SOC-constrained proximal actor
(:func:`~distributed_resource_optimization.create_admm_proximal_storage_actor`)
that co-optimizes charge/discharge timing within the same loop instead of
being pre-scheduled from a clearing price.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    create_admm_flex_actor_box_bounded,
    create_admm_proximal_storage_actor,
    create_admm_start_consensus,
    create_consensus_target_reach_admm_coordinator,
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
    SchedulingBehavior,
)
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    cost_over_time,
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
    _keep_hourly,
    _lookup_ts,
    _write_agent_recordings_csv,
    build_behavior,
    build_scenario_argparser,
    build_toy_network,
    compute_overall_cost,
    require_lossless_transport,
    resolve_scenario,
)

logger = logging.getLogger(__name__)


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
        behavior: SchedulingBehavior,
        number_loads: int,
        generator_aids: list[str],
        admm_trigger: Any,
        time_to_index: dict[float, int],
        target_series: np.ndarray,
        schedule_by_aid: dict[str, np.ndarray],
    ) -> None:
        def build_start_message() -> Any:
            return StartCoordinatedDistributedOptimization(
                input=create_admm_start_consensus(np.asarray(target_series, dtype=float))
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
            demand_target=target_series,
            balance_label="ADMM",
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
    loss_percent: float = 0.0,
    name_base: str = "admm",
    simulate_days: int = 3,
) -> None:
    """Run the ADMM benchmark once and write CSV + PDF outputs."""
    require_lossless_transport(loss_percent, "ADMM")

    # --- World setup ---
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    behavior = build_behavior(scenario)
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
    # Filter out hydro: its dispatch is driven by natural inflow, which the
    # storage actor does not model (it assumes freely schedulable
    # charge/discharge), so hydro units cannot participate meaningfully.
    gen_refs = [g for g in gen_refs if "hydro" not in g.component_id]
    gen_refs = [
        g
        for g in gen_refs
        if behavior.get_statics(g).get("p_nom", 0.0)
        != 0.0
    ]

    generator_aids = [ref.component_id for ref in gen_refs]
    rho = 1.0
    max_iters = 2000

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(
        leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid
    )

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}

    # --- Register one proximal ADMM actor per generator/storage unit ---
    for ref in gen_refs:
        statics = behavior.get_statics(ref)
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
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
            actor = create_admm_proximal_storage_actor(
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
            )
        else:
            lb_vec = np.zeros(horizon, dtype=float)
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                lb_vec = np.full(horizon, max(0.0, p_min_pu * p_nom), dtype=float)
            actor = create_admm_flex_actor_box_bounded(
                lb=lb_vec,
                u=np.asarray(p_max_vec, dtype=float),
                S=np.full(horizon, cost, dtype=float),
            )

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

    coordinator = create_consensus_target_reach_admm_coordinator(
        rho=rho, max_iters=max_iters, alpha=0.0
    )
    leader_agent.add_role(CoordinatorRole(coordinator))
    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            admm_trigger=leader_addr,
            time_to_index=time_to_index,
            target_series=target_series,
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
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")

    # Drop sub-second convergence-phase noise: keep the last recorded state
    # per hourly snapshot so the CSV and plots show one row per PyPSA timestep.
    t_P, Y_P = _keep_hourly(t_P, Y_P)
    t_t, Y_t = _keep_hourly(t_t, Y_t)
    target_recorded = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    m = min(len(t_P), len(target_recorded))
    t_P, Y_P, target_recorded = t_P[:m], Y_P[:m], target_recorded[:m]

    total_cost, cost_series = compute_overall_cost(cost_by_aid, t_P, Y_P, labels_P)
    logger.info("%s: overall cost = %.2f", name_base, total_cost)
    annotation = f"Total cost: {total_cost:,.2f}"

    _write_agent_recordings_csv(
        world, f"{name_base}-df.csv", snapshot_step_s=3600.0, extra=cost_series
    )
    visualize_results(
        world, write_to=f"{name_base}-observation.pdf", annotation=annotation
    )
    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_recorded,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power – ADMM",
        annotation=annotation,
        write_to=f"{name_base}-stacked.pdf",
    )
    cost_over_time(
        np.asarray(cost_series.index, dtype=float) / 3600.0,
        cost_series.to_numpy(),
        title="Cost per timestep",
        annotation=annotation,
        write_to=f"{name_base}-cost.pdf",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_scenario_argparser(__doc__, default_name_base="admm_withlosses")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenario = resolve_scenario(args.network, simulate_days=args.simulate_days)
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
