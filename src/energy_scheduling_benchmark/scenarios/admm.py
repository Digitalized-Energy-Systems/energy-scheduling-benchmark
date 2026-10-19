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

import argparse
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
from mango import RoleAgent, agent_composed_of

from energy_scheduling_benchmark import (
    LOAD,
    STORAGE,
    THERMAL,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as ADMMFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
    PowerLoadMonitoring,
    ScenarioData,
    SlackSpec,
    StorageParams,
    _clip_scenario,
    build_demand_horizon,
    build_group_map,
    build_p_max_vec,
    build_toy_network,
    build_world,
    collect_generator_refs,
    install_standard_recordings,
    install_topology,
    loss_tolerance,
    register_component_agent,
    resolve_loss_rate,
    run_scenario_main,
    run_simulation,
    thermal_cost_quad,
    write_scenario_outputs,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
) -> Any:
    """Return a callback that sends a generator's schedule to the leader.

    The leader normally has every schedule already (the coordinator runs on
    its agent, see ``start_locally``); this message is the redundant path.

    ADMM's finish notification is fired directly by :class:`ADMMGeneratorRole`
    with signature ``(algorithm, role, aid)`` — unlike the carrier-based
    algorithms (consensus/diffusion/FDGDM/DEED-ADMM) which read the schedule
    off ``carrier._parent``, so this stays a scenario-specific helper rather
    than the shared ``_common.make_finish_callback``.
    """

    def handle_admm_finished(algorithm: Any, role: Any, aid: str) -> None:
        schedule = np.asarray(algorithm.x, dtype=float).copy()
        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            role.context.schedule_instant_message(
                ADMMFinishedInfo(aid=aid, schedule=schedule), leader_addr
            )
        logger.info("ADMM finished for %s (schedule len=%s)", aid, schedule.size)

    return handle_admm_finished


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_rate: float = 0.0,
    seed: int = 0,
    round_timeout: float | None = None,
    result_poll_s: float = 300.0,
    result_retries: int = 12,
    load_timeout: float | None = None,
    max_missing_rounds: int | None = 20,
    topology: str = "complete",
    name_base: str = "admm",
    simulate_days: int = 3,
    rho: float = 0.1,
    max_iter: int = 2000,
    balance_tol: float = 0.01,
    strict: bool = False,
    slack: SlackSpec | None = None,
    cost_model: str = "linear",
) -> None:
    """Run the ADMM benchmark once and write CSV + PDF outputs.

    :param rho: ADMM penalty parameter. Each unit's x-update moves by roughly
        marginal_cost / rho per iteration, so with linear €/MWh costs over
        GW-scale fleets rho = 1 converges far too slowly to separate close
        merit-order prices (hits max_iter, up to +7 % cost on the PyPSA-Eur
        extracts); 0.1 reaches the central-dispatch optimum.
    :param max_iter: Maximum exchange-ADMM coordinator iterations.
    :param balance_tol: Max allowed per-timestep |generation - demand| /
        demand before the finished schedule is flagged as not converged.
    :param strict: If true, raise instead of writing outputs when the
        power-balance check fails.
    :param slack: Optional slack node (see :class:`SlackSpec`) that imports
        any shortfall and exports any surplus; off when ``None``.
    """
    loss_rate = resolve_loss_rate(loss_rate)
    lt = loss_tolerance(
        delay_s,
        loss_rate=loss_rate,
        round_timeout=round_timeout,
        result_poll_s=result_poll_s,
        result_retries=result_retries,
        load_timeout=load_timeout,
        max_missing_rounds=max_missing_rounds,
    )

    # --- World setup ---
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(
        scenario, delay_s=delay_s, loss_rate=loss_rate, seed=seed
    )

    # --- Demand target and time index ---
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for ADMM scenario.")

    demand = build_demand_horizon(
        behavior, scenario, load_refs, simulate_days=simulate_days
    )
    time_index, horizon = demand.time_index, demand.horizon
    target_series = demand.target_series
    time_to_index = demand.time_to_index

    # --- Generator agent creation ---
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    cost_quad_by_aid = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    group_map = build_group_map(gen_refs, statics_by_ref=statics_by_ref)

    generator_aids = [ref.component_id for ref in gen_refs]

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(leader_addr_ref=leader_addr_ref)

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}

    # --- Register one proximal ADMM actor per generator/storage unit ---
    for ref in gen_refs:
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            sp = StorageParams.from_statics(statics, p_nom)
            actor = create_admm_proximal_storage_actor(
                horizon=horizon,
                e_max=sp.e_max,
                p_charge_max=sp.p_charge_max,
                p_discharge_max=sp.p_discharge_max,
                eta_charge=sp.eta_charge,
                eta_discharge=sp.eta_discharge,
                e_initial=sp.e_initial,
                e_final=sp.e_initial,
                # Discharge-only cost, as in central dispatch and
                # compute_overall_cost: a full cycle is priced once.
                charge_cost=0.0,
                discharge_cost=max(0.0, cost),
            )
        else:
            p_max_vec = build_p_max_vec(scenario, ref, statics, time_index, horizon)
            lb_vec = np.zeros(horizon, dtype=float)
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                lb_vec = np.full(horizon, max(0.0, p_min_pu * p_nom), dtype=float)
            actor = create_admm_flex_actor_box_bounded(
                lb=lb_vec,
                u=np.asarray(p_max_vec, dtype=float),
                S=np.full(horizon, cost, dtype=float),
                Q=cost_quad_by_aid.get(ref.component_id),
            )

        agent = agent_composed_of(ADMMGeneratorRole(actor, finish_callback))
        register_component_agent(world, agent, ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for ADMM scenario.")

    # --- Leader agent ---
    leader_agent = RoleAgent()
    register_component_agent(world, leader_agent, load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    coordinator = create_consensus_target_reach_admm_coordinator(
        rho=rho,
        max_iters=max_iter,
        alpha=0.0,
        round_timeout=lt.round_timeout,
        max_missing_rounds=lt.max_missing_rounds,
    )
    coord_role = CoordinatorRole(coordinator)
    leader_agent.add_role(coord_role)

    def start_locally(start: StartCoordinatedDistributedOptimization) -> None:
        # The coordinator lives on the leader agent: start it without a
        # message (which could be lost) and take its results directly.
        done = coord_role.start(start.input)

        def deliver(future: Any) -> None:
            if future.cancelled() or future.exception() is not None:
                return
            aggregator.algorithm_stats = {
                "iterations": coordinator.iterations,
                "converged": coordinator.converged,
                "missed_answers": coordinator.missed_answers,
            }
            addrs = list(coord_role.context.neighbors())  # coordinator's order
            aggregator.accept_schedules(
                {
                    a.aid: np.asarray(x, dtype=float)
                    for a, x in zip(addrs, future.result())
                }
            )

        done.add_done_callback(deliver)

    def build_start_message() -> Any:
        return StartCoordinatedDistributedOptimization(
            input=create_admm_start_consensus(np.asarray(target_series, dtype=float))
        )

    aggregator = PowerLoadAggregator(
        behavior=behavior,
        number_loads=len(load_refs),
        generator_aids=generator_aids,
        trigger=leader_addr,
        time_to_index=time_to_index,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=ADMMFinishedInfo,
        build_start_message=build_start_message,
        start_locally=start_locally,
        load_timeout=lt.load_timeout,
        request_timeout=lt.request_timeout,
        demand_target=target_series,
        balance_label="ADMM",
        balance_tol=balance_tol,
        slack=slack,
    )
    leader_agent.add_role(aggregator)
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        register_component_agent(world, agent, ref)

    all_opt_agents = gen_agents + [leader_agent]
    # The coordinator (on the leader) reaches every participant; its
    # messages travel over the chosen topology.
    install_topology(world, scenario, topology, gen_agents, logical=all_opt_agents)

    # --- Recordings ---
    install_standard_recordings(world, behavior, role_cls=ADMMGeneratorRole)

    # --- Run simulation ---
    async with world:
        await run_simulation(world, simulate_days * 24 * 3600.0)

    # --- Write outputs ---
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        cost_quad_by_aid=cost_quad_by_aid,
        stacked_title="Stacked power – ADMM",
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
        slack=slack,
        group_map=group_map,
        net=scenario.net,
        label="ADMM",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _add_admm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rho", type=float, default=0.1, help="ADMM penalty parameter."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum exchange-ADMM coordinator iterations.",
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="admm",
        extra_args=_add_admm_args,
        extra_kwargs=("rho", "max_iter"),
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
