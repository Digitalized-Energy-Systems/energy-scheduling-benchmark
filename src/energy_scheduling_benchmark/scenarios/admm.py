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
from mango.simulation.world import discrete_step_until

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
    StorageParams,
    _clip_scenario,
    build_demand_horizon,
    build_group_map,
    build_p_max_vec,
    build_toy_network,
    build_world,
    collect_generator_refs,
    install_standard_recordings,
    require_lossless_transport,
    run_scenario_main,
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
    rho: float = 1.0,
    max_iter: int = 2000,
    balance_tol: float = 0.01,
    strict: bool = False,
) -> None:
    """Run the ADMM benchmark once and write CSV + PDF outputs.

    :param rho: ADMM penalty parameter.
    :param max_iter: Maximum exchange-ADMM coordinator iterations.
    :param balance_tol: Max allowed per-timestep |generation - demand| /
        demand before the finished schedule is flagged as not converged.
    :param strict: If true, raise instead of writing outputs when the
        power-balance check fails.
    """
    require_lossless_transport(loss_percent, "ADMM")

    # --- World setup ---
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(scenario, delay_s=delay_s, loss_percent=loss_percent)

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
    group_map = build_group_map(gen_refs, statics_by_ref=statics_by_ref)

    generator_aids = [ref.component_id for ref in gen_refs]

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(
        leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid
    )

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
                charge_cost=max(0.0, cost),
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
        rho=rho, max_iters=max_iter, alpha=0.0
    )
    leader_agent.add_role(CoordinatorRole(coordinator))

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
        demand_target=target_series,
        balance_label="ADMM",
        balance_tol=balance_tol,
    )
    leader_agent.add_role(aggregator)
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
    install_standard_recordings(world, behavior, role_cls=ADMMGeneratorRole)

    # --- Run simulation ---
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # --- Write outputs ---
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        stacked_title="Stacked power – ADMM",
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
        group_map=group_map,
        net=scenario.net,
        label="ADMM",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _add_admm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rho", type=float, default=1.0, help="ADMM penalty parameter."
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
        lossless_only=True,
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
