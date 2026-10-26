"""Distributed economic-dispatch DEED-ADMM scenario.

Implementation of the DEED-ADMM algorithm (Zhu et al. 2025) for the
electricity-only PyPSA benchmark.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. The leader pre-computes the aggregated demand vector and fires a
   single DEED-ADMM kick-off message to the first generator agent.
3. Each generator agent runs peer-to-peer DEED-ADMM iterations with all
   other generator agents, then reports back to the leader.
4. After all generators finish, the leader applies each generator's
   computed per-timestep power set-points as the simulation clock advances.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    DEEDADMMMessage,
    create_deed_admm_renewable_participant,
    create_deed_admm_storage_participant,
    create_deed_admm_thermal_participant,
)
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
)
from mango import (
    RoleAgent,
    agent_composed_of,
)

from energy_scheduling_benchmark import (
    LOAD,
    RENEWABLE,
    STORAGE,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as DEEDADMMFinishedInfo,
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
    build_unit_bounds,
    build_world,
    collect_generator_refs,
    generator_agent,
    install_topology,
    loss_tolerance,
    make_finish_callback,
    participant_graph,
    register_component_agent,
    resolve_loss_rate,
    run_scenario_main,
    run_simulation,
    thermal_cost_quad,
    write_scenario_outputs,
)
from energy_scheduling_benchmark.scenarios.fdgdm import PAPER_TOL_RATIO

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
    name_base: str = "deed_admm",
    simulate_days: int = 3,
    gamma: float = 0.05,
    max_iter: int = 500,
    tol_ratio: float | None = PAPER_TOL_RATIO,
    balance_tol: float = 0.01,
    strict: bool = False,
    slack: SlackSpec | None = None,
    cost_model: str = "quadratic",
) -> None:
    """Run the DEED-ADMM benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    gamma:
        ADMM penalty parameter γ (paper default: 0.05).
    max_iter:
        Maximum number of DEED-ADMM iterations (backstop).
    tol_ratio:
        Convergence stop (the paper defines none; same rule as FDGDM's Eq. 24
        stop): each agent is converged when its per-hour schedule step and
        its tracker ξ are below ``tol_ratio * p_nom``; the run ends when every
        agent and its neighbours agree. Default: FDGDM's paper ratio 0.01/45.
        ``None`` or ``0`` runs a fixed ``max_iter``.
    balance_tol:
        Max allowed per-timestep |generation - demand| / demand before the
        finished schedule is flagged as not converged.
    strict:
        If true, raise instead of writing outputs when the power-balance
        check fails.
    slack:
        Optional slack node (see :class:`SlackSpec`) that imports any
        shortfall and exports any surplus; off when ``None``.
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

    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(
        scenario, delay_s=delay_s, loss_rate=loss_rate, seed=seed
    )

    # ------------------------------------------------------------------
    # Build demand horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for DEED-ADMM scenario.")

    demand = build_demand_horizon(
        behavior, scenario, load_refs, simulate_days=simulate_days
    )
    time_index, horizon = demand.time_index, demand.horizon
    target_series = demand.target_series
    time_to_index = demand.time_to_index

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        finished_message_type=DEEDADMMFinishedInfo,
        algorithm_label="DEED-ADMM",
        schedule_attr="P",
    )

    # ------------------------------------------------------------------
    # Generator agents
    # ------------------------------------------------------------------
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    cost_quad_by_aid = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    group_map = build_group_map(gen_refs, statics_by_ref=statics_by_ref)
    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    # Demand allocation: only generators (thermal + renewable) share demand.
    # Storage contributes net injection rather than sharing d_i — the storage
    # participant factory below takes no d_i argument at all.
    pure_gen_refs = [r for r in gen_refs if r.element_type != STORAGE]
    n_gen_only = max(len(pure_gen_refs), 1)
    d_i_gen = target_series / n_gen_only

    pgraph = participant_graph(scenario, topology, gen_refs, attached=load_refs)
    lt = lt.for_hops(pgraph.hops_from(load_refs[0]))
    # W = I − βL must be doubly stochastic (Zhu et al. 2025, Assumption 1):
    # β = 1/n on a complete graph (the default), the max-degree weight
    # 1/(Δ + 1) on sparser ones (same value on a complete graph).
    beta = None if topology == "complete" else 1.0 / (pgraph.max_degree + 1)
    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}
    for ref in gen_refs:
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            sp = StorageParams.from_statics(statics, p_nom)
            participant = create_deed_admm_storage_participant(
                finish_callback,
                e_max=sp.e_max,
                p_charge_max=sp.p_charge_max,
                p_discharge_max=sp.p_discharge_max,
                eta_charge=sp.eta_charge,
                eta_discharge=sp.eta_discharge,
                e_initial=sp.e_initial,
                marginal_cost=cost,
                tau=horizon,
                gamma=gamma,
                max_iter=max_iter,
                round_timeout=lt.round_timeout,
                tol=tol_ratio * p_nom if tol_ratio else None,
                n_agents=n_gens,
                beta=beta,
                diameter=pgraph.diameter,
            )
        elif ref.element_type == RENEWABLE:
            participant = create_deed_admm_renewable_participant(
                finish_callback,
                p_max_timeseries=build_p_max_vec(
                    scenario, ref, statics, time_index, horizon
                ),
                d_i=d_i_gen,
                marginal_cost=cost,
                gamma=gamma,
                max_iter=max_iter,
                round_timeout=lt.round_timeout,
                tol=tol_ratio * p_nom if tol_ratio else None,
                n_agents=n_gens,
                beta=beta,
                diameter=pgraph.diameter,
            )
        else:
            p_min_vec, p_max_vec = build_unit_bounds(
                scenario, ref, statics, time_index, horizon
            )
            participant = create_deed_admm_thermal_participant(
                finish_callback,
                cost_quad=cost_quad_by_aid.get(ref.component_id, 0.0),
                p_min=p_min_vec,
                p_max=p_max_vec,
                marginal_cost=cost,
                d_i=d_i_gen,
                gamma=gamma,
                max_iter=max_iter,
                round_timeout=lt.round_timeout,
                tol=tol_ratio * p_nom if tol_ratio else None,
                n_agents=n_gens,
                beta=beta,
                diameter=pgraph.diameter,
            )

        opt_role = DistributedOptimizationRole(participant)
        agent = generator_agent(opt_role)
        register_component_agent(world, agent, ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for DEED-ADMM scenario.")

    # ------------------------------------------------------------------
    # Load agents
    # ------------------------------------------------------------------
    leader_agent = RoleAgent()
    register_component_agent(world, leader_agent, load_refs[0])

    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        return DEEDADMMMessage(
            lam=np.zeros(horizon),
            xi=np.zeros(horizon),
            k=0,
            data=None,
            initial=True,
        )

    aggregator = PowerLoadAggregator(
        behavior=behavior,
        number_loads=len(load_refs),
        generator_aids=generator_aids,
        trigger=gen_agents[0].addr,
        time_to_index=time_to_index,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=DEEDADMMFinishedInfo,
        build_start_message=build_start_message,
        generator_addrs={a.aid: a.addr for a in gen_agents},
        load_timeout=lt.load_timeout,
        request_timeout=lt.request_timeout,
        result_poll_s=lt.result_poll_s,
        result_retries=lt.result_retries,
        demand_target=target_series,
        balance_label="DEED-ADMM",
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

    # Fully-connected peer topology across all generator agents.
    install_topology(world, pgraph, gen_agents)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    async with world:
        await run_simulation(world, simulate_days * 24 * 3600.0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        cost_quad_by_aid=cost_quad_by_aid,
        stacked_title="Stacked power (DEED-ADMM)",
        cost_title="Cost per timestep (DEED-ADMM)",
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
        slack=slack,
        group_map=group_map,
        net=scenario.net,
        label="DEED-ADMM",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_deed_admm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
        help="ADMM penalty parameter γ (paper default: 0.05).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of DEED-ADMM iterations.",
    )
    parser.add_argument(
        "--tol-ratio",
        type=float,
        default=PAPER_TOL_RATIO,
        help=(
            "Convergence stop: per-hour step and tracker tolerance as a fraction "
            "of each unit's p_nom (default 0.01/45, as FDGDM); 0 disables it."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="deed_admm",
        extra_args=_add_deed_admm_args,
        extra_kwargs=("gamma", "max_iter", "tol_ratio"),
        argv=argv,
    )


if __name__ == "__main__":
    main()
