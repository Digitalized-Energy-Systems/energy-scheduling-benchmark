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
    auto_assign,
    complete_topology,
)
from mango.simulation.world import discrete_step_until

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
    _clip_scenario,
    _lookup_ts,
    build_demand_horizon,
    build_toy_network,
    build_world,
    collect_generator_refs,
    install_standard_recordings,
    make_finish_callback,
    require_lossless_transport,
    run_scenario_main,
    write_scenario_outputs,
)

# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.0,
    name_base: str = "deed_admm",
    simulate_days: int = 3,
    gamma: float = 0.05,
    max_iter: int = 500,
    balance_tol: float = 0.01,
    strict: bool = False,
) -> None:
    """Run the DEED-ADMM benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    gamma:
        ADMM penalty parameter γ (paper default: 0.05).
    max_iter:
        Maximum number of DEED-ADMM iterations.
    balance_tol:
        Max allowed per-timestep |generation - demand| / demand before the
        finished schedule is flagged as not converged.
    strict:
        If true, raise instead of writing outputs when the power-balance
        check fails.
    """
    require_lossless_transport(loss_percent, "DEED-ADMM")

    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(scenario, delay_s=delay_s, loss_percent=loss_percent)

    # ------------------------------------------------------------------
    # Build demand horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for DEED-ADMM scenario.")

    # Unlike its siblings, this does not truncate time_index to
    # simulate_days*24 — kept as-is (relies on _clip_scenario having already
    # trimmed the timeseries) rather than changed under a behaviour-
    # preserving refactor.
    demand = build_demand_horizon(behavior, scenario, load_refs, simulate_days=None)
    time_index, horizon = demand.time_index, demand.horizon
    target_series = demand.target_series
    time_to_index = demand.time_to_index

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=DEEDADMMFinishedInfo,
        algorithm_label="DEED-ADMM",
        schedule_attr="P",
    )

    # ------------------------------------------------------------------
    # Generator agents
    # ------------------------------------------------------------------
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    # Demand allocation: only generators (thermal + renewable) share demand.
    # Storage contributes net injection rather than sharing d_i — the storage
    # participant factory below takes no d_i argument at all.
    pure_gen_refs = [r for r in gen_refs if r.element_type != STORAGE]
    n_gen_only = max(len(pure_gen_refs), 1)
    d_i_gen = target_series / n_gen_only

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}
    for ref in gen_refs:
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            p_min_pu = float(statics.get("p_min_pu", -1.0))
            p_max_pu = float(statics.get("p_max_pu", 1.0))
            p_disch = max(0.0, p_max_pu * p_nom)
            p_chg = max(0.0, (-p_min_pu * p_nom) if p_min_pu < 0.0 else p_nom)
            max_hours = float(statics.get("max_hours", 100.0))
            e_max_val = max(1e-6, p_nom * max_hours)
            eta_c = float(
                statics.get("efficiency_store", statics.get("efficiency_charge", 0.95))
            )
            eta_d = float(
                statics.get(
                    "efficiency_dispatch", statics.get("efficiency_discharge", 0.95)
                )
            )
            soc0_raw = statics.get("state_of_charge_initial", np.nan)
            soc0_raw_f = (
                float(soc0_raw) if np.isfinite(float(soc0_raw)) else float("nan")
            )
            if not np.isnan(soc0_raw_f) and soc0_raw_f > 1e-9:
                soc0_abs = soc0_raw_f
            else:
                # PyPSA default is 0.0 MWh (empty); treat as "unspecified"
                # and start at 50 % SOC so the battery can participate.
                soc0_abs = 0.5 * e_max_val
            soc0 = float(np.clip(soc0_abs / e_max_val, 0.0, 1.0))
            participant = create_deed_admm_storage_participant(
                finish_callback,
                e_max=e_max_val,
                p_charge_max=p_chg,
                p_discharge_max=p_disch,
                eta_charge=max(1e-6, eta_c),
                eta_discharge=max(1e-6, eta_d),
                e_initial=soc0,
                tau=horizon,
                gamma=gamma,
                max_iter=max_iter,
                n_agents=n_gens,
            )
        elif ref.element_type == RENEWABLE:
            # Availability timeseries is per-unit in PyPSA; scale by p_nom.
            ts = _lookup_ts(scenario, ref)
            if ts is None:
                p_max_vec = np.full(horizon, p_nom, dtype=float)
            else:
                p_max_vec = np.asarray(ts.reindex(time_index), dtype=float) * p_nom
            participant = create_deed_admm_renewable_participant(
                finish_callback,
                p_max_timeseries=p_max_vec,
                d_i=d_i_gen,
                gamma=gamma,
                max_iter=max_iter,
                n_agents=n_gens,
            )
        else:
            p_min_pu = float(statics.get("p_min_pu", 0.0))
            p_min = max(0.0, p_min_pu * p_nom)
            participant = create_deed_admm_thermal_participant(
                finish_callback,
                p_min=p_min,
                p_max=max(p_min, p_nom),
                marginal_cost=cost,
                d_i=d_i_gen,
                gamma=gamma,
                max_iter=max_iter,
                n_agents=n_gens,
            )

        opt_role = DistributedOptimizationRole(participant)
        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for DEED-ADMM scenario.")

    # ------------------------------------------------------------------
    # Load agents
    # ------------------------------------------------------------------
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])

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
        demand_target=target_series,
        balance_label="DEED-ADMM",
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

    # Fully-connected peer topology across all generator agents.
    topology = complete_topology(len(gen_agents))
    auto_assign(topology, gen_agents)

    # ------------------------------------------------------------------
    # Recordings
    # ------------------------------------------------------------------
    install_standard_recordings(world, behavior)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        stacked_title="Stacked power (DEED-ADMM)",
        cost_title="Cost per timestep (DEED-ADMM)",
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
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


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="deed_admm",
        extra_args=_add_deed_admm_args,
        extra_kwargs=("gamma", "max_iter"),
        lossless_only=True,
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
