"""Distributed economic-dispatch consensus scenario.

Python port of ``EnergySchedulingBenchmark.jl/scenario/consensus_scenario.jl``.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. The leader starts a single averaging-consensus run once on the full
   load *timeseries* (target demand vector).
3. After consensus finishes, the leader applies each generator's computed
   per-timestep power set-points as the simulation clock advances.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    AveragingConsensusMessage,
    LinearCostEconomicDispatchConsensusActor,
    ReservoirStorageConsensusActor,
    create_averaging_consensus_participant,
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
    STORAGE,
    THERMAL,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as ConsensusFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
    PowerLoadMonitoring,
    ScenarioData,
    StorageParams,
    _clip_scenario,
    build_demand_horizon,
    build_p_max_vec,
    build_toy_network,
    build_world,
    capacity_scaled_epsilon,
    collect_generator_refs,
    install_standard_recordings,
    make_finish_callback,
    run_scenario_main,
    write_scenario_outputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.0,
    name_base: str = "consensus",
    simulate_days: int = 3,
    max_iter: int = 500,
    alpha: float = 0.2,
    balance_tol: float = 0.01,
    strict: bool = False,
) -> None:
    """Run the consensus benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    max_iter:
        Maximum averaging-consensus iterations per generator.
    alpha:
        Follower neighbour-averaging step size.
    balance_tol:
        Max allowed per-timestep |generation - demand| / demand before the
        finished schedule is flagged as not converged.
    strict:
        If true, raise instead of writing outputs when the power-balance
        check fails.
    """
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(scenario, delay_s=delay_s, loss_percent=loss_percent)

    # ------------------------------------------------------------------
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for consensus scenario.")

    demand = build_demand_horizon(
        behavior, scenario, load_refs, simulate_days=simulate_days
    )
    time_index, horizon = demand.time_index, demand.horizon
    target_series = demand.target_series
    time_to_index = demand.time_to_index

    # This gets populated as each generator consensus participant finishes.
    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=ConsensusFinishedInfo,
        algorithm_label="Consensus",
        schedule_attr="actor.P",
    )

    # -- Generator agents --
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    generator_aids = [ref.component_id for ref in gen_refs]

    # --- Per-generator epsilon (capacity-scaled) ---
    # See capacity_scaled_epsilon's docstring for the rationale.
    eps = capacity_scaled_epsilon(gen_refs, statics_by_ref)

    # --- Leader gain (demand-scaled) ---
    # The leader's price update is `lam += leader_gain * ΔP` (Jian et al.
    # 2020, eq. 22), where ΔP is the *absolute* system-wide power imbalance
    # (MW). A fixed leader_gain conflates two unrelated scales: ΔP grows with
    # total demand (tens to tens-of-thousands of MW on real networks), while
    # a sensible λ correction should stay within a few multiples of the cost
    # spread (a handful to ~100 €/MWh). A gain tuned for a toy network's
    # ~100 MW demand becomes wildly too large at ~60,000 MW: each correction
    # overshoots the cost spread by orders of magnitude, so λ oscillates
    # chaotically and never settles, no matter how many iterations run.
    # Scaling the gain by cost_range / demand keeps one demand-scale
    # imbalance mapped to roughly one cost-spread-scale price correction,
    # independent of network size.
    mean_target = float(np.mean(target_series)) if horizon > 0 else 0.0
    leader_gain = max(eps.cost_range, 1.0) / max(mean_target, 1.0)

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}
    for gen_idx, ref in enumerate(gen_refs):
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            sp = StorageParams.from_statics(statics, p_nom)
            actor = ReservoirStorageConsensusActor(
                e_max=sp.e_max,
                p_charge_max=sp.p_charge_max,
                p_discharge_max=sp.p_discharge_max,
                eta_charge=sp.eta_charge,
                eta_discharge=sp.eta_discharge,
                e_initial=sp.e_initial,
                e_final=sp.e_initial,
                soc_min=0.0,
                soc_max=1.0,
                charge_cost=max(0.0, cost),
                discharge_cost=max(0.0, cost),
                epsilon=eps.target_band / max(p_nom, 1.0),
            )
        else:
            p_max_vec = build_p_max_vec(scenario, ref, statics, time_index, horizon)
            p_min = 0.0
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                p_min = max(0.0, p_min_pu * p_nom)
            actor = LinearCostEconomicDispatchConsensusActor(
                cost=cost,
                p_max=p_max_vec,
                p_min=p_min,
                epsilon=eps.eps_by_aid[ref.component_id],
            )
        # The first generator agent is the leader (Jian et al. 2020, eq. 22);
        # it pins λ toward the real system-wide power imbalance ΔP, while all
        # other generators are followers doing pure neighbour averaging.
        participant = create_averaging_consensus_participant(
            finish_callback=finish_callback,
            consensus_actor=actor,
            max_iter=max_iter,
            alpha=alpha,
            is_leader=(gen_idx == 0),
            leader_gain=leader_gain,
        )

        opt_role = DistributedOptimizationRole(participant)

        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for consensus scenario.")

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        # Kick off consensus exactly once on the full time series.
        # The consensus actor's p_max is also vectorised, so it constrains
        # each timestep independently.
        return AveragingConsensusMessage(
            lam=np.full(len(target_series), 10.0),
            k=0,
            data=np.asarray(target_series, dtype=float),
            initial=True,
        )

    aggregator = PowerLoadAggregator(
        behavior=behavior,
        number_loads=len(load_refs),
        generator_aids=generator_aids,
        trigger=gen_agents[0].addr,
        time_to_index=time_to_index,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=ConsensusFinishedInfo,
        build_start_message=build_start_message,
        demand_target=target_series,
        balance_label="Consensus",
        balance_tol=balance_tol,
    )
    leader_agent.add_role(aggregator)
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    # add all other load agents
    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    # Fully-connected consensus topology across all generator agents.
    topology = complete_topology(len(gen_agents))
    auto_assign(topology, gen_agents)

    # -- Recordings --
    install_standard_recordings(world, behavior)

    # -- Simulate --
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # -- Output --
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_consensus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum averaging-consensus iterations per generator.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Follower neighbour-averaging step size.",
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="consensus",
        extra_args=_add_consensus_args,
        extra_kwargs=("max_iter", "alpha"),
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
