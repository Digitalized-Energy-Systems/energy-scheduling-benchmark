"""Distributed economic-dispatch exact-diffusion scenario.

Same wiring as :mod:`.diffusion`, but runs the bias-corrected
:class:`~distributed_resource_optimization.ExactDiffusionAlgorithm`
(adapt-correct-combine, Ces et al. 2025 Sec. 3.4) instead of classical
diffusion. Requires a degree-regular communication topology, which the
complete topology used below satisfies.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. The leader starts a single exact-diffusion run once on the full
   load *timeseries* (target demand vector).
3. After exact diffusion finishes, the leader applies each generator's
   computed per-timestep power set-points as the simulation clock advances.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    DiffusionMessage,
    LinearCostEconomicDispatchDiffusionActor,
    ReservoirStorageDiffusionActor,
    create_exact_diffusion_participant,
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
    OptimizationFinishedInfo as ExactDiffusionFinishedInfo,
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
    name_base: str = "exact_diffusion",
    simulate_days: int = 3,
    weight_rule: str = "averaging",
    max_iter: int = 6000,
    tol: float = 1e-3,
    balance_tol: float = 0.01,
    strict: bool = False,
) -> None:
    """Run the exact-diffusion benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    weight_rule:
        Combination-weight rule for the combine step -- one of
        ``"averaging"``, ``"relative_degree"``, ``"mean_metropolis"``,
        ``"hastings"``. Defaults to ``"averaging"`` (rule #1 in Ces et al.
        2025 Table I's numbering), which their results section finds
        best-performing.
    max_iter:
        Maximum exact-diffusion iterations per generator.
    tol:
        Per-round λ-change convergence tolerance.
    balance_tol:
        Max allowed per-timestep |generation - demand| / demand before the
        finished schedule is flagged as not converged.
    strict:
        If true, raise instead of writing outputs when the power-balance
        check fails.
    """
    require_lossless_transport(loss_percent, "Exact Diffusion")

    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    world, behavior = build_world(scenario, delay_s=delay_s, loss_percent=loss_percent)

    # ------------------------------------------------------------------
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for exact-diffusion scenario.")

    demand = build_demand_horizon(
        behavior, scenario, load_refs, simulate_days=simulate_days
    )
    time_index, horizon = demand.time_index, demand.horizon
    target_series = demand.target_series
    time_to_index = demand.time_to_index

    # This gets populated as each generator exact-diffusion participant finishes.
    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=ExactDiffusionFinishedInfo,
        algorithm_label="ExactDiffusion",
        schedule_attr="actor.P",
    )

    # -- Generator agents --
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    # --- Per-generator epsilon (capacity-scaled) ---
    # See capacity_scaled_epsilon's docstring for the rationale.
    eps = capacity_scaled_epsilon(gen_refs, statics_by_ref)

    # --- Stability-derived gradient step ---
    # Ces et al. 2025 tune the feedback gain ε offline (genetic algorithm,
    # Sec. 3.5) because their agents cannot see the system. This setup code
    # can: each actor's price response has slope p_nom/target_band, so the
    # dual-ascent loop (whose combine step averages the n gradients) is
    # stable iff ε < 2n/Σ(p_nom_i/band). Take a quarter of that bound. A
    # fixed ε that converges on the toy network (~100 MW) oscillates without
    # ever balancing on GW-scale networks, where the aggregate slope is four
    # orders of magnitude steeper.
    total_p_nom = sum(float(statics_by_ref[ref].get("p_nom", 0.0)) for ref in gen_refs)
    # Half the classical-diffusion step: the correction stage acts like a
    # momentum term, roughly halving the stable step range. Measured on
    # base_s_5_elec_2019: at the classical step exact diffusion settles into
    # a permanent limit cycle (1.4% energy imbalance, 20% worst hour, even
    # after 8000 iterations); at half the step it converges to a fully
    # balanced dispatch.
    grad_step = 0.25 * n_gens * eps.target_band / max(total_p_nom, 1.0)

    # Warm-start λ at the mean marginal cost — Ces et al. 2025 initialise the
    # incremental cost from the cost coefficients at the initial dispatch
    # (eqs. 23/24) rather than from an arbitrary constant, which cuts the
    # approach phase of the iteration considerably on networks whose clearing
    # price is far from any fixed default.
    initial_lam = float(np.mean(eps.costs_all)) if eps.costs_all else 10.0

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}
    for ref in gen_refs:
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            sp = StorageParams.from_statics(statics, p_nom)
            actor = ReservoirStorageDiffusionActor(
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
                n_guess=n_gens,
            )
        else:
            p_max_vec = build_p_max_vec(scenario, ref, statics, time_index, horizon)
            p_min = 0.0
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                p_min = max(0.0, p_min_pu * p_nom)
            actor = LinearCostEconomicDispatchDiffusionActor(
                cost=cost,
                p_max=p_max_vec,
                epsilon=eps.eps_by_aid[ref.component_id],
                p_min=p_min,
                n_guess=n_gens,
            )
        participant = create_exact_diffusion_participant(
            finish_callback=finish_callback,
            diffusion_actor=actor,
            initial_lam=initial_lam,
            max_iter=max_iter,
            epsilon=grad_step,
            tol=tol,
            horizon=horizon,
            weight_rule=weight_rule,
        )

        opt_role = DistributedOptimizationRole(participant)

        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for exact-diffusion scenario.")

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        # Kick off exact diffusion exactly once on the full time series.
        # The diffusion actor's p_max is also vectorised, so it constrains
        # each timestep independently.
        return DiffusionMessage(
            phi=np.full(len(target_series), initial_lam),
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
        finished_message_type=ExactDiffusionFinishedInfo,
        build_start_message=build_start_message,
        demand_target=target_series,
        balance_label="Exact Diffusion",
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

    # Fully-connected exact-diffusion topology across all generator agents.
    # Exact Diffusion requires a degree-regular graph (all nodes share the
    # same degree); the complete topology satisfies this trivially.
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


def _add_exact_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weight-rule",
        type=str,
        default="averaging",
        choices=["averaging", "relative_degree", "mean_metropolis", "hastings"],
        help="Combination-weight rule for the combine step (default: averaging, "
        "the rule Ces et al. 2025's results find best-performing).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=6000,
        help="Maximum exact-diffusion iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Per-round λ-change convergence tolerance.",
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="exact_diffusion",
        extra_args=_add_exact_diffusion_args,
        extra_kwargs=("weight_rule", "max_iter", "tol"),
        lossless_only=True,
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
