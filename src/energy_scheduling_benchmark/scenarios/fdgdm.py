"""Distributed economic-dispatch FDGDM scenario.

Implements the Fast Distributed Gradient Descent Method (FDGDM) from:
    Bai et al. (2022) "Fast distributed gradient descent method for economic
    dispatch of microgrids via upper bounds of second derivatives",
    Energy Reports 8, 1051-1060.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. Renewables are pre-scheduled at their available capacity.  FDGDM runs
   on the thermals.  Storage (Bai et al. have none) stays outside FDGDM but
   is price-driven: after each FDGDM run the leader takes the converged
   incremental cost λ*(t), gives every storage unit the exact best response
   of the shared storage model (decision Q15), and reruns FDGDM on the
   remaining demand — a few damped outer rounds.
   FDGDM is run on thermals only for the residual demand.
3. The epsilon (curvature bound) for thermals is computed from the cost
   range to prevent first-step overshoot of the gradient update.
4. After FDGDM finishes, the leader applies all schedules as the
   simulation clock advances.

Scope (decision Q24, kept as in the paper): FDGDM conserves ΣP in every step
(Eq. 10) and handles unit limits only by clipping (Eq. 16), which loses power
whenever a limit binds; renewables run at full availability (MPPT) and are
never curtailed. It works only with strictly convex costs, no binding limits
at the optimum and no renewable surplus. Outside that scope (linear costs
without the proxy, quadratic costs at the limits, ``lower_saxony_2030``'s 2030
renewable surplus) it fails structurally — see ``docs/deviations.md``, FDGDM.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    FDGDMActor,
    LinearCostEconomicDispatchFDGDMActor,
    create_fdgdm_participant,
    create_fdgdm_start,
)
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
)
from distributed_resource_optimization.misc.storage import StorageModel
from mango import (
    RoleAgent,
    agent_composed_of,
)

from energy_scheduling_benchmark import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as FDGDMFinishedInfo,
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

logger = logging.getLogger(__name__)

#: Outer rounds of the price-driven storage loop (FDGDM runs after the first).
STORAGE_ROUNDS = 5
#: Damping of each storage update (a convex mix of feasible schedules stays feasible).
STORAGE_DAMPING = 0.5
#: Stop the outer loop once no storage schedule would move more than this (MW).
STORAGE_TOL_MW = 0.01

#: Eq. 24 accuracy of Bai et al. 2022 relative to unit size: the paper stops at
#: ε = 0.01 for units with 0–45 kW boxes (§4). Each thermal agent gets
#: ``tol = PAPER_TOL_RATIO * p_nom`` on its per-hour step.
PAPER_TOL_RATIO = 0.01 / 45.0


# ---------------------------------------------------------------------------
# FDGDM-specific pre-scheduling helpers
# ---------------------------------------------------------------------------


def _capacity_proportional_allocation(
    adjusted_target: np.ndarray,
    p_max_vecs: list[np.ndarray],
    p_min_vecs: list[np.ndarray],
) -> list[np.ndarray]:
    """Per-agent demand-feasible initial allocations proportional to slack capacity.

    Computes ``alloc_i = p_min_i + residual * (p_max_i - p_min_i) / total_slack``
    where ``residual = adjusted_target - Σ p_min_i`` and
    ``total_slack = Σ (p_max_i - p_min_i)``.

    Guarantees ``Σ_i alloc_i[t] = adjusted_target[t]`` exactly and
    ``p_min_i[t] ≤ alloc_i[t] ≤ p_max_i[t]`` for all i and t, provided
    ``Σ p_min_i[t] ≤ adjusted_target[t] ≤ Σ p_max_i[t]``.
    """
    total_p_min = np.sum(p_min_vecs, axis=0)
    total_p_max = np.sum(p_max_vecs, axis=0)
    total_slack = total_p_max - total_p_min
    residual = np.maximum(adjusted_target - total_p_min, 0.0)

    infeasible = adjusted_target > total_p_max + 1e-6
    if np.any(infeasible):
        shortage = (adjusted_target - total_p_max)[infeasible]
        logger.warning(
            "Thermal capacity insufficient at %d timestep(s); max shortage %.1f MW. "
            "FDGDM will conserve a total below demand for those hours.",
            int(infeasible.sum()),
            float(shortage.max()),
        )

    allocs = []
    for pmax, pmin in zip(p_max_vecs, p_min_vecs):
        slack_i = pmax - pmin
        with np.errstate(divide="ignore", invalid="ignore"):  # masked by np.where
            share = np.where(
                total_slack > 0, pmin + residual * slack_i / total_slack, pmin
            )
        allocs.append(np.clip(share, pmin, pmax))
    return allocs


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
    name_base: str = "fdgdm",
    simulate_days: int = 3,
    max_iter: int = 300,
    tol_ratio: float | None = PAPER_TOL_RATIO,
    balance_tol: float = 0.01,
    strict: bool = False,
    slack: SlackSpec | None = None,
    cost_model: str = "linear",
) -> None:
    """Run the FDGDM benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    max_iter:
        Maximum FDGDM iterations per thermal generator (backstop).
    tol_ratio:
        Eq. 24 early stopping: each thermal agent stops contributing once its
        largest per-hour step is below ``tol_ratio * p_nom``, and the run ends
        when every agent and its neighbours agree (see the dro FDGDM module).
        Default: the paper's accuracy relative to unit size
        (:data:`PAPER_TOL_RATIO`). ``None`` or ``0`` disables it (always
        ``max_iter`` rounds).
    balance_tol:
        Max allowed per-timestep |generation - demand| / demand before the
        finished schedule is flagged as not converged. Note FDGDM's weight
        matrix conserves the pre-schedule power balance by construction, so
        this check is a weaker signal here than for the other algorithms —
        see the module-level review notes.
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
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for FDGDM scenario.")

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
        finished_message_type=FDGDMFinishedInfo,
        algorithm_label="FDGDM",
        schedule_attr="actor.P",
    )

    # -- Generator classification --
    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    cost_quad_by_aid = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    group_map = build_group_map(gen_refs, statics_by_ref=statics_by_ref)

    thermal_refs = [r for r in gen_refs if r.element_type == THERMAL]
    nonthermal_refs = [r for r in gen_refs if r.element_type != THERMAL]
    generator_aids = [ref.component_id for ref in gen_refs]

    # ------------------------------------------------------------------
    # Pre-schedule non-thermal generators and compute residual thermal demand.
    #
    # Pass 1 — Renewables at available capacity.
    # Necessary because including zero-cost generators in FDGDM drives thermals
    # to 0 on the first iteration (large w_abs = 1/(n·ε) × cost_diff).
    #
    # Storage takes part in FDGDM itself (shared storage model, Q15).
    # ------------------------------------------------------------------
    renewable_refs = [r for r in nonthermal_refs if r.element_type == RENEWABLE]
    storage_refs = [r for r in nonthermal_refs if r.element_type == STORAGE]
    renewable_gen_ts = np.zeros(horizon, dtype=float)

    cost_by_aid: dict[str, float] = {}
    for ref in renewable_refs:
        statics = statics_by_ref[ref]
        cost_by_aid[ref.component_id] = float(statics.get("marginal_cost", 0.0))
        sched = build_p_max_vec(scenario, ref, statics, time_index, horizon)
        schedule_by_aid[ref.component_id] = sched
        renewable_gen_ts += sched

    # Residual load after renewables: positive = deficit, negative = surplus.
    net_load_ts = (target_series - renewable_gen_ts).astype(float)

    adjusted_target = np.maximum(net_load_ts, 0.0)

    # Per-thermal (p_min, p_max) in MW, reused below for agent creation and
    # initial allocation.
    thermal_bounds = [
        build_unit_bounds(scenario, ref, statics_by_ref[ref], time_index, horizon)
        for ref in thermal_refs
    ]
    thermal_p_min_vecs = [p_min for p_min, _ in thermal_bounds]
    thermal_p_max_vecs = [p_max for _, p_max in thermal_bounds]

    # FDGDM requires ≥ 2 participants: with 0 neighbours the finish callback
    # is never called and the simulation hangs.  When only one thermal exists,
    # pre-schedule it directly (min of residual demand and its own p_max) and
    # treat it as a nonthermal for agent setup so n_fdgdm_participants = 0.
    if len(thermal_refs) == 1:
        _sched_1 = np.clip(
            adjusted_target, thermal_p_min_vecs[0], thermal_p_max_vecs[0]
        )
        schedule_by_aid[thermal_refs[0].component_id] = _sched_1
        cost_by_aid[thermal_refs[0].component_id] = float(
            statics_by_ref[thermal_refs[0]].get("marginal_cost", 0.0)
        )
        adjusted_target = np.maximum(adjusted_target - _sched_1, 0.0)
        nonthermal_refs = nonthermal_refs + thermal_refs
        thermal_refs = []
        thermal_p_min_vecs = []
        thermal_p_max_vecs = []

    # ------------------------------------------------------------------
    # Mask p_max to 0 at timesteps where thermals have no residual demand.
    # At fully-renewable hours (adjusted_target ≈ 0) the linear gradient
    # ε·0 + c_i = c_i is non-zero, so without masking FDGDM would push
    # cheap generators above zero (spurious over-generation).  Setting
    # p_max[t] = p_min[t] forces project() to clip any update back to the
    # unit's minimum (zero unless it is must-run).
    # ------------------------------------------------------------------
    thermal_p_max_unmasked = [pmax.copy() for pmax in thermal_p_max_vecs]
    zero_demand_mask = adjusted_target < 1.0
    if zero_demand_mask.any():
        thermal_p_max_vecs = [pmax.copy() for pmax in thermal_p_max_vecs]
        for pmin, pmax in zip(thermal_p_min_vecs, thermal_p_max_vecs):
            pmax[zero_demand_mask] = pmin[zero_demand_mask]

    # ------------------------------------------------------------------
    # Per-agent demand-feasible initial allocations proportional to p_nom.
    # Necessary because equal-split initial_p clips small generators (e.g.
    # geothermal, oil) at their p_max before the first iteration, permanently
    # breaking the zero-row-sum conservation invariant.
    # ------------------------------------------------------------------
    n_thermals = len(thermal_refs)
    if n_thermals >= 1:
        fdgdm_initial_p_by_agent = _capacity_proportional_allocation(
            adjusted_target, thermal_p_max_vecs, thermal_p_min_vecs
        )
    else:
        fdgdm_initial_p_by_agent = []

    # ------------------------------------------------------------------
    # Per-generator epsilon: ε_i = base_ε × Σp_nom / p_nom_i
    #
    # Cancels the p_nom dependence in the gradient at the proportional
    # initial allocation so first-step changes are bounded by cost_diff
    # alone, regardless of capacity heterogeneity.
    # ------------------------------------------------------------------
    if n_thermals >= 2:
        thermal_costs = [
            float(statics_by_ref[r].get("marginal_cost", 0.0)) for r in thermal_refs
        ]
        cost_diff = max(thermal_costs) - min(thermal_costs)
        thermal_p_nom_list = [
            float(statics_by_ref[r].get("p_nom", 0.0)) for r in thermal_refs
        ]
        total_p_nom = sum(thermal_p_nom_list)
        active_steps = adjusted_target[~zero_demand_mask]
        min_active = float(active_steps.min()) if len(active_steps) > 0 else 1.0
        base_epsilon = max(0.1, cost_diff / min_active)
        epsilon_by_agent = [
            base_epsilon * total_p_nom / max(p_nom, 1.0) for p_nom in thermal_p_nom_list
        ]
    else:
        epsilon_by_agent = [0.1] * max(n_thermals, 1)

    def _storage_epsilon(p_nom: float) -> float:
        # Same capacity scaling as the thermals' proxy curvature.
        if n_thermals >= 2:
            return base_epsilon * total_p_nom / max(p_nom, 1.0)
        return 0.1

    # Dummy kickoff data — each actor uses its own initial_schedule instead.
    fdgdm_initial_p = adjusted_target / max(n_thermals, 1)

    # -- Thermal generator agents (FDGDM participants) --
    pgraph = participant_graph(
        scenario, topology, thermal_refs, attached=[*load_refs, *nonthermal_refs]
    )
    lt = lt.for_hops(pgraph.hops_from(load_refs[0]))
    thermal_gen_agents: list[RoleAgent] = []
    thermal_actors: list[LinearCostEconomicDispatchFDGDMActor] = []
    for i, ref in enumerate(thermal_refs):
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_max_vec = thermal_p_max_vecs[i]

        c2 = cost_quad_by_aid.get(ref.component_id)
        if c2:
            # Quadratic costs (Q2/Q17): the gradient uses the true curvature
            # 2c₂; the weights use u = max(2c₂, proxy). Bai et al. only need an
            # *upper bound* of F'' — u = 2c₂ alone overshoots on the benchmark's
            # wide cost spreads (toy: units pinned at their limits after 3 rounds).
            actor = LinearCostEconomicDispatchFDGDMActor(
                cost=cost,
                p_max=p_max_vec,
                epsilon=2.0 * c2,
                curvature_upper_bound=max(2.0 * c2, epsilon_by_agent[i]),
                p_min=thermal_p_min_vecs[i],
                initial_schedule=fdgdm_initial_p_by_agent[i].copy(),
            )
        else:
            actor = LinearCostEconomicDispatchFDGDMActor(
                cost=cost,
                p_max=p_max_vec,
                epsilon=epsilon_by_agent[i],
                p_min=thermal_p_min_vecs[i],
                initial_schedule=fdgdm_initial_p_by_agent[i].copy(),
            )
        p_nom = float(statics.get("p_nom", 0.0))
        participant = create_fdgdm_participant(
            finish_callback=finish_callback,
            fdgdm_actor=actor,
            max_iter=max_iter,
            round_timeout=lt.round_timeout,
            tol=tol_ratio * p_nom if tol_ratio else None,
            diameter=pgraph.diameter,
        )
        opt_role = DistributedOptimizationRole(participant)
        agent = generator_agent(opt_role)
        register_component_agent(world, agent, ref)
        thermal_gen_agents.append(agent)
        thermal_actors.append(actor)

    # -- Storage units: outside FDGDM, price-driven by the outer loop below --
    storage_units: list[tuple[str, StorageModel, float]] = []
    for ref in storage_refs:
        statics = statics_by_ref[ref]
        cost = max(0.0, float(statics.get("marginal_cost", 0.0)))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))
        sp = StorageParams.from_statics(statics, p_nom)
        model = StorageModel(
            e_max=sp.e_max,
            p_charge_max=sp.p_charge_max,
            p_discharge_max=sp.p_discharge_max,
            eta_charge=sp.eta_charge,
            eta_discharge=sp.eta_discharge,
            e_initial=sp.e_initial,
            discharge_cost=cost,
        )
        storage_units.append((ref.component_id, model, _storage_epsilon(p_nom)))
        schedule_by_aid[ref.component_id] = np.zeros(horizon)

    # -- Renewable and storage agents (registered for recording/regulation; not in FDGDM) --
    for ref in nonthermal_refs:
        noop_participant = create_fdgdm_participant(
            finish_callback=lambda _a, _c: None,
            fdgdm_actor=FDGDMActor(),
            max_iter=0,
        )
        opt_role = DistributedOptimizationRole(noop_participant)
        agent = generator_agent(opt_role)
        register_component_agent(world, agent, ref)

    gen_agents = thermal_gen_agents  # FDGDM participants

    if not generator_aids:
        raise RuntimeError("No generator agents found for FDGDM scenario.")

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    register_component_agent(world, leader_agent, load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        # Kick off FDGDM on the thermal sub-problem.  The kickoff data is an
        # equal split of the residual demand, but each thermal actor overrides
        # it with its own pre-computed initial_schedule on the first project().
        return create_fdgdm_start(data=fdgdm_initial_p)

    outer = {"round": 0, "p_max": [pmax.copy() for pmax in thermal_p_max_vecs]}

    def _incremental_cost() -> np.ndarray:
        """λ*(t) from the thermals' final schedules: the mean gradient of the
        units strictly inside their limits, else the highest gradient of a
        running unit (scarcity), else 0 (renewable surplus)."""
        lam = np.zeros(horizon)
        P = [
            np.asarray(schedule_by_aid[r.component_id], dtype=float)
            for r in thermal_refs
        ]
        grads = [a.gradient(p, None) for a, p in zip(thermal_actors, P)]
        for t in range(horizon):
            interior = [
                g[t]
                for g, p, lo, hi in zip(grads, P, thermal_p_min_vecs, outer["p_max"])
                if lo[t] + 1e-6 < p[t] < hi[t] - 1e-6
            ]
            if interior:
                lam[t] = float(np.mean(interior))
            else:
                running = [g[t] for g, p in zip(grads, P) if p[t] > 1e-6]
                lam[t] = max(running) if running else 0.0
        return lam

    def _next_storage_round() -> Any:
        """Update storage from λ*; return the next FDGDM start, or None to stop."""
        if outer["round"] >= STORAGE_ROUNDS:
            return None
        lam = _incremental_cost()
        proposal: dict[str, np.ndarray] = {}
        change = 0.0
        for aid, model, eps in storage_units:
            response, _ = model.price_response(lam, eps)
            old = schedule_by_aid[aid]
            new = (1.0 - STORAGE_DAMPING) * old + STORAGE_DAMPING * response
            proposal[aid] = new
            change = max(change, float(np.max(np.abs(new - old))))
        if change < STORAGE_TOL_MW:
            return None  # the thermals were dispatched against the current storage
        outer["round"] += 1
        schedule_by_aid.update(proposal)
        residual = np.maximum(net_load_ts - sum(proposal.values()), 0.0)
        p_max_k = [pmax.copy() for pmax in thermal_p_max_unmasked]
        mask = residual < 1.0
        for pmin, pmax in zip(thermal_p_min_vecs, p_max_k):
            pmax[mask] = pmin[mask]
        alloc = _capacity_proportional_allocation(residual, p_max_k, thermal_p_min_vecs)
        for actor, pmax, a in zip(thermal_actors, p_max_k, alloc):
            actor.p_max = pmax
            actor._p_max = np.asarray(pmax, dtype=float)
            actor.initial_schedule = a.copy()
        outer["p_max"] = p_max_k
        logger.info(
            "FDGDM storage round %d: max storage change %.3f MW", outer["round"], change
        )
        return create_fdgdm_start(data=residual / max(n_thermals, 1))

    aggregator = PowerLoadAggregator(
        behavior=behavior,
        number_loads=len(load_refs),
        generator_aids=generator_aids,
        trigger=gen_agents[0].addr if gen_agents else None,
        time_to_index=time_to_index,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=FDGDMFinishedInfo,
        build_start_message=build_start_message,
        generator_addrs={a.aid: a.addr for a in gen_agents},
        load_timeout=lt.load_timeout,
        request_timeout=lt.request_timeout,
        result_poll_s=lt.result_poll_s,
        result_retries=lt.result_retries,
        n_finished_required=len(thermal_refs),
        on_run_finished=_next_storage_round
        if storage_units and n_thermals >= 2
        else None,
        demand_target=target_series,
        balance_label="FDGDM",
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

    install_topology(world, pgraph, gen_agents)

    # -- Simulate --
    # Run until just past the LAST snapshot inside the window (index
    # simulate_days*24-1). Since _clip_scenario trims the timeseries before
    # PyPSABehavior schedules its ticks, no snapshot fires after that point
    # anyway; stopping here just avoids simulating an empty final hour.
    sim_end_s = (simulate_days * 24 - 1) * 3600.0 + 1.0
    async with world:
        await run_simulation(world, sim_end_s)

    # -- Output --
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        cost_quad_by_aid=cost_quad_by_aid,
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
        slack=slack,
        group_map=group_map,
        net=scenario.net,
        label="FDGDM",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_fdgdm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum FDGDM iterations per thermal generator (backstop).",
    )
    parser.add_argument(
        "--tol-ratio",
        type=float,
        default=PAPER_TOL_RATIO,
        help=(
            "Eq. 24 early stop: per-hour step tolerance as a fraction of each "
            "unit's p_nom (default: the paper's 0.01/45); 0 disables it."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="fdgdm",
        extra_args=_add_fdgdm_args,
        extra_kwargs=("max_iter", "tol_ratio"),
        argv=argv,
    )


if __name__ == "__main__":
    main()
