"""Distributed economic-dispatch FDGDM scenario.

Implements the Fast Distributed Gradient Descent Method (FDGDM) from:
    Bai et al. (2022) "Fast distributed gradient descent method for economic
    dispatch of microgrids via upper bounds of second derivatives",
    Energy Reports 8, 1051-1060.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. Renewables are pre-scheduled at their available capacity; storage is
   pre-scheduled with an energy-constrained SOC-aware forward pass (charges
   from renewable surplus, discharges into deficit, tracks reservoir level).
   FDGDM is run on thermals only for the residual demand.
3. The epsilon (curvature bound) for thermals is computed from the cost
   range to prevent first-step overshoot of the gradient update.
4. After FDGDM finishes, the leader applies all schedules as the
   simulation clock advances.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    LinearCostEconomicDispatchFDGDMActor,
    NoFDGDMActor,
    create_fdgdm_participant,
    create_fdgdm_start,
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
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as FDGDMFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
    PowerLoadMonitoring,
    ScenarioData,
    _clip_scenario,
    _lookup_ts,
    build_behavior,
    build_group_map,
    build_scenario_argparser,
    build_toy_network,
    filter_and_cache_statics,
    make_finish_callback,
    require_lossless_transport,
    resolve_scenario,
    write_scenario_outputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FDGDM-finished callback
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
):
    """Build the ``(algorithm, carrier) -> None`` hook that stores the schedule."""
    return make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=FDGDMFinishedInfo,
        algorithm_label="FDGDM",
        schedule_attr="actor.P",
    )


# ---------------------------------------------------------------------------
# Scenario entry point
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
        share = np.where(total_slack > 0, pmin + residual * slack_i / total_slack, pmin)
        allocs.append(np.clip(share, pmin, pmax))
    return allocs


def _schedule_storage_soc(
    net_load_ts: np.ndarray,
    p_nom: float,
    p_min_pu: float,
    max_hours: float,
    soc_initial_pu: float,
    efficiency_store: float,
    efficiency_dispatch: float,
    inflow_mwh: np.ndarray,
) -> np.ndarray:
    """Energy-constrained storage dispatch over a scheduling horizon.

    Charges during surplus-renewable hours (net_load < 0) and discharges
    during deficit hours (net_load > 0), tracking state-of-charge each step.

    .. note::
        Assumes **1-hour timesteps**.  SOC is in MWh and power limits are in MW;
        the comparison ``dispatch = min(p_nom, net_load, soc × η)`` is only
        numerically valid when each interval is exactly 1 h.  Sub-hourly
        networks will produce incorrect SOC accounting.

    :param net_load_ts: Residual load after renewables (MW, per hour).
                        Positive = demand exceeds supply; negative = surplus.
    :param p_nom: Discharge power limit (MW).
    :param p_min_pu: Charge power limit as fraction of p_nom (negative, e.g. -1.0).
    :param max_hours: Energy capacity in hours at full power (MWh = p_nom × max_hours).
    :param soc_initial_pu: Initial state-of-charge as fraction of max energy.
    :param efficiency_store: Round-trip charge efficiency (≤ 1).
    :param efficiency_dispatch: Round-trip discharge efficiency (≤ 1).
    :param inflow_mwh: Natural energy inflow each hour (MWh), e.g. river inflow.
    :returns: Net dispatch in MW per hour.  Positive = discharge; negative = charge.
    """
    max_energy = max_hours * p_nom
    p_charge_max = (-p_min_pu) * p_nom
    soc = soc_initial_pu * max_energy
    net = np.zeros(len(net_load_ts), dtype=float)

    for t in range(len(net_load_ts)):
        soc = min(soc + float(inflow_mwh[t]), max_energy)

        if net_load_ts[t] > 0.0:
            deliverable = soc * efficiency_dispatch
            dispatch = min(p_nom, net_load_ts[t], deliverable)
            if dispatch > 0.0:
                soc -= dispatch / efficiency_dispatch
                net[t] = dispatch
        else:
            surplus = -net_load_ts[t]
            storable = (
                (max_energy - soc) / efficiency_store if efficiency_store > 0 else 0.0
            )
            charge = min(p_charge_max, surplus, storable)
            if charge > 0.0:
                soc += charge * efficiency_store
                net[t] = -charge

        soc = max(0.0, soc)

    return net


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.0,
    name_base: str = "fdgdm",
    simulate_days: int = 3,
) -> None:
    """Run the FDGDM benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    """
    require_lossless_transport(loss_percent, "FDGDM")

    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    behavior = build_behavior(scenario)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(
        default_delay_s=delay_s, loss_percent=loss_percent
    )
    world = create_world(
        start_time=0.0, communication_sim=com_sim, environment=environment
    )

    # ------------------------------------------------------------------
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for FDGDM scenario.")

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
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = _make_finish_callback(
        leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid
    )

    # -- Generator classification --
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    gen_refs = [gen for gen in gen_refs if "hydro" not in gen.component_id]
    # sort out devices with zero/non-finite max power/nominal power
    gen_refs, statics_by_ref = filter_and_cache_statics(behavior, gen_refs)
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
    # Pass 2 — Storage with energy-constrained SOC-aware dispatch.
    # Storage starts at state_of_charge_initial (often 0 for PHS); pre-scheduling
    # at constant max-discharge would exceed the reservoir capacity immediately.
    # Instead, we charge from renewable surplus hours and discharge during deficit
    # hours while tracking SOC each step.
    # ------------------------------------------------------------------
    renewable_refs = [r for r in nonthermal_refs if r.element_type == RENEWABLE]
    storage_refs = [r for r in nonthermal_refs if r.element_type == STORAGE]
    renewable_gen_ts = np.zeros(horizon, dtype=float)

    cost_by_aid: dict[str, float] = {}
    for ref in renewable_refs:
        statics = statics_by_ref[ref]
        cost_by_aid[ref.component_id] = float(statics.get("marginal_cost", 0.0))
        p_nom = float(statics.get("p_nom", 0.0))
        ts = _lookup_ts(scenario, ref)
        sched = (
            np.full(horizon, p_nom, dtype=float)
            if ts is None
            else np.asarray(ts.reindex(time_index), dtype=float) * p_nom
        )
        schedule_by_aid[ref.component_id] = sched
        renewable_gen_ts += sched

    # Residual load after renewables: positive = deficit, negative = surplus.
    net_load_ts = (target_series - renewable_gen_ts).astype(float)

    for ref in storage_refs:
        statics = statics_by_ref[ref]
        cost_by_aid[ref.component_id] = float(statics.get("marginal_cost", 0.0))
        p_nom = float(statics.get("p_nom", 0.0))

        # Inflow from PyPSA timeseries (e.g. reservoir hydro); falls back to
        # the static "inflow" attribute if no timeseries column is present.
        su_t = getattr(getattr(scenario, "net", None), "storage_units_t", None)
        inflow_df = getattr(su_t, "inflow", None)
        if inflow_df is not None and ref.component_id in inflow_df.columns:
            inflow_mwh = np.asarray(
                inflow_df[ref.component_id].reindex(time_index).fillna(0.0), dtype=float
            )
        else:
            inflow_mwh = np.full(
                horizon, float(statics.get("inflow", 0.0)), dtype=float
            )

        net = _schedule_storage_soc(
            net_load_ts=net_load_ts,
            p_nom=p_nom,
            p_min_pu=float(statics.get("p_min_pu", -1.0)),
            max_hours=float(statics.get("max_hours", 6.0)),
            soc_initial_pu=float(statics.get("state_of_charge_initial", 0.0)),
            efficiency_store=float(statics.get("efficiency_store", 1.0)),
            efficiency_dispatch=float(statics.get("efficiency_dispatch", 1.0)),
            inflow_mwh=inflow_mwh,
        )
        schedule_by_aid[ref.component_id] = net
        net_load_ts -= net  # update residual for subsequent storage units

    adjusted_target = np.maximum(net_load_ts, 0.0)

    # ------------------------------------------------------------------
    # Pre-compute per-thermal p_max time series and p_nom values.
    # p_max_pu from generators_t is per-unit (0–1) and must be scaled by
    # p_nom to get MW.  This block fixes a bug where the timeseries branch
    # passed raw per-unit values to LinearCostEconomicDispatchFDGDMActor.
    # The vectors are reused below for agent creation and initial allocation.
    # ------------------------------------------------------------------
    thermal_p_max_vecs: list[np.ndarray] = []
    for _ref_th in thermal_refs:
        _statics_th = statics_by_ref[_ref_th]
        _p_nom_th = float(_statics_th.get("p_nom", 0.0))
        _ts_th = _lookup_ts(scenario, _ref_th)
        thermal_p_max_vecs.append(
            np.full(horizon, _p_nom_th, dtype=float)
            if _ts_th is None
            else np.asarray(_ts_th.reindex(time_index), dtype=float) * _p_nom_th
        )

    # FDGDM requires ≥ 2 participants: with 0 neighbours the finish callback
    # is never called and the simulation hangs.  When only one thermal exists,
    # pre-schedule it directly (min of residual demand and its own p_max) and
    # treat it as a nonthermal for agent setup so n_fdgdm_participants = 0.
    if len(thermal_refs) == 1:
        _sched_1 = np.minimum(adjusted_target, thermal_p_max_vecs[0])
        schedule_by_aid[thermal_refs[0].component_id] = _sched_1
        cost_by_aid[thermal_refs[0].component_id] = float(
            statics_by_ref[thermal_refs[0]].get("marginal_cost", 0.0)
        )
        adjusted_target = np.maximum(adjusted_target - _sched_1, 0.0)
        nonthermal_refs = nonthermal_refs + thermal_refs
        thermal_refs = []
        thermal_p_max_vecs = []

    # ------------------------------------------------------------------
    # Mask p_max to 0 at timesteps where thermals have no residual demand.
    # At fully-renewable hours (adjusted_target ≈ 0) the linear gradient
    # ε·0 + c_i = c_i is non-zero, so without masking FDGDM would push
    # cheap generators above zero (spurious over-generation).  Setting
    # p_max[t] = 0 forces project() to clip any update back to zero.
    # ------------------------------------------------------------------
    zero_demand_mask = adjusted_target < 1.0
    if zero_demand_mask.any():
        thermal_p_max_vecs = [pmax.copy() for pmax in thermal_p_max_vecs]
        for pmax in thermal_p_max_vecs:
            pmax[zero_demand_mask] = 0.0

    # ------------------------------------------------------------------
    # Per-agent demand-feasible initial allocations proportional to p_nom.
    # Necessary because equal-split initial_p clips small generators (e.g.
    # geothermal, oil) at their p_max before the first iteration, permanently
    # breaking the zero-row-sum conservation invariant.
    # ------------------------------------------------------------------
    n_thermals = len(thermal_refs)
    if n_thermals >= 2:
        thermal_p_min_vecs = [np.zeros(horizon) for _ in thermal_refs]
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
        epsilon_by_agent = [0.1]

    # Dummy kickoff data — each actor uses its own initial_schedule instead.
    fdgdm_initial_p = adjusted_target / max(n_thermals, 1)

    # -- Thermal generator agents (FDGDM participants) --
    thermal_gen_agents: list[RoleAgent] = []
    for i, ref in enumerate(thermal_refs):
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_max_vec = thermal_p_max_vecs[i]

        actor = LinearCostEconomicDispatchFDGDMActor(
            cost=cost,
            p_max=p_max_vec,
            epsilon=epsilon_by_agent[i],
            initial_schedule=fdgdm_initial_p_by_agent[i].copy(),
        )
        participant = create_fdgdm_participant(
            finish_callback=finish_callback,
            fdgdm_actor=actor,
            max_iter=300,
            horizon=horizon,
        )
        opt_role = DistributedOptimizationRole(participant)
        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        thermal_gen_agents.append(agent)

    # -- Non-thermal agents (registered for recording/regulation; not in FDGDM) --
    for ref in nonthermal_refs:
        noop_participant = create_fdgdm_participant(
            finish_callback=lambda _a, _c: None,
            fdgdm_actor=NoFDGDMActor(),
            max_iter=0,
            horizon=horizon,
        )
        opt_role = DistributedOptimizationRole(noop_participant)
        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    gen_agents = thermal_gen_agents  # FDGDM topology only for thermals

    if not generator_aids:
        raise RuntimeError("No generator agents found for FDGDM scenario.")

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        # Kick off FDGDM on the thermal sub-problem.  The kickoff data is an
        # equal split of the residual demand, but each thermal actor overrides
        # it with its own pre-computed initial_schedule on the first project().
        return create_fdgdm_start(data=fdgdm_initial_p)

    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            trigger=gen_agents[0].addr if gen_agents else None,
            time_to_index=time_to_index,
            schedule_by_aid=schedule_by_aid,
            finished_message_type=FDGDMFinishedInfo,
            build_start_message=build_start_message,
            n_finished_required=len(thermal_refs),
            demand_target=target_series,
            balance_label="FDGDM",
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    if gen_agents:
        topology = complete_topology(len(gen_agents))
        auto_assign(topology, gen_agents)

    # -- Recordings --
    record_agent_having(
        world,
        "target",
        PowerLoadAggregator,
        lambda a: next(
            (r.target for r in a.roles if isinstance(r, PowerLoadAggregator)), 0.0
        ),
    )
    # "P" is summed by compute_overall_cost (scenarios/_common.py), which clips
    # negative values to 0 before costing — see its docstring for the sign
    # convention. FDGDM already clips here (below) rather than relying on that.
    record_agent_having(
        world,
        "P",
        DistributedOptimizationRole,
        # max(0, ...) so storage in charging mode (negative p_set) does not
        # appear as a downward bar in the stacked generation plot.
        lambda a: max(0.0, float(behavior.observe(a.aid, "active_power") or 0.0)),
    )

    # -- Simulate --
    # Run until just past the LAST snapshot inside the window (index
    # simulate_days*24-1). Since _clip_scenario trims the timeseries before
    # PyPSABehavior schedules its ticks, no snapshot fires after that point
    # anyway; stopping here just avoids simulating an empty final hour.
    sim_end_s = (simulate_days * 24 - 1) * 3600.0 + 1.0
    async with world:
        await discrete_step_until(world, sim_end_s)

    # -- Output --
    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        group_map=group_map,
        net=scenario.net,
        label="FDGDM",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_scenario_argparser(__doc__, default_name_base="fdgdm_withlosses")
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
