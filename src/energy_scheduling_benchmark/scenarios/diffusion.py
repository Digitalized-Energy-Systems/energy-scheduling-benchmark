"""Distributed economic-dispatch diffusion scenario.

Python port of ``EnergySchedulingBenchmark.jl/scenario/diffusion_scenario.jl``.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. The leader starts a single averaging-diffusion run once on the full
   load *timeseries* (target demand vector).
3. After diffusion finishes, the leader applies each generator's computed
   per-timestep power set-points as the simulation clock advances.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from distributed_resource_optimization import (
    DiffusionMessage,
    LinearCostEconomicDispatchDiffusionActor,
    ReservoirStorageDiffusionActor,
    create_diffusion_participant,
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
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    cost_over_time,
    stacked_area,
    visualize_results,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as DiffusionFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
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
    make_finish_callback,
    require_lossless_transport,
    resolve_scenario,
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
    name_base: str = "diffusion",
    simulate_days: int = 3,
) -> None:
    """Run the diffusion benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    """
    require_lossless_transport(loss_percent, "Diffusion")

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
        raise RuntimeError("No loads found for diffusion scenario.")

    load_series_0 = _lookup_ts(scenario, load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index[: simulate_days * 24]
    horizon = len(time_index)  # get horizon for simulation
    # Total (aggregated) demand per timestep fitted to new time index.
    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(scenario, ref)
        if s is None:
            raise RuntimeError(f"Load timeseries missing for {ref}.")
        target_series += np.asarray(s.reindex(time_index), dtype=float)

    # Map mango simulation time (seconds from simulation start) → index in schedule. Creates dict for later plotting
    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

    # This gets populated as each generator diffusion participant finishes.
    schedule_by_aid: dict[str, np.ndarray] = {}
    leader_addr_ref: dict[str, Any | None] = {"addr": None}
    finish_callback = make_finish_callback(
        leader_addr_ref=leader_addr_ref,
        schedule_by_aid=schedule_by_aid,
        finished_message_type=DiffusionFinishedInfo,
        algorithm_label="Diffusion",
        schedule_attr="actor.P",
    )

    # -- Generator agents --
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    # filter out hydro as they are not chargeable
    gen_refs = [gen for gen in gen_refs if "hydro" not in gen.component_id]
    # sort out devices with zero max power/nominal power
    gen_refs = [
        gen for gen in gen_refs if behavior.get_statics(gen).get("p_nom", 0.0) != 0.0
    ]

    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    # --- Per-generator epsilon (capacity-scaled) ---
    # A single shared epsilon gives every generator the same price-response
    # band width (epsilon * p_nom) above its own marginal cost. For merit
    # order to hold, that band must be small relative to the spread of
    # marginal costs across generators — otherwise many generators are still
    # in their partial "ramp" region at the clearing price simultaneously, so
    # power gets spread roughly by capacity rather than sorted by cost. On
    # real PyPSA-Eur networks generator capacities span orders of magnitude
    # (tens to tens-of-thousands of MW); scaling the band off typical/mean
    # capacity makes it far wider than the cost spread, breaking merit order.
    # Scale epsilon inversely with p_nom so every generator's band is a
    # small, fixed fraction of the cost spread, independent of its capacity.
    default_epsilon = 0.1
    nonstorage_refs = [ref for ref in gen_refs if ref.element_type != STORAGE]
    eps_by_aid: dict[str, float] = {}
    costs_all: list[float] = []
    if len(nonstorage_refs) >= 2:
        costs_all = [
            float(behavior.get_statics(r).get("marginal_cost", 0.0))
            for r in nonstorage_refs
        ]
        cost_range = max(costs_all) - min(costs_all)
        target_band = max(default_epsilon, 0.1 * cost_range)
    else:
        target_band = default_epsilon
    for ref in nonstorage_refs:
        p_nom_ref = float(behavior.get_statics(ref).get("p_nom", 0.0))
        eps_by_aid[ref.component_id] = target_band / max(p_nom_ref, 1.0)

    # --- Stability-derived gradient step ---
    # Ces et al. 2025 tune the feedback gain ε offline (genetic algorithm,
    # Sec. 3.5) because their agents cannot see the system. This setup code
    # can: each actor's price response has slope p_nom/target_band, so the
    # dual-ascent loop (whose combine step averages the n gradients) is
    # stable iff ε < 2n/Σ(p_nom_i/band). Take a quarter of that bound. A
    # fixed ε that converges on the toy network (~100 MW) oscillates without
    # ever balancing on GW-scale networks, where the aggregate slope is four
    # orders of magnitude steeper.
    total_p_nom = sum(
        float(behavior.get_statics(ref).get("p_nom", 0.0)) for ref in gen_refs
    )
    grad_step = 0.5 * n_gens * target_band / max(total_p_nom, 1.0)

    # Warm-start λ at the mean marginal cost — Ces et al. 2025 initialise the
    # incremental cost from the cost coefficients at the initial dispatch
    # (eqs. 23/24) rather than from an arbitrary constant, which cuts the
    # approach phase of the iteration considerably on networks whose clearing
    # price is far from any fixed default.
    initial_lam = float(np.mean(costs_all)) if costs_all else 10.0

    gen_agents: list[RoleAgent] = []
    cost_by_aid: dict[str, float] = {}
    for ref in gen_refs:
        statics = behavior.get_statics(ref)
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        # Build a p_max vector aligned with the load horizon.
        ts = _lookup_ts(scenario, ref)
        if ts is None:
            p_max_vec = np.full(horizon, p_nom, dtype=float)
        else:
            values = np.asarray(ts.reindex(time_index), dtype=float)
            # For renewable generators, `PyPSABehavior` interprets these as
            # per-unit availability and multiplies by the nominal capacity.
            if ref.element_type == RENEWABLE:
                p_max_vec = values * p_nom
            else:
                p_max_vec = values

        if ref.element_type == STORAGE:
            # get data for storage from model and if not provided use substitutes
            p_min_pu = float(statics.get("p_min_pu", -1.0))
            p_max_pu = float(statics.get("p_max_pu", 1.0))

            p_charge_max = max(0.0, (-p_min_pu * p_nom) if p_min_pu < 0.0 else p_nom)
            p_discharge_max = max(0.0, p_max_pu * p_nom)

            max_hours = float(statics.get("max_hours", 100.0))
            e_max = max(1e-6, p_nom * max_hours)

            eta_charge = float(
                statics.get("efficiency_store", statics.get("efficiency_charge", 0.95))
            )
            eta_discharge = float(
                statics.get(
                    "efficiency_dispatch", statics.get("efficiency_discharge", 0.95)
                )
            )

            soc_initial_raw = statics.get("state_of_charge_initial", np.nan)
            if np.isfinite(soc_initial_raw) and float(soc_initial_raw) > 1e-9:
                soc_initial_abs = float(soc_initial_raw)
            else:
                # PyPSA default is often 0.0; treat that as "unspecified" here so
                # storage can participate in the benchmark without extra config.
                soc_initial_abs = 0.5 * e_max
            e_initial = float(np.clip(soc_initial_abs / e_max, 0.0, 1.0))

            actor = ReservoirStorageDiffusionActor(
                e_max=e_max,
                p_charge_max=p_charge_max,
                p_discharge_max=p_discharge_max,
                eta_charge=max(1e-6, eta_charge),
                eta_discharge=max(1e-6, eta_discharge),
                e_initial=e_initial,
                e_final=e_initial,
                soc_min=0.0,
                soc_max=1.0,
                charge_cost=max(0.0, cost),
                discharge_cost=max(0.0, cost),
                epsilon=target_band / max(p_nom, 1.0),
                n_guess=n_gens,
            )
        else:
            p_min = 0.0
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                p_min = max(0.0, p_min_pu * p_nom)
            actor = LinearCostEconomicDispatchDiffusionActor(
                cost=cost,
                p_max=p_max_vec,
                epsilon=eps_by_aid[ref.component_id],
                p_min=p_min,
                n_guess=n_gens,
            )
        participant = create_diffusion_participant(
            finish_callback=finish_callback,
            diffusion_actor=actor,
            initial_lam=initial_lam,
            max_iter=2000,
            epsilon=grad_step,
            tol=1e-3,
            horizon=horizon,
        )

        opt_role = DistributedOptimizationRole(participant)

        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    if not gen_agents:
        raise RuntimeError("No generator agents found for diffusion scenario.")

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    def build_start_message() -> Any:
        # Kick off diffusion exactly once on the full time series.
        # The diffusion actor's p_max is also vectorised, so it constrains
        # each timestep independently.
        return DiffusionMessage(
            phi=np.full(len(target_series), initial_lam),
            k=0,
            data=np.asarray(target_series, dtype=float),
            initial=True,
        )

    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            trigger=gen_agents[0].addr,
            time_to_index=time_to_index,
            schedule_by_aid=schedule_by_aid,
            finished_message_type=DiffusionFinishedInfo,
            build_start_message=build_start_message,
            demand_target=target_series,
            balance_label="Diffusion",
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    # add all other load agents
    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    # Fully-connected diffusion topology across all generator agents.
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
    record_agent_having(
        world,
        "P",
        DistributedOptimizationRole,
        lambda a: float(behavior.observe(a.aid, "active_power") or 0.0),
    )

    # -- Simulate --
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # -- Output --

    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")

    # Drop sub-second convergence-phase noise (diffusion iterations tick
    # multiple times per hour): keep the last recorded state per hourly
    # snapshot so the CSV and plots show one row per PyPSA timestep.
    t_P, Y_P = _keep_hourly(t_P, Y_P)
    t_t, Y_t = _keep_hourly(t_t, Y_t)
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    m = min(len(t_P), len(target_series))
    t_P, Y_P, target_series = t_P[:m], Y_P[:m], target_series[:m]

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
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power",
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
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_scenario_argparser(__doc__, default_name_base="diffusion_withlosses")
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
