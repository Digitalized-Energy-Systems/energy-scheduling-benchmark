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

import asyncio
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
    PyPSABehavior,
)
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    stacked_area,
    visualize_results,
)
from energy_scheduling_benchmark.scenarios._common import (
    OptimizationFinishedInfo as ConsensusFinishedInfo,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
    PowerLoadMonitoring,
    ScenarioData,
    _clip_scenario,
    _lookup_ts,
    _write_agent_recordings_csv,
    build_scenario_argparser,
    build_toy_network,
    load_scenario,
    make_finish_callback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.00005,
    name_base: str = "consensus",
    simulate_days: int = 3,
) -> None:
    """Run the consensus benchmark once and write out CSV + plots.

    Parameters
    ----------
    scenario:
        Pre-built :class:`ScenarioData`.  Defaults to the toy 5-bus network.
    """
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    behavior = PyPSABehavior.from_scenario(scenario)
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
        raise RuntimeError("No loads found for consensus scenario.")

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

    # Map mango simulation time (seconds from simulation start) → index in schedule.
    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

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
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    # sort out hydro as they are not charable
    gen_refs = [gen for gen in gen_refs if "hydro" not in gen.component_id]
    generator_aids = [ref.component_id for ref in gen_refs]

    gen_agents: list[RoleAgent] = []
    for gen_idx, ref in enumerate(gen_refs):
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
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

            actor = ReservoirStorageConsensusActor(
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
                epsilon=0.1,
            )
        else:
            p_min = 0.0
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                p_min = max(0.0, p_min_pu * p_nom)
            actor = LinearCostEconomicDispatchConsensusActor(
                cost=cost, p_max=p_max_vec, p_min=p_min
            )
        # The first generator agent is the leader (Jian et al. 2020, eq. 22);
        # it pins λ toward the real system-wide power imbalance ΔP, while all
        # other generators are followers doing pure neighbour averaging.
        participant = create_averaging_consensus_participant(
            finish_callback=finish_callback,
            consensus_actor=actor,
            max_iter=200,
            alpha=0.2,
            is_leader=(gen_idx == 0),
            leader_gain=0.05,
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

    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            trigger=gen_agents[0].addr,
            time_to_index=time_to_index,
            schedule_by_aid=schedule_by_aid,
            finished_message_type=ConsensusFinishedInfo,
            build_start_message=build_start_message,
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

    # Fully-connected consensus topology across all generator agents.
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
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))

    visualize_results(world, write_to=f"{name_base}-observation.pdf")

    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power",
        write_to=f"{name_base}-stacked.pdf",
    )

    _write_agent_recordings_csv(world, f"{name_base}-df.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_scenario_argparser(__doc__, default_name_base="consensus_withoutlosses")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.network == "toy":
        scenario = build_toy_network(periods=args.simulate_days * 24)
    else:
        scenario = load_scenario(args.network)

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
