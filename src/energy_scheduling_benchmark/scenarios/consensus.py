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
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
    Role,
    RoleAgent,
    agent_composed_of,
    auto_assign,
    complete_topology,
)
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import (create_world, discrete_step_until, record_agent_having)

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    THERMAL,
    STORAGE,
    PowerUpdateInfo,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    stacked_area,
    visualize_results,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PowerLoadInfo:
    """Load-side update pushed to the aggregator."""

    power_load: float
    time: float


@dataclass
class ConsensusFinishedInfo:
    """Notifies the leader that a generator consensus run is done."""

    aid: str


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class PowerLoadMonitoring(Role):
    """Observes ``max_active_power`` and forwards it to *target*."""

    def __init__(self, behavior: PyPSABehavior, target) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        power = self._behavior.observe(self.context.aid, "max_active_power")
        sim_time = self.context.current_timestamp
        asyncio.create_task(
            self.context.send_message(
                PowerLoadInfo(power_load=float(power), time=sim_time),
                self._target,
            )
        )


class PowerLoadAggregator(Role):
    """Coordinates consensus once, then applies the resulting schedule."""

    def __init__(
        self,
        *,
        behavior: PyPSABehavior,
        number_loads: int,
        generator_aids: list[str],
        consensus_trigger,
        time_to_index: dict[float, int],
        target_series: np.ndarray,
        schedule_by_aid: dict[str, np.ndarray],
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = consensus_trigger
        self._time_to_index = time_to_index
        self._target_series = target_series
        self._schedule_by_aid = schedule_by_aid

        self._consensus_ready: bool = False
        self._consensus_finished_aids: set[str] = set()
        self._pending: dict[int, float] = {}

        self.demand_map: dict[Any, list[float]] = {}
        self.target: float = 0.0

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_load_info,
            lambda c, m: isinstance(c, PowerLoadInfo),
        )
        self.context.subscribe_message(
            self,
            self._handle_consensus_finished,
            lambda c, m: isinstance(c, ConsensusFinishedInfo),
        )

    def on_ready(self) -> None:
        # Kick off consensus exactly once on the full time series.
        # The consensus actor's p_max is also vectorised, so it constrains
        # each timestep independently.
        initial = AveragingConsensusMessage(
            lam=np.full(len(self._target_series), 10.0),
            k=0,
            data=np.asarray(self._target_series, dtype=float),
            initial=True,
        )
        asyncio.create_task(self.context.send_message(initial, self._trigger))

    def _apply_schedule_index(self, idx: int, target_total: float) -> None:
        self.target = float(target_total)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            p_schedule = np.asarray(schedule).ravel()
            if idx >= p_schedule.size:
                continue
            self._behavior.act(aid, "regulate", float(p_schedule[idx]))

    def _handle_consensus_finished(self, message: ConsensusFinishedInfo, meta: dict) -> None:
        self._consensus_finished_aids.add(message.aid)
        if len(self._consensus_finished_aids) == len(self._generator_aids):
            self._consensus_ready = True
            if self._pending:
                for idx in sorted(self._pending):
                    total = self._pending[idx]
                    self._apply_schedule_index(idx, total)
                self._pending.clear()

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        bucket = self.demand_map.setdefault(message.time, [])
        bucket.append(float(message.power_load))

        if len(bucket) != self.number_loads:
            return

        total = float(sum(bucket))
        idx = self._time_to_index.get(round(float(message.time), 6))
        if idx is None:
            return

        if self._consensus_ready:
            self._apply_schedule_index(idx, total)
        else:
            # Consensus is still running; remember what we need to apply once
            # the schedule is available.
            self._pending[idx] = total
            # Do not update self.target yet; that keeps P and target aligned in recordings.


# ---------------------------------------------------------------------------
# Consensus-finished callback
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
):
    """Build the ``(algorithm, carrier) -> None`` hook that stores the schedule."""

    def handle_consensus_finished(algorithm, carrier) -> None:
        role = carrier._parent
        aid = role.context.aid

        # Store the full per-timestep schedule.
        schedule_by_aid[aid] = np.asarray(algorithm.actor.P, dtype=float).copy()

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(ConsensusFinishedInfo(aid=aid), leader_addr)
            )
        logger.info("Consensus finished for %s (schedule len=%s)", aid, schedule_by_aid[aid].size)

    return handle_consensus_finished


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

    behavior = PyPSABehavior.from_scenario(scenario)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=delay_s, loss_percent=loss_percent)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    # ------------------------------------------------------------------
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for consensus scenario.")

    def _lookup_ts(ref) -> Any | None:
        # `scenario.timeseries` uses ComponentRef keys; prefer exact match, but
        # fall back to a (element_type, component_id) lookup to be robust.
        ts = scenario.timeseries.get(ref)
        if ts is not None:
            return ts
        for k, v in scenario.timeseries.items():
            if getattr(k, "element_type", None) == getattr(ref, "element_type", None) and getattr(
                k, "component_id", None
            ) == getattr(ref, "component_id", None):
                return v
        return None

    load_series_0 = _lookup_ts(load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    horizon = len(time_index)  # get horizon for simulation
    # Total (aggregated) demand per timestep fitted to new time index.
    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(ref)
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
    finish_callback = _make_finish_callback(leader_addr_ref=leader_addr_ref, schedule_by_aid=schedule_by_aid)

    # -- Generator agents --
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    # sort out hydro as they are not charable
    gen_refs = [gen for gen in gen_refs if "hydro" not in gen.component_id]
    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    gen_agents: list[RoleAgent] = []
    for ref in gen_refs:
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
        p_nom = float(statics.get("p_nom", 0.0))

        # Build a p_max vector aligned with the load horizon.
        ts = _lookup_ts(ref)
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

            eta_charge = float(statics.get("efficiency_store", statics.get("efficiency_charge", 0.95)))
            eta_discharge = float(statics.get("efficiency_dispatch", statics.get("efficiency_discharge", 0.95)))

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
                n_guess=n_gens,
            )
        else:
            actor = LinearCostEconomicDispatchConsensusActor(
                cost=cost, p_max=p_max_vec, n_guess=n_gens, rho=0.05
            )
        participant = create_averaging_consensus_participant(
            finish_callback=finish_callback,
            consensus_actor=actor,
            max_iter=200,
            alpha=0.2,
        )

        opt_role = DistributedOptimizationRole(participant)

        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    # -- Load agents (two-pass: register leader first so its addr is known) --
    leader_agent = RoleAgent()
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr
    leader_addr_ref["addr"] = leader_addr

    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            consensus_trigger=gen_agents[0].addr,
            time_to_index=time_to_index,
            target_series=target_series,
            schedule_by_aid=schedule_by_aid,
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    # add all other load agents
    for ref in load_refs[1:]:
        agent = agent_composed_of(PowerLoadMonitoring(behavior=behavior, target=leader_addr))
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    # Fully-connected consensus topology across all generator agents.
    topology = complete_topology(len(gen_agents))
    auto_assign(topology, gen_agents)

    if not gen_agents:
        raise RuntimeError("No generator agents found for consensus scenario.")


    # -- Recordings --
    record_agent_having(
        world,
        "target",
        PowerLoadAggregator,
        lambda a: next((r.target for r in a.roles if isinstance(r, PowerLoadAggregator)), 0.0
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

def _scalar(v: Any) -> float:
    arr = np.asarray(v).ravel()
    return float(arr[0]) if arr.size else 0.0


def _write_agent_recordings_csv(world, path: str) -> None:
    """Serialise every per-agent recording as one wide CSV.

    Columns are named ``{key}:{aid}``; values are scalarised to floats.
    """
    frames: list[pd.DataFrame] = []
    for key, rec in world.data_agent_collections.items():
        if not rec.timeseries:
            continue
        length = min([len(rec.time)] + [len(v) for v in rec.timeseries.values()])
        data = {
            f"{key}:{aid}": [_scalar(v) for v in values[:length]]
            for aid, values in rec.timeseries.items()
        }
        data["time"] = rec.time[:length]
        frames.append(pd.DataFrame(data).set_index("time"))

    if not frames:
        pd.DataFrame().to_csv(path)
        return

    df = pd.concat(frames, axis=1)
    df.to_csv(path)


def _scalar(v: Any) -> float:
    arr = np.asarray(v).ravel()
    return float(arr[0]) if arr.size else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=str,
        default="toy",
        help=(
            "Network source. Either 'toy' (the built-in 5-bus fixture), "
            "a PyPSA example name "
            f"({', '.join(available_examples())}), or a path to a "
            ".nc/.h5/.xlsx file or CSV folder."
        ),
    )
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--loss-percent", type=float, default=0.00005)
    parser.add_argument("--name-base", type=str, default="consensus_withlosses")
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
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
    #main() # commeted out for testing

    ###############
    ### Testing ###
    ###############

    simulate_days = 3

    #scenario = build_toy_network(periods=simulate_days * 24) # toy
    #scenario = load_scenario("storage-hvdc")
    scenario = load_scenario("../networks/base_s_1_elec_.nc")

    logging.basicConfig(level=getattr(logging, "INFO", logging.INFO))


    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=0.02,
            loss_percent=0.0,
            name_base="consensus_without_losses",
            simulate_days=simulate_days,
        )
    )
