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
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
    Role,
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

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    PowerUpdateInfo,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.scenarios._common import _clip_scenario
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
class DEEDADMMFinishedInfo:
    """Notifies the leader that a generator DEED-ADMM run is done."""

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
    """Coordinates DEED-ADMM once, then applies the resulting schedule."""

    def __init__(
        self,
        *,
        behavior: PyPSABehavior,
        number_loads: int,
        generator_aids: list[str],
        deed_admm_trigger,
        time_to_index: dict[float, int],
        target_series: np.ndarray,
        schedule_by_aid: dict[str, np.ndarray],
        n_time: int,
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = deed_admm_trigger
        self._time_to_index = time_to_index
        self._target_series = target_series
        self._schedule_by_aid = schedule_by_aid
        self._n_time = n_time

        self._ready: bool = False
        self._finished_aids: set[str] = set()
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
            self._handle_finished,
            lambda c, m: isinstance(c, DEEDADMMFinishedInfo),
        )

    def on_ready(self) -> None:
        initial = DEEDADMMMessage(
            lam=np.zeros(self._n_time),
            xi=np.zeros(self._n_time),
            k=0,
            data=None,
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

    def _handle_finished(self, message: DEEDADMMFinishedInfo, meta: dict) -> None:
        self._finished_aids.add(message.aid)
        if len(self._finished_aids) == len(self._generator_aids):
            self._ready = True
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

        if self._ready:
            self._apply_schedule_index(idx, total)
        else:
            self._pending[idx] = total


# ---------------------------------------------------------------------------
# Finish callback
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
):
    """Build the ``(algorithm, carrier) -> None`` hook that stores the schedule."""

    def handle_finished(algorithm, carrier) -> None:
        role = carrier._parent
        aid = role.context.aid

        schedule_by_aid[aid] = np.asarray(algorithm.P, dtype=float).copy()

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(DEEDADMMFinishedInfo(aid=aid), leader_addr)
            )
        logger.info(
            "DEED-ADMM finished for %s (schedule len=%s)", aid, schedule_by_aid[aid].size
        )

    return handle_finished


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.00005,
    name_base: str = "deed_admm",
    simulate_days: int = 3,
    gamma: float = 0.05,
    max_iter: int = 500,
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
    """
    if scenario is None:
        scenario = build_toy_network(periods=simulate_days * 24)

    scenario = _clip_scenario(scenario, simulate_days)
    behavior = PyPSABehavior.from_scenario(scenario)
    environment = DefaultEnvironment(behavior=behavior)
    com_sim = SimpleCommunicationSimulation(default_delay_s=delay_s, loss_percent=loss_percent)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    # ------------------------------------------------------------------
    # Build demand horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for DEED-ADMM scenario.")

    def _lookup_ts(ref) -> Any | None:
        ts = scenario.timeseries.get(ref)
        if ts is not None:
            return ts
        for k, v in scenario.timeseries.items():
            if (
                getattr(k, "element_type", None) == getattr(ref, "element_type", None)
                and getattr(k, "component_id", None) == getattr(ref, "component_id", None)
            ):
                return v
        return None

    load_series_0 = _lookup_ts(load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    horizon = len(time_index)

    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(ref)
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

    # ------------------------------------------------------------------
    # Generator agents
    # ------------------------------------------------------------------
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    gen_refs = [ref for ref in gen_refs if "hydro" not in ref.component_id]
    n_gens = len(gen_refs)
    generator_aids = [ref.component_id for ref in gen_refs]

    # Demand allocation: only generators (thermal + renewable) share demand.
    # Storage gets d_i = 0 — it contributes net injection, not demand consumption.
    pure_gen_refs = [r for r in gen_refs if r.element_type != STORAGE]
    n_gen_only = max(len(pure_gen_refs), 1)
    d_i_gen = target_series / n_gen_only
    d_i_stor = np.zeros(horizon, dtype=float)

    gen_agents: list[RoleAgent] = []
    for ref in gen_refs:
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
        p_nom = float(statics.get("p_nom", 0.0))

        ts = _lookup_ts(ref)
        if ts is None:
            p_max_vec = np.full(horizon, p_nom, dtype=float)
        else:
            values = np.asarray(ts.reindex(time_index), dtype=float)
            if ref.element_type == RENEWABLE:
                p_max_vec = values * p_nom
            else:
                p_max_vec = values

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
            soc0_raw_f = float(soc0_raw) if np.isfinite(float(soc0_raw)) else float("nan")
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
            d_i_ref = d_i_stor
        elif ref.element_type == RENEWABLE:
            participant = create_deed_admm_renewable_participant(
                finish_callback,
                p_max_timeseries=p_max_vec,
                d_i=d_i_gen,
                gamma=gamma,
                max_iter=max_iter,
                n_agents=n_gens,
            )
            d_i_ref = d_i_gen
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
            d_i_ref = d_i_gen
        _ = d_i_ref  # used above in factory calls; suppress unused-var lint

        opt_role = DistributedOptimizationRole(participant)
        agent = agent_composed_of(opt_role)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    # ------------------------------------------------------------------
    # Load agents
    # ------------------------------------------------------------------
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
            deed_admm_trigger=gen_agents[0].addr,
            time_to_index=time_to_index,
            target_series=target_series,
            schedule_by_aid=schedule_by_aid,
            n_time=horizon,
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(PowerLoadMonitoring(behavior=behavior, target=leader_addr))
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

    # Fully-connected peer topology across all generator agents.
    topology = complete_topology(len(gen_agents))
    auto_assign(topology, gen_agents)

    if not gen_agents:
        raise RuntimeError("No generator agents found for DEED-ADMM scenario.")

    # ------------------------------------------------------------------
    # Recordings
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")
    target_out = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))

    visualize_results(world, write_to=f"{name_base}-observation.pdf")

    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_out,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power (DEED-ADMM)",
        write_to=f"{name_base}-stacked.pdf",
    )

    _write_agent_recordings_csv(world, f"{name_base}-df.csv")


def _scalar(v: Any) -> float:
    arr = np.asarray(v).ravel()
    return float(arr[0]) if arr.size else 0.0


def _write_agent_recordings_csv(world, path: str) -> None:
    """Serialise every per-agent recording as one wide CSV."""
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
    parser.add_argument("--name-base", type=str, default="deed_admm")
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=500)
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
            gamma=args.gamma,
            max_iter=args.max_iter,
        )
    )


if __name__ == "__main__":
    # main() # commeted out for testing

    ###############
    ### Testing ###
    ###############

    simulate_days = 3

    # scenario = build_toy_network(periods=simulate_days * 24) # toy
    # scenario = load_scenario("storage-hvdc")
    scenario = load_scenario("../networks/base_s_1_elec_2020.nc")

    logging.basicConfig(level=getattr(logging, "INFO", logging.INFO))

    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=0.02,
            loss_percent=0.0,
            name_base="deed-admm_without_losses",
            simulate_days=simulate_days,
        )
    )
