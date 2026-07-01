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

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
class FDGDMFinishedInfo:
    """Notifies the leader that a generator FDGDM run is done."""

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
    """Coordinates FDGDM once, then applies the resulting schedule."""

    def __init__(
        self,
        *,
        behavior: PyPSABehavior,
        number_loads: int,
        generator_aids: list[str],
        n_fdgdm_participants: int,
        fdgdm_trigger,
        time_to_index: dict[float, int],
        target_series: np.ndarray,
        schedule_by_aid: dict[str, np.ndarray],
        fdgdm_initial_p: np.ndarray,
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._n_fdgdm_participants = n_fdgdm_participants
        self._trigger = fdgdm_trigger
        self._time_to_index = time_to_index
        self._target_series = target_series
        self._schedule_by_aid = schedule_by_aid
        self._fdgdm_initial_p = fdgdm_initial_p

        self._fdgdm_ready: bool = False
        self._fdgdm_finished_aids: set[str] = set()
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
            self._handle_fdgdm_finished,
            lambda c, m: isinstance(c, FDGDMFinishedInfo),
        )

    def on_ready(self) -> None:
        if self._trigger is None or self._n_fdgdm_participants == 0:
            # All schedules are pre-filled; nothing to run.
            self._fdgdm_ready = True
            return
        # Kick off FDGDM on the thermal sub-problem using the pre-computed
        # demand-feasible initial allocation (capped at min p_max per step).
        start_msg = create_fdgdm_start(data=self._fdgdm_initial_p)
        asyncio.create_task(self.context.send_message(start_msg, self._trigger))

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

    def _handle_fdgdm_finished(self, message: FDGDMFinishedInfo, meta: dict) -> None:
        self._fdgdm_finished_aids.add(message.aid)
        if len(self._fdgdm_finished_aids) == self._n_fdgdm_participants:
            self._fdgdm_ready = True
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

        if self._fdgdm_ready:
            self._apply_schedule_index(idx, total)
        else:
            # FDGDM still running; buffer until schedule is available.
            self._pending[idx] = total


# ---------------------------------------------------------------------------
# FDGDM-finished callback
# ---------------------------------------------------------------------------


def _make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
):
    """Build the ``(algorithm, carrier) -> None`` hook that stores the schedule."""

    def handle_fdgdm_finished(algorithm, carrier) -> None:
        role = carrier._parent
        aid = role.context.aid

        schedule_by_aid[aid] = np.asarray(algorithm.actor.P, dtype=float).copy()

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(FDGDMFinishedInfo(aid=aid), leader_addr)
            )
        logger.info("FDGDM finished for %s (schedule len=%s)", aid, schedule_by_aid[aid].size)

    return handle_fdgdm_finished


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
            storable = (max_energy - soc) / efficiency_store if efficiency_store > 0 else 0.0
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
    loss_percent: float = 0.00005,
    name_base: str = "fdgdm",
    simulate_days: int = 3,
) -> None:
    """Run the FDGDM benchmark once and write out CSV + plots.

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
    com_sim = SimpleCommunicationSimulation(default_delay_s=delay_s, loss_percent=loss_percent)
    world = create_world(start_time=0.0, communication_sim=com_sim, environment=environment)

    # ------------------------------------------------------------------
    # Vectorise across the full load horizon
    # ------------------------------------------------------------------
    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for FDGDM scenario.")

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

    time_index = load_series_0.index[: simulate_days * 24]
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

    # -- Generator classification --
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    gen_refs = [gen for gen in gen_refs if "hydro" not in gen.component_id]
    gen_refs = [
        gen
        for gen in gen_refs
        if behavior._dataframe_for(gen.element_type).loc[gen.component_id].get("p_nom", 0.0) != 0.0
    ]

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

    for ref in renewable_refs:
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        p_nom = float(statics.get("p_nom", 0.0))
        ts = _lookup_ts(ref)
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
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
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
            inflow_mwh = np.full(horizon, float(statics.get("inflow", 0.0)), dtype=float)

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
        _statics_th = behavior._dataframe_for(_ref_th.element_type).loc[_ref_th.component_id]
        _p_nom_th = float(_statics_th.get("p_nom", 0.0))
        _ts_th = _lookup_ts(_ref_th)
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
            float(behavior._dataframe_for(r.element_type).loc[r.component_id].get("marginal_cost", 0.0))
            for r in thermal_refs
        ]
        cost_diff = max(thermal_costs) - min(thermal_costs)
        thermal_p_nom_list = [
            float(behavior._dataframe_for(r.element_type).loc[r.component_id].get("p_nom", 0.0))
            for r in thermal_refs
        ]
        total_p_nom = sum(thermal_p_nom_list)
        active_steps = adjusted_target[~zero_demand_mask]
        min_active = float(active_steps.min()) if len(active_steps) > 0 else 1.0
        base_epsilon = max(0.1, cost_diff / min_active)
        epsilon_by_agent = [
            base_epsilon * total_p_nom / max(p_nom, 1.0)
            for p_nom in thermal_p_nom_list
        ]
    else:
        epsilon_by_agent = [0.1]

    # Dummy kickoff data — each actor uses its own initial_schedule instead.
    fdgdm_initial_p = adjusted_target / max(n_thermals, 1)

    # -- Thermal generator agents (FDGDM participants) --
    thermal_gen_agents: list[RoleAgent] = []
    for i, ref in enumerate(thermal_refs):
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
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

    leader_agent.add_role(
        PowerLoadAggregator(
            behavior=behavior,
            number_loads=len(load_refs),
            generator_aids=generator_aids,
            n_fdgdm_participants=len(thermal_refs),
            fdgdm_trigger=gen_agents[0].addr if gen_agents else None,
            time_to_index=time_to_index,
            target_series=adjusted_target,
            schedule_by_aid=schedule_by_aid,
            fdgdm_initial_p=fdgdm_initial_p,
        )
    )
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(PowerLoadMonitoring(behavior=behavior, target=leader_addr))
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
    record_agent_having(
        world,
        "P",
        DistributedOptimizationRole,
        # max(0, ...) so storage in charging mode (negative p_set) does not
        # appear as a downward bar in the stacked generation plot.
        lambda a: max(0.0, float(behavior.observe(a.aid, "active_power") or 0.0)),
    )

    # -- Simulate --
    # Run until just past the LAST desired snapshot (index simulate_days*24-1).
    # Using simulate_days*24*3600 exactly would hit the NEXT day's midnight
    # snapshot in a multi-day network (e.g. a 6-day .nc with simulate_days=3
    # fires the day-4 snapshot at t=259200 s), creating a stale duplicate row
    # because the 0.02 s-delayed PowerLoadInfo for that step never arrives.
    sim_end_s = (simulate_days * 24 - 1) * 3600.0 + 1.0
    async with world:
        await discrete_step_until(world, sim_end_s)

    # -- Output --
    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")

    # Drop sub-second convergence-phase noise: keep the last recorded state
    # per hourly snapshot so the CSV and plots show one row per PyPSA timestep.
    t_P, Y_P = _keep_hourly(t_P, Y_P)
    t_t, Y_t = _keep_hourly(t_t, Y_t)

    target_series_plot = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))

    _write_agent_recordings_csv(world, f"{name_base}-df.csv")

    visualize_results(world, write_to=f"{name_base}-observation.pdf")

    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_series_plot,
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power",
        write_to=f"{name_base}-stacked.pdf",
    )


def _keep_hourly(
    t_list: list | np.ndarray,
    Y_arr: np.ndarray,
    step_s: float = 3600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the last recorded state per hourly snapshot, dropping sub-second noise."""
    if len(t_list) == 0:
        return np.asarray(t_list), Y_arr
    t = np.asarray(t_list, dtype=float)
    buckets = (t // step_s).astype(int)
    # last index of each unique bucket (reverse → find-first → un-reverse)
    _, first_in_rev = np.unique(buckets[::-1], return_index=True)
    last_idx = np.sort(len(t) - 1 - first_in_rev)
    return t[last_idx], Y_arr[last_idx]


def _write_agent_recordings_csv(world, path: str, snapshot_step_s: float = 3600.0) -> None:
    """Serialise every per-agent recording as one wide CSV.

    When *snapshot_step_s* > 0, only the last recorded state per time bucket
    is kept (default: one row per hour), removing sub-second FDGDM convergence
    noise from the output.
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

    if snapshot_step_s > 0 and not df.empty:
        bucket = (df.index.to_series() // snapshot_step_s).astype(int)
        df = df.groupby(bucket.values).last()
        df.index.name = "time"

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
    parser.add_argument("--name-base", type=str, default="fdgdm_withlosses")
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
            name_base="fdgdm_without_losses",
            simulate_days=simulate_days,
        )
    )
