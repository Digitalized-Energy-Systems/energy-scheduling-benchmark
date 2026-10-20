"""Central (single-agent) economic dispatch scenario — perfect-foresight baseline.

Flow
----
1. Every generator and storage unit sends the leader one :class:`UnitReport`
   on ``on_ready``: its marginal cost plus its full-horizon power bounds
   (generators) or its SOC parameters (storage).
2. Once all units have reported, the leader solves a single multi-period
   Pyomo LP over the whole horizon
   (:func:`~energy_scheduling_benchmark.dispatch.solve_multiperiod_dispatch`)
   that meets the aggregated demand at every timestep and co-optimises
   storage charge/discharge timing.
3. Loads stream their demand each hour; the leader applies that hour's
   set-points via ``regulate`` (the shared :class:`PowerLoadAggregator`
   machinery, including its post-solve power-balance check).

Participants, bounds and storage parameters are built with the same helpers
the distributed scenarios use (``collect_generator_refs``, ``build_p_max_vec``,
``StorageParams.from_statics``), and storage uses the same SOC model and
cyclic terminal condition as their storage actors. The leader sees the same
full-horizon information as the distributed leaders, so this scenario's cost
is a lower bound for theirs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from distributed_resource_optimization.carrier.mango import RoleLocalMessage
from mango import AgentAddress, Role, RoleAgent, agent_composed_of

from energy_scheduling_benchmark import LOAD, STORAGE, THERMAL
from energy_scheduling_benchmark.dispatch import StorageSpec, solve_multiperiod_dispatch
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
    build_world,
    collect_generator_refs,
    install_topology,
    loss_tolerance,
    participant_graph,
    register_component_agent,
    resolve_loss_rate,
    run_scenario_main,
    run_simulation,
    thermal_cost_quad,
    write_scenario_outputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class UnitReport:
    """One-shot report of a unit's horizon parameters to the leader.

    Generators fill ``p_min``/``p_max`` (per-timestep bounds, MW); storage
    units fill ``storage`` instead.
    """

    aid: str
    cost: float
    p_min: np.ndarray | None = None
    #: Quadratic cost coefficient c₂ (``--cost-model quadratic``, thermal only).
    cost_quad: float = 0.0
    p_max: np.ndarray | None = None
    storage: StorageParams | None = None


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class ReportRequest(RoleLocalMessage):
    """Leader → unit: send your :class:`UnitReport` (its first one was lost)."""


class UnitReporter(Role):
    """Sends the unit's :class:`UnitReport` to the leader once on ``on_ready``,
    and again whenever the leader asks for it (:class:`ReportRequest`).

    Set-points are applied by the leader through the environment's
    ``regulate`` action, so this role has nothing else to do.
    """

    def __init__(self, report: UnitReport, leader: AgentAddress) -> None:
        super().__init__()
        self._report = report
        self._leader = leader

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._on_request, lambda c, m: isinstance(c, ReportRequest)
        )

    def on_ready(self) -> None:
        self.context.schedule_instant_message(self._report, self._leader)

    def _on_request(self, content: ReportRequest, meta: dict) -> None:
        self.context.schedule_instant_task(self.context.reply_to(self._report, meta))


class CentralDispatchAggregator(PowerLoadAggregator):
    """Leader: collects all :class:`UnitReport`\\ s, solves the horizon LP once,
    then dispatches it hour by hour like the distributed scenarios' leader.
    """

    def __init__(self, *, demand_target: np.ndarray, **kwargs: Any) -> None:
        super().__init__(
            trigger=None,
            finished_message_type=UnitReport,
            build_start_message=lambda: None,
            demand_target=demand_target,
            **kwargs,
        )
        self._reports: dict[str, UnitReport] = {}
        self.solver_status: str | None = None
        self.objective: float | None = None

    def on_ready(self) -> None:
        # Nothing to trigger: units report on their own.  Ask the ones whose
        # report has not arrived after the poll interval (lost message).
        self._arm_hour_deadlines()
        self._schedule_result_poll(1)

    def _result_request(self) -> ReportRequest:
        return ReportRequest()

    def _give_up(self, missing: list[str]) -> None:
        """Solve without the units that never reported (degraded run)."""
        self._solve()
        for aid in missing:
            PowerLoadAggregator._handle_finished(
                self, UnitReport(aid=aid, cost=0.0), {}
            )

    def _handle_finished(self, message: UnitReport, meta: dict) -> None:
        self._reports[message.aid] = message
        if len(self._reports) == self._n_finished_required:
            self._solve()
        super()._handle_finished(message, meta)

    def _solve(self) -> None:
        demand = np.asarray(self._demand_target, dtype=float)
        gens = [r for r in self._reports.values() if r.storage is None]
        stores = [r for r in self._reports.values() if r.storage is not None]
        result = solve_multiperiod_dispatch(
            costs=[r.cost for r in gens],
            costs_quad=[r.cost_quad for r in gens],
            p_max=np.array([r.p_max for r in gens]).reshape(len(gens), demand.size),
            p_min=np.array([r.p_min for r in gens]).reshape(len(gens), demand.size),
            demand=demand,
            storage=[
                StorageSpec(
                    e_max=r.storage.e_max,
                    p_charge_max=r.storage.p_charge_max,
                    p_discharge_max=r.storage.p_discharge_max,
                    eta_charge=r.storage.eta_charge,
                    eta_discharge=r.storage.eta_discharge,
                    e_initial=r.storage.e_initial,
                    discharge_cost=max(0.0, r.cost),
                )
                for r in stores
            ],
            # Same slack as the distributed scenarios get, so the LP stays the
            # true optimum of the problem they solve.
            slack=self._slack,
        )
        self.solver_status = result.solver_status
        if not result.success:
            # Leave the schedules empty: the balance check then flags the run.
            logger.error(
                "Central dispatch LP failed (status=%s); no schedule applied.",
                result.solver_status,
            )
            return
        self.objective = result.objective
        for r, row in zip(gens, result.generation):
            self._schedule_by_aid[r.aid] = row
        for r, row in zip(stores, result.storage):
            self._schedule_by_aid[r.aid] = row
        logger.info(
            "Central dispatch LP solved: %d generators, %d storage units, "
            "%d timesteps, objective=%.2f",
            len(gens),
            len(stores),
            demand.size,
            result.objective,
        )
        if result.slack_import is not None:
            logger.info(
                "Central dispatch LP slack: import %.1f MWh, export %.1f MWh",
                float(result.slack_import.sum()),
                float(result.slack_export.sum()),
            )


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
    name_base: str = "central_dispatch",
    simulate_days: int = 3,
    balance_tol: float = 0.01,
    strict: bool = False,
    slack: SlackSpec | None = None,
    cost_model: str = "linear",
) -> None:
    """Run the central-dispatch benchmark once and write out CSV + plots.

    :param balance_tol: Max allowed per-timestep |generation - demand| /
        demand before the schedule is flagged as infeasible.
    :param strict: If true, raise instead of writing outputs when the
        power-balance check fails.
    :param slack: Optional slack node (see :class:`SlackSpec`) that imports
        any shortfall and exports any surplus; off when ``None``.
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

    load_refs = behavior.get_components_by_type([LOAD])
    if not load_refs:
        raise RuntimeError("No loads found for central-dispatch scenario.")
    demand = build_demand_horizon(
        behavior, scenario, load_refs, simulate_days=simulate_days
    )

    gen_refs, statics_by_ref = collect_generator_refs(behavior)
    pgraph = participant_graph(scenario, topology, gen_refs, attached=load_refs)
    lt = lt.for_hops(pgraph.hops_from(load_refs[0]))
    cost_quad_by_aid = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    if not gen_refs:
        raise RuntimeError("No generators found for central-dispatch scenario.")
    group_map = build_group_map(gen_refs, statics_by_ref=statics_by_ref)

    # --- Leader agent (sits on the first load, like the distributed scenarios) ---
    leader_agent = RoleAgent()
    register_component_agent(world, leader_agent, load_refs[0])
    leader_addr = leader_agent.addr

    unit_addrs: dict[str, AgentAddress] = {}  # filled as the units register
    aggregator = CentralDispatchAggregator(
        behavior=behavior,
        number_loads=len(load_refs),
        generator_aids=[ref.component_id for ref in gen_refs],
        time_to_index=demand.time_to_index,
        schedule_by_aid={},
        demand_target=demand.target_series,
        balance_label="central dispatch",
        balance_tol=balance_tol,
        slack=slack,
        generator_addrs=unit_addrs,
        load_timeout=lt.load_timeout,
        request_timeout=lt.request_timeout,
        result_poll_s=lt.result_poll_s,
        result_retries=lt.result_retries,
    )
    leader_agent.add_role(aggregator)
    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        register_component_agent(world, agent, ref)

    # --- One reporting agent per generator / storage unit ---
    cost_by_aid: dict[str, float] = {}
    unit_agents: list[Any] = []
    for ref in gen_refs:
        statics = statics_by_ref[ref]
        cost = float(statics.get("marginal_cost", 0.0))
        cost_by_aid[ref.component_id] = cost
        p_nom = float(statics.get("p_nom", 0.0))

        if ref.element_type == STORAGE:
            report = UnitReport(
                aid=ref.component_id,
                cost=cost,
                storage=StorageParams.from_statics(statics, p_nom),
            )
        else:
            p_max = build_p_max_vec(
                scenario, ref, statics, demand.time_index, demand.horizon
            )
            p_min = np.zeros(demand.horizon, dtype=float)
            if ref.element_type == THERMAL:
                p_min_pu = float(statics.get("p_min_pu", 0.0))
                p_min[:] = max(0.0, p_min_pu * p_nom)
            report = UnitReport(
                aid=ref.component_id,
                cost=cost,
                cost_quad=cost_quad_by_aid.get(ref.component_id, 0.0),
                p_min=p_min,
                p_max=np.nan_to_num(np.asarray(p_max, dtype=float)),
            )

        agent = agent_composed_of(UnitReporter(report, leader_addr))
        register_component_agent(world, agent, ref)
        unit_addrs[ref.component_id] = agent.addr
        unit_agents.append(agent)
    # Unit reports and requests travel over the chosen topology.
    install_topology(world, pgraph, unit_agents, logical=[])

    async with world:
        await run_simulation(world, simulate_days * 24 * 3600.0)

    write_scenario_outputs(
        world,
        name_base=name_base,
        cost_by_aid=cost_by_aid,
        cost_quad_by_aid=cost_quad_by_aid,
        stacked_title="Stacked power – central dispatch",
        aggregator=aggregator,
        balance_tol=balance_tol,
        strict=strict,
        slack=slack,
        group_map=group_map,
        net=scenario.net,
        label="central dispatch",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    run_scenario_main(
        execute_test_case,
        doc=__doc__,
        default_name_base="central_dispatch",
        with_balance_tol=True,
        argv=argv,
    )


if __name__ == "__main__":
    main()
