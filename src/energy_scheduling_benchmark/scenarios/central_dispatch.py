"""Central (single-agent) economic dispatch scenario.

Python port of ``EnergySchedulingBenchmark.jl/scenario/central_dispatch_scenario.jl``.

Flow
----
1. Thermal generators are *static*: they publish their cost and nominal
   capacity once on ``on_ready``.
2. Renewable generators and loads are *dynamic*: they publish their
   current ``max_active_power`` whenever the PyPSA timeseries ticks.
3. The aggregator collects these reports per timestamp.  When the total
   matches the total number of components it solves a Pyomo LP that
   dispatches generation cost-optimally and meets the aggregated load,
   and ships ``PowerInfo`` messages back with each generator's share.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from mango import (
    AgentAddress,
    Role,
    RoleAgent,
    agent_composed_of,
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
    THERMAL,
    PowerUpdateInfo,
    PyPSABehavior,
)
from energy_scheduling_benchmark.dispatch import solve_central_dispatch
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    stacked_area,
    visualize_results,
)
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadInfo,
    PowerLoadMonitoring,
    ScenarioData,
    _clip_scenario,
    _write_agent_recordings_csv,
    build_scenario_argparser,
    build_toy_network,
    load_scenario,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PowerInfo:
    """Power set-point / observation exchanged with the aggregator."""

    power_load: float
    time: datetime


@dataclass
class GeneratorInfo:
    """Generator-side report to the aggregator."""

    max_power: float
    cost: float
    time: datetime
    addr: AgentAddress
    static: bool


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def _role_addr(role: Role) -> AgentAddress:
    return AgentAddress(role.context.addr, role.context.aid)


class StaticHandler(Role):
    """Reports a generator's static parameters once on ``on_ready``."""

    def __init__(self, behavior: PyPSABehavior, target: AgentAddress) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_ready(self) -> None:
        aid = self.context.aid
        power = self._behavior.observe(aid, "max_active_power")
        cost = self._behavior.observe(aid, "cost")
        t = self.context.current_timestamp

        asyncio.create_task(
            self.context.send_message(
                GeneratorInfo(
                    max_power=float(power),
                    cost=float(cost),
                    time=t,
                    addr=_role_addr(self),
                    static=True,
                ),
                self._target,
            )
        )


class GeneratorMonitoring(Role):
    """Handles dynamic generator updates *and* applies dispatch set-points.

    On ``PowerUpdateInfo`` events → publish the new max power + cost to the
    aggregator.  On incoming :class:`PowerInfo` → call the ``regulate``
    action to set the generator to the commanded value.
    """

    def __init__(self, behavior: PyPSABehavior, target: AgentAddress) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target
        self.P: float = 0.0

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_power_info,
            lambda c, m: isinstance(c, PowerInfo),
        )

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        aid = self.context.aid
        power = self._behavior.observe(aid, "max_active_power")
        cost = self._behavior.observe(aid, "cost")
        t = self.context.current_timestamp
        asyncio.create_task(
            self.context.send_message(
                GeneratorInfo(
                    max_power=float(power),
                    cost=float(cost),
                    time=t,
                    addr=_role_addr(self),
                    static=False,
                ),
                self._target,
            )
        )

    def _handle_power_info(self, message: PowerInfo, meta: dict) -> None:
        self.P = float(message.power_load)
        self._behavior.act(self.context.aid, "regulate", self.P)


class Aggregator(Role):
    """Central LP dispatcher — collects reports and emits set-points."""

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num = num_components
        self.demand_map: dict[Any, list[float]] = {}
        self.generator_map: dict[Any, list[GeneratorInfo]] = {}
        self.static_generators: list[GeneratorInfo] = []
        self.target: float = 0.0

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_generator_info,
            lambda c, m: isinstance(c, GeneratorInfo),
        )
        self.context.subscribe_message(
            self,
            self._handle_load_info,
            lambda c, m: isinstance(c, PowerLoadInfo),
        )

    def _handle_generator_info(self, message: GeneratorInfo, meta: dict) -> None:
        if message.static:
            self.static_generators.append(message)
        else:
            self.generator_map.setdefault(message.time, []).append(message)
        self._check_start(message.time)

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        self.demand_map.setdefault(message.time, []).append(float(message.power_load))
        self._check_start(message.time)

    def _check_start(self, time: Any) -> None:
        gen_list = self.generator_map.get(time, [])
        load_list = self.demand_map.get(time, [])
        total = len(load_list) + len(gen_list) + len(self.static_generators)
        if total != self.num:
            return

        self.target = float(sum(load_list))
        all_gen = list(self.static_generators) + list(gen_list)

        result = solve_central_dispatch(
            costs=[g.cost for g in all_gen],
            p_max=[g.max_power for g in all_gen],
            demand=self.target,
        )

        if not result.success:
            logger.warning(
                "Central dispatch failed at %s: status=%s",
                time,
                result.solver_status,
            )
            return

        logger.info(
            "Central dispatch @ %s: demand=%.2f MW, cost=%.2f",
            time,
            self.target,
            result.objective,
        )
        for gen, p in zip(all_gen, result.dispatch):
            asyncio.create_task(
                self.context.send_message(
                    PowerInfo(power_load=float(p), time=time), gen.addr
                )
            )


# ---------------------------------------------------------------------------
# Scenario entry point
# ---------------------------------------------------------------------------


async def execute_test_case(
    *,
    scenario: ScenarioData | None = None,
    delay_s: float = 0.02,
    loss_percent: float = 0.00005,
    name_base: str = "central_dispatch",
    simulate_days: int = 3,
) -> None:
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

    all_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, LOAD])
    leader_agent: RoleAgent | None = None

    for ref in all_refs:
        # The first agent carries the Aggregator role (acts as leader).
        if leader_agent is None:
            aggregator = Aggregator(num_components=len(all_refs))
            roles: list[Role] = [aggregator]
            agent = agent_composed_of(*roles)
            world.register(agent, suggested_aid=ref.component_id)
            world.environment.install(agent, id=ref)
            leader_agent = agent
            leader_addr = leader_agent.addr
            # The leader itself may also be e.g. a thermal gen → add monitoring.
            _install_component_role(ref, behavior, leader_addr, agent)
            continue

        leader_addr = leader_agent.addr
        agent = RoleAgent()
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        _install_component_role(ref, behavior, leader_addr, agent)

    record_agent_having(
        world,
        "target",
        Aggregator,
        lambda a: next((r.target for r in a.roles if isinstance(r, Aggregator)), 0.0),
    )
    record_agent_having(
        world,
        "P",
        GeneratorMonitoring,
        lambda a: next(
            (r.P for r in a.roles if isinstance(r, GeneratorMonitoring)), 0.0
        ),
    )

    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    visualize_results(world, write_to=f"{name_base}-observation.pdf")

    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    _, Y_t, _ = agent_recording_as_plottable(world, "target")
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    m = min(len(t_P), len(target_series))

    _write_agent_recordings_csv(world, f"{name_base}-df.csv")

    stacked_area(
        np.asarray(t_P[:m]) / 3600.0,
        Y_P[:m],
        labels_P,
        target_series[:m],
        xlabel="Hour",
        ylabel="P in MW",
        title="Stacked power",
        write_to=f"{name_base}-stacked.pdf",
    )


def _install_component_role(ref, behavior, leader_addr, agent) -> None:
    """Attach the role mix appropriate for the component type."""
    if ref.element_type == LOAD:
        agent.add_role(PowerLoadMonitoring(behavior, leader_addr))
    elif ref.element_type == THERMAL:
        agent.add_role(StaticHandler(behavior, leader_addr))
        agent.add_role(GeneratorMonitoring(behavior, leader_addr))
    elif ref.element_type == RENEWABLE:
        agent.add_role(GeneratorMonitoring(behavior, leader_addr))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_scenario_argparser(
        __doc__, default_name_base="central_dispatch_withlosses"
    )
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
