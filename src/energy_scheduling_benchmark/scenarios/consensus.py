"""Distributed economic-dispatch consensus scenario.

Python port of ``EnergySchedulingBenchmark.jl/scenario/consensus_scenario.jl``.

Flow
----
1. Load agents observe their ``max_active_power`` whenever their
   timeseries fires and send a ``PowerLoadInfo`` to the leader.
2. The leader (carrying :class:`PowerLoadAggregator`) accumulates the
   per-time-step load values.  When all loads have reported, it kicks off
   an averaging-consensus run on the generator agents by sending them an
   initial :class:`AveragingConsensusMessage` with the target demand.
3. Generator agents host a :class:`DistributedOptimizationRole` with a
   :class:`LinearCostEconomicDispatchConsensusActor`.  When the consensus
   finishes, each generator regulates itself to its share of the total.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

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

from distributed_resource_optimization import (
    AveragingConsensusMessage,
    LinearCostEconomicDispatchConsensusActor,
    create_averaging_consensus_participant,
)
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
)

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    THERMAL,
    ComponentRef,
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
    time: datetime


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
    """Collects per-time-step load values and fires the consensus when complete."""

    def __init__(self, number_loads: int, trigger) -> None:
        super().__init__()
        self.number_loads = number_loads
        self._trigger = trigger
        self.demand_map: dict[Any, list[float]] = {}
        self.target: float = 0.0

    def setup(self) -> None:
        self.context.subscribe_message(
            self,
            self._handle_load_info,
            lambda c, m: isinstance(c, PowerLoadInfo),
        )

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        bucket = self.demand_map.setdefault(message.time, [])
        bucket.append(float(message.power_load))

        if len(bucket) == self.number_loads:
            total = float(sum(bucket))
            self.target = total
            logger.info("Starting consensus (target=%.2f MW)", total)
            initial = AveragingConsensusMessage(
                lam=np.array([10.0]),
                k=0,
                data=np.array([total]),
                initial=True,
            )
            asyncio.create_task(self.context.send_message(initial, self._trigger))


class GeneratorMaxPowerTracker(Role):
    """Refreshes the consensus actor's ``p_max`` when the PyPSA timeseries ticks.

    Mirrors the Julia ``on_agent_event(role::DistributedOptimizationRole, ...)``
    method that updates ``role.algorithm.actor.P_max``.
    """

    def __init__(
        self,
        behavior: PyPSABehavior,
        optimization_role: DistributedOptimizationRole,
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self._opt_role = optimization_role

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        pmax = self._behavior.observe(self.context.aid, "max_active_power")
        if pmax is None:
            return
        actor = self._opt_role.algorithm.actor
        actor.p_max = float(pmax)


# ---------------------------------------------------------------------------
# Consensus-finished callback
# ---------------------------------------------------------------------------


def _make_finish_callback(behavior: PyPSABehavior):
    """Build the ``(algorithm, carrier) -> None`` hook that applies the result."""

    def handle_consensus_finished(algorithm, carrier) -> None:
        role = carrier._parent
        aid = role.context.aid
        p = float(np.asarray(algorithm.actor.P).ravel()[0])
        behavior.act(aid, "regulate", p)
        logger.info("Consensus finished for %s: P=%.3f MW", aid, p)

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
    com_sim = SimpleCommunicationSimulation(
        default_delay_s=delay_s, loss_percent=loss_percent
    )
    world = create_world(
        start_time=0.0, communication_sim=com_sim, environment=environment
    )

    finish_callback = _make_finish_callback(behavior)

    # -- Generator agents --
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE])
    n_gens = len(gen_refs)

    gen_agents: list[RoleAgent] = []
    for ref in gen_refs:
        statics = behavior._dataframe_for(ref.element_type).loc[ref.component_id]
        cost = float(statics.get("marginal_cost", 0.0))
        p_max = float(statics.get("p_nom", 0.0))

        actor = LinearCostEconomicDispatchConsensusActor(
            cost=cost, p_max=p_max, n_guess=n_gens, rho=0.03
        )
        participant = create_averaging_consensus_participant(
            finish_callback=finish_callback,
            consensus_actor=actor,
            max_iter=200,
            alpha=0.2,
        )
        opt_role = DistributedOptimizationRole(participant)
        tracker = GeneratorMaxPowerTracker(behavior, opt_role)

        agent = agent_composed_of(opt_role, tracker)
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)
        gen_agents.append(agent)

    # Fully-connected consensus topology across all generator agents.
    topology = complete_topology(len(gen_agents))
    auto_assign(topology, gen_agents)

    # -- Load agents (two-pass: register leader first so its addr is known) --
    load_refs = behavior.get_components_by_type([LOAD])

    leader_agent = RoleAgent()
    leader_agent.add_role(
        PowerLoadAggregator(
            number_loads=len(load_refs),
            trigger=gen_agents[0].addr,
        )
    )
    world.register(leader_agent, suggested_aid=load_refs[0].component_id)
    world.environment.install(leader_agent, id=load_refs[0])
    leader_addr = leader_agent.addr

    leader_agent.add_role(PowerLoadMonitoring(behavior=behavior, target=leader_addr))

    for ref in load_refs[1:]:
        agent = agent_composed_of(
            PowerLoadMonitoring(behavior=behavior, target=leader_addr)
        )
        world.register(agent, suggested_aid=ref.component_id)
        world.environment.install(agent, id=ref)

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
        lambda a: _first_actor_P(a),
    )
    record_agent_having(
        world,
        "lam",
        DistributedOptimizationRole,
        lambda a: _first_actor_lam(a),
    )

    # -- Simulate --
    async with world:
        await discrete_step_until(world, simulate_days * 24 * 3600.0)

    # -- Output --
    visualize_results(world, write_to=f"{name_base}-observation.pdf")

    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    # Align target length to P length.
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


def _first_actor_P(agent: RoleAgent) -> float:
    for role in agent.roles:
        if isinstance(role, DistributedOptimizationRole):
            p = np.asarray(role.algorithm.actor.P).ravel()
            return float(p[0]) if p.size else 0.0
    return 0.0


def _first_actor_lam(agent: RoleAgent) -> float:
    for role in agent.roles:
        if isinstance(role, DistributedOptimizationRole):
            lam = np.asarray(role.algorithm._lam).ravel()
            return float(lam[0]) if lam.size else 0.0
    return 0.0


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
            f"{key}:{aid}": [
                _scalar(v) for v in values[:length]
            ]
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
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--loss-percent", type=float, default=0.00005)
    parser.add_argument("--name-base", type=str, default="consensus_withlosses")
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    asyncio.run(
        execute_test_case(
            delay_s=args.delay_s,
            loss_percent=args.loss_percent,
            name_base=args.name_base,
            simulate_days=args.simulate_days,
        )
    )


if __name__ == "__main__":
    main()
