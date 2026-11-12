"""Dummy "algorithm" for end-to-end runs: merit order in the leader, over the DRO carrier.

A test fixture for the benchmark plumbing, not a benchmarked algorithm. The leader asks every
bus agent for its offer (per unit: cost, p_min/p_max over the horizon; plus the bus load), waits
up to a timeout, then per hour puts every unit at p_min and fills the rest of the received demand
cheapest-c1-first up to p_max (c2 only counts in the cost). Storage stays idle. Each bus's
``{component: schedule}`` goes back as its ``OptimizationFinishedMessage``; a lost offer leaves
that bus's units without a schedule and its load out of the demand. Pre-dispatched units are not
offered: the bus sets them to p_max itself and offers its net load (S1 plan D3).
``stop_reason`` "complete" only means every offer arrived: schedules lost on the way back
still show up as empty columns.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from distributed_resource_optimization.algorithm.core import (
    Coordinator,
    DistributedAlgorithm,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
    OptimizationFinishedMessage,
)
from mango_energy_environments import LOAD, STORAGE

from .outputs import Outcome
from .problem import predispatched
from .world import LEADER, SLACK

#: Simulated seconds the leader waits for offers (a ring of ~70 bus agents at 0.01 s per hop
#: needs < 1 s for request + reply).
TIMEOUT_S = 60.0


def setup(params: dict, network, full, args):
    """Benchmark wiring (algorithm interface, ``run.ALGORITHMS``): adds the dummy roles and
    returns ``outcome() -> Outcome``."""

    def add_roles(behavior, agents, routes):
        coordinator = MeritOrderCoordinator(TIMEOUT_S)
        for node, agent in agents.items():
            if node == LEADER:
                role = CoordinatorRole(coordinator, routes=routes[agent.aid])
            else:
                unit = DummyBus(None if node == SLACK else behavior, agent.aid)
                role = DistributedOptimizationRole(unit, routes=routes[agent.aid])
            agent.add_role(role)
        return lambda: Outcome(coordinator.stop_reason, 1, sim_time_s=coordinator.time)

    return add_roles


@dataclass
class Offer:
    kind: str
    c1: float
    p_min: np.ndarray
    p_max: np.ndarray


@dataclass
class BusOffer:
    units: dict[str, Offer]
    load: np.ndarray  # MW per step, sum of the bus's loads (zeros without one)


class OfferRequest:
    pass


def merit_order(
    offers: list[Offer | None], demand: np.ndarray
) -> list[np.ndarray | None]:
    """Schedule per offer (None stays None); storage idle."""
    schedules = [
        None
        if o is None
        else np.zeros_like(demand)
        if o.kind == STORAGE
        else o.p_min.copy()
        for o in offers
    ]
    remaining = demand - sum(s for s in schedules if s is not None)
    pool = [i for i, o in enumerate(offers) if o is not None and o.kind != STORAGE]
    for i in sorted(pool, key=lambda i: offers[i].c1):
        add = np.clip(remaining, 0.0, offers[i].p_max - offers[i].p_min)
        schedules[i] = schedules[i] + add
        remaining = remaining - add
    return schedules


class MeritOrderCoordinator(Coordinator):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.stop_reason: str | None = None
        self.time: float | None = None  # simulated time when the offers are in (Q-A3)

    async def start_optimization(self, carrier, message_data, meta):
        futures = [
            carrier.send_awaitable(OfferRequest(), a) for a in carrier.others("leader")
        ]
        # Kept (decision 2026-11-09, TODO 14c): same as mango's gather(timeout=), but DRO's
        # Carrier.gather has no timeout and mango's cannot route multi-hop; a fixture is not worth either.
        await carrier.race(asyncio.wait(futures), carrier.sleep(self.timeout_s))
        self.stop_reason = "complete" if all(f.done() for f in futures) else "timeout"
        self.time = carrier.now()
        buses = [f.result() if f.done() else None for f in futures]
        received = [b for b in buses if b is not None]
        demand = sum(b.load for b in received)
        units = [(i, cid) for i, b in enumerate(buses) if b for cid in b.units]
        flat = merit_order([buses[i].units[cid] for i, cid in units], demand)
        result = [None if b is None else {} for b in buses]
        for (i, cid), schedule in zip(units, flat):
            result[i][cid] = schedule
        return result


class DummyBus(DistributedAlgorithm):
    """Replies its bus offer and sets the schedules it gets; ``behavior=None``: no bus (slack)."""

    def __init__(self, behavior, aid: str) -> None:
        self.behavior = behavior
        self.aid = aid

    async def on_exchange_message(self, carrier, message_data, meta):
        if isinstance(message_data, OptimizationFinishedMessage):
            for cid, schedule in (message_data.result or {}).items():
                self.behavior.act(self.aid, "set_schedule", schedule, cid)
            return
        offer = None
        if self.behavior is not None:
            observe = self.behavior.observe
            cids = observe(self.aid, "components")
            units, load = {}, np.zeros_like(observe(self.aid, "p_max", cids[0]))
            for cid in cids:
                statics = observe(self.aid, "statics", cid)
                if statics["kind"] == LOAD:
                    load = load + observe(self.aid, "demand", cid)
                elif predispatched(statics["kind"], statics["cost"][0]):
                    p_max = observe(self.aid, "p_max", cid)
                    self.behavior.act(self.aid, "set_schedule", p_max, cid)
                    load = load - p_max
                else:
                    units[cid] = Offer(
                        statics["kind"],
                        statics["cost"][0],
                        observe(self.aid, "p_min", cid),
                        observe(self.aid, "p_max", cid),
                    )
            offer = BusOffer(units, load)
        carrier.reply_to_other(offer, meta)
