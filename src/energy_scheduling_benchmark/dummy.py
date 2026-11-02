"""Dummy "algorithm" for end-to-end runs: merit order in the leader, over the DRO carrier.

A test fixture for the benchmark plumbing, not a benchmarked algorithm. The leader asks every
agent for its offer (cost, p_min/p_max over the horizon), waits up to a timeout, then per hour
puts every unit at p_min and fills the rest of the demand cheapest-c1-first up to p_max
(c2 only counts in the cost). Storage stays idle. Each schedule goes back as the participant's
``OptimizationFinishedMessage``; lost messages simply leave a unit without a schedule.
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
    OptimizationFinishedMessage,
)
from mango_energy_environments.environments.scheduling import STORAGE


@dataclass
class Offer:
    kind: str
    c1: float
    p_min: np.ndarray
    p_max: np.ndarray


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
    def __init__(self, demand: np.ndarray, timeout_s: float) -> None:
        self.demand = demand
        self.timeout_s = timeout_s
        self.stop_reason: str | None = None

    async def start_optimization(self, carrier, message_data, meta):
        futures = [
            carrier.send_awaitable(OfferRequest(), a) for a in carrier.others("leader")
        ]
        await carrier.race(asyncio.wait(futures), carrier.sleep(self.timeout_s))
        self.stop_reason = "complete" if all(f.done() for f in futures) else "timeout"
        offers = [f.result() if f.done() else None for f in futures]
        return merit_order(offers, self.demand)


class DummyUnit(DistributedAlgorithm):
    """Replies its offer and sets the schedule it gets; ``behavior=None``: no component (slack)."""

    def __init__(self, behavior, aid: str) -> None:
        self.behavior = behavior
        self.aid = aid

    async def on_exchange_message(self, carrier, message_data, meta):
        if isinstance(message_data, OptimizationFinishedMessage):
            if message_data.result is not None:
                self.behavior.act(self.aid, "set_schedule", message_data.result)
            return
        offer = None
        if self.behavior is not None:
            observe = self.behavior.observe
            statics = observe(self.aid, "statics")
            offer = Offer(
                statics["kind"],
                statics["cost"][0],
                observe(self.aid, "p_min"),
                observe(self.aid, "p_max"),
            )
        carrier.reply_to_other(offer, meta)
