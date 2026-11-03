"""One benchmark run: load prepared network → cut window → mango world → algorithm → outputs.

Usage (after ``esb-prepare --network lower_saxony_2030``)::

    esb-run --network lower_saxony_2030 --topology ring --start 2019-01-14 --hours 168 --slack
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
)
from mango import DISCRETE_EVENT
from mango_energy_environments import LOAD, CopperPlateBehavior, Network, load_network

from . import datasets
from .central import solve_central
from .dummy import DummyUnit, MeritOrderCoordinator
from .outputs import run_dir, write_run
from .world import LEADER, SLACK, TOPOLOGIES, build_world

ALGORITHMS = ("central", "dummy")
#: Simulated seconds the dummy leader waits for offers (a ring of ~430 agents at 0.01 s per hop
#: needs ~5 s for request + reply).
DUMMY_TIMEOUT_S = 60.0


def window(network: Network, start, hours: int) -> Network:
    """Cut *network* to *hours* snapshots from *start* (a snapshot; None = the first)."""
    try:
        i = 0 if start is None else network.time_index.get_loc(pd.Timestamp(start))
    except KeyError:
        raise ValueError(f"--start {start} is not a snapshot of the network") from None
    if i + hours > len(network.time_index):
        raise ValueError(
            f"window of {hours} h from {start} exceeds the network horizon"
        )
    s = slice(i, i + hours)
    components = {
        cid: replace(c, p_min=c.p_min[s], p_max=c.p_max[s])
        for cid, c in network.components.items()
    }
    return Network(network.time_index[s], components, network.lines)


async def _run_dummy(network, args):
    """Returns (behavior, messages, stop_reason)."""
    world, behavior, agents, communication, routes = build_world(
        network, args.topology, args.loss_rate, args.delay_s, args.seed, args.slack
    )
    demand = sum(c.p_max for c in network.components.values() if c.kind == LOAD)
    coordinator = MeritOrderCoordinator(demand, DUMMY_TIMEOUT_S)
    for node, agent in agents.items():
        if node == LEADER:
            role = CoordinatorRole(coordinator, routes=routes[agent.aid])
        else:
            unit = DummyUnit(None if node == SLACK else behavior, agent.aid)
            role = DistributedOptimizationRole(unit, routes=routes[agent.aid])
        agent.add_role(role)
    async with world:
        # kick-off is local (not sent over the lossy network, not counted)
        agents[LEADER].handle_message(StartCoordinatedDistributedOptimization(None), {})
        while await world.step(DISCRETE_EVENT) is not None:
            pass
    messages = {
        "sent": communication.sent,
        "delivered": world.messages_delivered_count,
        "lost": communication.lost,
    }
    return behavior, messages, coordinator.stop_reason


async def run(args: argparse.Namespace) -> Path:
    network_file = datasets.prepared_path(args.network)
    network = window(load_network(network_file), args.start, args.hours)
    began = time.perf_counter()
    if args.algorithm == "central":
        behavior = CopperPlateBehavior(network)
        behavior.schedules, stop_reason = solve_central(network, args.slack)
        messages = {"sent": 0, "delivered": 0, "lost": 0}
    else:
        behavior, messages, stop_reason = await _run_dummy(network, args)
    wall_time_s = time.perf_counter() - began

    start = network.time_index[0]
    out = run_dir(
        args.out,
        network_file.stem,
        args.algorithm,
        args.topology,
        args.loss_rate,
        args.seed,
        start,
        args.hours,
    )
    write_run(
        out,
        behavior,
        messages=messages,
        inputs={**vars(args), "start": str(start)},
        network_file=network_file,
        algorithm=args.algorithm,
        iterations=1,
        stop_reason=stop_reason,
        wall_time_s=wall_time_s,
        slack=args.slack,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--network",
        default="toy",
        help="dataset name (its resources/prepared/<name>.nc) or a prepared .nc file",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="dummy")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="grid")
    parser.add_argument(
        "--loss-rate", type=float, default=0.0, help="fraction 0–1 per hop"
    )
    parser.add_argument("--delay-s", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--slack", action="store_true", help="grid import (islanded default)"
    )
    parser.add_argument(
        "--start", help="first snapshot, e.g. 2019-01-14 (default: first)"
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out", default="outputs")
    print(asyncio.run(run(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
