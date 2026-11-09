"""One benchmark run: load prepared network → cut window → mango world → algorithm → outputs.

Usage (after ``esb-prepare --network lower_saxony_2030``)::

    esb-run --network lower_saxony_2030 --topology ring --start 2019-01-14 --hours 168 --slack
"""

from __future__ import annotations

import argparse
import asyncio
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from distributed_resource_optimization import (
    ConsensusCoordinator,
    ConsensusInnovationsAlgorithm,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
    StartCoordinatedDistributedOptimization,
)
from mango import DISCRETE_EVENT
from mango_energy_environments import (
    STORAGE,
    CopperPlateBehavior,
    Network,
    load_network,
)

from . import consensus, datasets
from .central import solve_central
from .dummy import DummyBus, MeritOrderCoordinator
from .outputs import run_dir, write_run
from .world import LEADER, SLACK, TOPOLOGIES, build_world

ALGORITHMS = ("central", "consensus", "dummy")
#: Simulated seconds the dummy leader waits for offers (a ring of ~70 bus agents at 0.01 s per
#: hop needs < 1 s for request + reply).
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


def without_storage(network: Network) -> Network:
    """*network* without its storage units (``--no-storage``)."""
    components = {cid: c for cid, c in network.components.items() if c.kind != STORAGE}
    return Network(network.time_index, components, network.lines)


def _dummy(behavior, agents, routes):
    """Adds the dummy roles; returns ``() -> (stop_reason, iterations)``."""
    coordinator = MeritOrderCoordinator(DUMMY_TIMEOUT_S)
    for node, agent in agents.items():
        if node == LEADER:
            role = CoordinatorRole(coordinator, routes=routes[agent.aid])
        else:
            unit = DummyBus(None if node == SLACK else behavior, agent.aid)
            role = DistributedOptimizationRole(unit, routes=routes[agent.aid])
        agent.add_role(role)
    return lambda: (coordinator.stop_reason, 1)


def _consensus(params: dict, steps: int):
    """Adds the consensus roles (bus agents + leader; the slack takes no part, plan Q6)."""

    def setup(behavior, agents, routes):
        alpha, beta = consensus.steps(params)
        buses = [a for node, a in agents.items() if node not in (LEADER, SLACK)]
        coordinator = ConsensusCoordinator([a.addr for a in buses], np.zeros(steps))
        agents[LEADER].add_role(
            CoordinatorRole(coordinator, routes=routes[agents[LEADER].aid])
        )
        for agent in buses:
            actor = consensus.BusActor(behavior, agent.aid, params["eps_p"])
            algorithm = ConsensusInnovationsAlgorithm(
                lambda alg, carrier: alg.actor.finish(),
                actor,
                alpha,
                beta,
                params["eps_lambda"],
                params["max_iter"],
            )
            agent.add_role(
                DistributedOptimizationRole(algorithm, routes=routes[agent.aid])
            )

        def outcome():
            r = coordinator.result
            return ("stalled", None) if r is None else (r.stop_reason, r.iterations)

        return outcome

    return setup


async def _simulate(network, args, setup):
    """Run the mango world with the roles *setup* adds; returns
    (behavior, messages, stop_reason, iterations)."""
    world, behavior, agents, communication, routes = build_world(
        network, args.topology, args.loss_rate, args.delay_s, args.seed, args.slack
    )
    outcome = setup(behavior, agents, routes)
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
    return behavior, messages, *outcome()


def _value(text: str):
    """A ``--set`` value: TOML syntax (numbers, true/false), else the plain string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


async def run(args: argparse.Namespace) -> Path:
    network_file = datasets.prepared_path(args.network)
    if args.day:
        days = datasets.STANDARD_DAYS.get(network_file.stem)
        if days is None:
            raise SystemExit(
                f"--day: no standard days for {network_file.stem}; use --start"
            )
        args.start, args.hours = days[args.day], 24
    network = window(load_network(network_file), args.start, args.hours)
    if not args.storage:
        network = without_storage(network)
    inputs = {**vars(args), "start": str(network.time_index[0])}
    began = time.perf_counter()
    if args.algorithm == "central":
        behavior = CopperPlateBehavior(network)
        behavior.schedules, stop_reason = solve_central(network, args.slack)
        messages, iterations = {"sent": 0, "delivered": 0, "lost": 0}, 1
    elif args.algorithm == "consensus":
        overrides = dict(kv.split("=", 1) for kv in args.set)
        params = consensus.parameters(
            network_file.stem,
            args.topology,
            args.tuning,
            {k: _value(v) for k, v in overrides.items()},
        )
        inputs["consensus"] = params
        setup = _consensus(params, len(network.time_index))
        behavior, messages, stop_reason, iterations = await _simulate(
            network, args, setup
        )
    else:
        behavior, messages, stop_reason, iterations = await _simulate(
            network, args, _dummy
        )
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
        args.storage,
    )
    write_run(
        out,
        behavior,
        messages=messages,
        inputs=inputs,
        network_file=network_file,
        algorithm=args.algorithm,
        iterations=iterations,
        stop_reason=stop_reason,
        wall_time_s=wall_time_s,
        slack=args.slack,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
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
    when = parser.add_mutually_exclusive_group()
    when.add_argument(
        "--start", help="first snapshot, e.g. 2019-01-14 (default: first)"
    )
    when.add_argument(
        "--day",
        choices=datasets.SEASONS,
        help="standard 24 h day of the network (datasets.STANDARD_DAYS); sets --start, --hours",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument(
        "--no-storage",
        dest="storage",
        action="store_false",
        help="drop the network's storage units",
    )
    parser.add_argument(
        "--tuning",
        choices=consensus.TUNINGS,
        help="consensus parameter set (default: manual if tuned, else paper)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a consensus parameter (resources/tuned_parameters.toml)",
    )
    parser.add_argument("--out", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    print(asyncio.run(run(build_parser().parse_args(argv))))


if __name__ == "__main__":
    main()
