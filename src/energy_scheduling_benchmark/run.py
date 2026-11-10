"""One benchmark run: load prepared network → cut window → mango world → algorithm → outputs.

Usage (after ``esb-prepare --network lower_saxony_2030``)::

    esb-run --network lower_saxony_2030 --topology ring --start 2019-01-14 --hours 168 --slack
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    LOAD,
    STORAGE,
    CopperPlateBehavior,
    Network,
    load_network,
)

from . import consensus, datasets
from .central import solve_central
from .dummy import DummyBus, MeritOrderCoordinator
from .outputs import RENEWABLES_COVER, run_dir, write_run
from .world import LEADER, SLACK, TOPOLOGIES, build_world, predispatched

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


def no_dispatch(network: Network) -> dict[str, np.ndarray] | None:
    """Schedules when the pre-dispatched units alone cover the load in every hour (no ED needed,
    the same for every algorithm; S1 plan D7): those at p_max, other units at p_min, storage
    idle. None if any hour needs dispatch."""
    comps = network.components
    fixed = {
        cid: c.p_max
        for cid, c in comps.items()
        if c.kind != LOAD and predispatched(c.kind, c.cost[0])
    }
    demand = sum(c.p_max for c in comps.values() if c.kind == LOAD)
    if not np.all(sum(fixed.values(), 0.0) >= demand):
        return None
    rest = {
        cid: np.zeros_like(c.p_max) if c.kind == STORAGE else c.p_min
        for cid, c in comps.items()
        if c.kind != LOAD and cid not in fixed
    }
    return fixed | rest


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
                consensus.LAM_MIN,
            )
            agent.add_role(
                DistributedOptimizationRole(algorithm, routes=routes[agent.aid])
            )

        def outcome():
            r = coordinator.result
            return ("stalled", None) if r is None else (r.stop_reason, r.iterations)

        return outcome

    return setup


async def _simulate(network, args, setup, trace=None):
    """Run the mango world with the roles *setup* adds; returns
    (behavior, messages, stop_reason, iterations)."""
    world, behavior, agents, communication, routes = build_world(
        network,
        args.topology,
        args.loss_rate,
        args.delay_s,
        args.seed,
        args.slack,
        trace,
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
    network = window(load_network(network_file), args.start, args.hours)
    if not args.storage:
        network = without_storage(network)
    inputs = {**vars(args), "start": str(network.time_index[0])}
    began = time.perf_counter()
    variant = ""
    setup = _dummy
    simultaneous = (
        None  # MWh charged + discharged in the same hour; only central knows it
    )
    covered = no_dispatch(network)
    if args.algorithm == "central" and covered is None:
        behavior = CopperPlateBehavior(network)
        behavior.schedules, stop_reason, simultaneous = solve_central(
            network, args.slack
        )
        messages, iterations = {"sent": 0, "delivered": 0, "lost": 0}, 1
    elif args.algorithm == "consensus":
        if bad := [kv for kv in args.set if "=" not in kv]:
            raise SystemExit(f"--set needs KEY=VALUE, got {bad}")
        overrides = dict(kv.split("=", 1) for kv in args.set)
        params = consensus.parameters(
            network_file.stem,
            args.topology,
            args.tuning,
            {k: _value(v) for k, v in overrides.items()},
        )
        inputs["consensus"] = params
        # tuning route, plus a short hash of --set overrides, so settings do not overwrite
        variant = params["tuning"]
        if args.set:
            digest = hashlib.sha1(" ".join(sorted(args.set)).encode()).hexdigest()
            variant += f"_set{digest[:6]}"
        setup = _consensus(params, len(network.time_index))
    out = run_dir(
        args.out,
        network_file.stem,
        args.algorithm,
        args.topology,
        args.loss_rate,
        args.seed,
        network.time_index[0],
        args.hours,
        args.storage,
        variant,
    )
    if covered is not None:
        behavior = CopperPlateBehavior(network)
        behavior.schedules = covered
        stop_reason, iterations = RENEWABLES_COVER, 0
        messages = {"sent": 0, "delivered": 0, "lost": 0}
    elif args.algorithm != "central":
        trace = None
        if args.trace:
            out.mkdir(parents=True, exist_ok=True)
            trace = out / "mango_trace.jsonl"
        behavior, messages, stop_reason, iterations = await _simulate(
            network, args, setup, trace
        )
    wall_time_s = time.perf_counter() - began

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
        storage_simultaneous_mwh=simultaneous,
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
    parser.add_argument(
        "--start",
        help="first snapshot, e.g. 2019-01-14 (default: first); standard days: README",
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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="mango trace + HTML viewer in the run folder (debugging; slow, large)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    print(asyncio.run(run(build_parser().parse_args(argv))))


if __name__ == "__main__":
    main()
