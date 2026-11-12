"""One benchmark run: load prepared network → cut window → mango world → algorithm → outputs.

Usage (after ``esb-prepare --network lower_saxony_2025``)::

    esb-run --network lower_saxony_2025 --algorithm consensus --topology ring --start 2025-01-14

Algorithm interface (audit 2026-11-12 Q-A4, decided by Jonas: module convention): every
distributed algorithm is one ESB module in :data:`ALGORITHMS` with
``setup(params, network, full, args) -> (behavior, agents, routes) -> outcome`` that adds its mango
roles to the world's agents and returns ``outcome() -> outputs.Outcome``; its parameters are the
``[<algorithm>]`` section of ``resources/tuned_parameters.toml`` (``--tuning``, ``--set``); if it
can be tuned it also has ``sweep_grid``, ``SWEEP_RUN`` and ``VARIANT`` (``esb-tune``). Control
messages (start, stop, reports) are marked with DRO's ``ControlMessage`` and never lost.
Central dispatch is the one case without a world.
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
from distributed_resource_optimization import StartCoordinatedDistributedOptimization
from mango import DISCRETE_EVENT
from mango_energy_environments import (
    LOAD,
    STORAGE,
    CopperPlateBehavior,
    Network,
    load_network,
)

from . import consensus, datasets, dummy
from .central import solve_central
from .outputs import RENEWABLES_COVER, run_dir, write_run
from .parameters import TUNINGS, has_parameters, parameters
from .problem import predispatched
from .world import LEADER, TOPOLOGIES, build_world

#: name → algorithm module (None: central dispatch, no world). Add a new algorithm here.
ALGORITHMS = {"central": None, "consensus": consensus, "dummy": dummy}


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


async def _simulate(network, args, add_roles, trace=None):
    """Run the mango world with the roles *add_roles* adds; returns (behavior, messages, outcome);
    the algorithm's counts join *messages*."""
    world, behavior, agents, communication, routes = build_world(
        network,
        args.topology,
        args.loss_rate,
        args.delay_s,
        args.seed,
        trace,
    )
    outcome = add_roles(behavior, agents, routes)
    async with world:
        # kick-off is local (not sent over the lossy network, not counted)
        agents[LEADER].handle_message(StartCoordinatedDistributedOptimization(None), {})
        while await world.step(DISCRETE_EVENT) is not None:
            pass
    result = outcome()
    messages = {
        "sent": communication.sent,
        "delivered": world.messages_delivered_count,
        "lost": communication.lost,
        **result.counts,
    }
    return behavior, messages, result


def _value(text: str):
    """A ``--set`` value: TOML syntax (numbers, true/false), else the plain string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _parameters(args, network_name: str) -> tuple[dict, str]:
    """The algorithm's parameters and the run-folder variant (tuning route + a short hash of the
    ``--set`` overrides, so settings do not overwrite each other); ({}, "") without a section."""
    if not has_parameters(args.algorithm):
        if args.set or args.tuning:
            raise SystemExit(f"{args.algorithm} has no parameters (--set/--tuning)")
        return {}, ""
    if bad := [kv for kv in args.set if "=" not in kv]:
        raise SystemExit(f"--set needs KEY=VALUE, got {bad}")
    overrides = {k: _value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    params = parameters(
        args.algorithm, network_name, args.topology, args.tuning, overrides
    )
    variant = params["tuning"]
    if args.set:
        digest = hashlib.sha1(" ".join(sorted(args.set)).encode()).hexdigest()
        variant += f"_set{digest[:6]}"
    return params, variant


async def run(args: argparse.Namespace) -> Path:
    network_file = datasets.prepared_path(args.network)
    network = full = window(load_network(network_file), args.start, args.hours)
    if not args.storage:
        network = without_storage(network)
    inputs = {**vars(args), "start": str(network.time_index[0])}
    began = time.perf_counter()
    params, variant = _parameters(args, network_file.stem)
    if params:
        inputs[args.algorithm] = params
    # MWh charged + discharged in the same hour; only central knows it
    simultaneous = sim_time_s = iteration_trace = None
    covered = no_dispatch(network)
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
    no_messages = {"sent": 0, "delivered": 0, "lost": 0}
    if covered is not None:
        behavior = CopperPlateBehavior(network)
        behavior.schedules = covered
        stop_reason, iterations, messages = RENEWABLES_COVER, 0, no_messages
    elif ALGORITHMS[args.algorithm] is None:  # central
        behavior = CopperPlateBehavior(network)
        behavior.schedules, stop_reason, simultaneous = solve_central(
            network, args.grid_exchange
        )
        messages, iterations = no_messages, 1
    else:
        trace = None
        if args.trace:
            out.mkdir(parents=True, exist_ok=True)
            trace = out / "mango_trace.jsonl"
        add_roles = ALGORITHMS[args.algorithm].setup(params, network, full, args)
        behavior, messages, result = await _simulate(network, args, add_roles, trace)
        stop_reason, iterations = result.stop_reason, result.iterations
        sim_time_s, iteration_trace = result.sim_time_s, result.trace
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
        grid_exchange=args.grid_exchange,
        storage_simultaneous_mwh=simultaneous,
        sim_time_s=sim_time_s,
        iteration_trace=iteration_trace,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--network",
        default="toy",
        help="dataset name (its resources/prepared/<name>.nc) or a prepared .nc file",
    )
    # default central: the reference optimum (was the dummy test fixture; audit Q-A12a)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="central")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="grid")
    parser.add_argument(
        "--loss-rate",
        type=float,
        default=0.0,
        help="fraction 0–1 per hop of the algorithm messages (control messages always arrive)",
    )
    parser.add_argument("--delay-s", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grid-exchange",
        action="store_true",
        help="import/export at the grid connection after the run (islanded default; was --slack)",
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
        choices=TUNINGS,
        help="the algorithm's parameter set (default: manual, else sweep, else paper + warning)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an algorithm parameter (resources/tuned_parameters.toml)",
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
