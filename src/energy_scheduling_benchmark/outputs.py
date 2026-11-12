"""Run outputs: one folder per run with ``schedule.csv`` and ``meta.json`` (decided Q-B7/Q-B8/Q-B8b).

The cost gap vs. the central optimum across runs: ``esb-compare`` (``compare.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from mango_energy_environments import LOAD, STORAGE, CopperPlateBehavior

from . import storage
from .problem import GRID_IMPORT_PRICE, predispatched

#: schedule.csv column + meta block of the grid exchange (was "slack", audit 2026-11-12 Q-A6)
GRID_EXCHANGE = "grid_exchange"

#: stop_reason of a run that needed no ED: pre-dispatched units cover the load in every hour
#: (S1 plan D7); a valid result for every algorithm, central included.
RENEWABLES_COVER = "renewables_cover"


@dataclass
class Outcome:
    """What a distributed algorithm reports after its run (algorithm interface, audit
    2026-11-12 Q-A4): *counts* join meta ``messages``, *sim_time_s* = simulated time of the stop
    (None without one), *trace* = per-iteration ``trace.csv`` (None for non-iterative ones)."""

    stop_reason: str
    iterations: int | None
    counts: dict = field(default_factory=dict)
    sim_time_s: float | None = None
    trace: pd.DataFrame | None = None


def run_dir(
    root: Path | str,
    network: str,
    algorithm: str,
    topology: str,
    loss_rate: float,
    seed: int,
    start: pd.Timestamp,
    hours: int,
    storage: bool,
    variant: str = "",
) -> Path:
    """``<root>/<network>/<algorithm>/<topology>_loss<rate>_seed<seed>_from<start>_<hours>h_
    [no]storage[_<variant>]`` (reruns overwrite); *variant* tells algorithm settings apart."""
    name = (
        f"{topology}_loss{loss_rate:g}_seed{seed}_from{start:%Y%m%dT%H}_{hours}h_"
        f"{'' if storage else 'no'}storage"
    ) + (f"_{variant}" if variant else "")
    return Path(root) / network / algorithm / name


def write_run(
    out_dir: Path | str,
    behavior: CopperPlateBehavior,
    *,
    messages: dict[str, int],
    inputs: dict,
    network_file: Path | None,
    algorithm: str,
    iterations: int,
    stop_reason: str,
    wall_time_s: float,
    grid_exchange: bool = False,
    storage_simultaneous_mwh: float | None = None,
    sim_time_s: float | None = None,
    iteration_trace: pd.DataFrame | None = None,
) -> None:
    """Write ``schedule.csv`` and ``meta.json`` (and ``trace.csv``) into *out_dir*.

    ``sim_time_s``: simulated time of the coordinator's stop = time to solution of a distributed
    run (``null`` for central, a covered window or a run without a stop). Communication only:
    mango's simulation has no computation-time model, agents compute in zero simulated time
    (audit 2026-11-12 Q-A3). ``trace.csv`` (*iteration_trace*, distributed iterative runs): one
    row per iteration, see ``consensus.IterationTrace.frame``.

    ``schedule.csv``: time × non-load unit MW (empty = the unit never set a schedule), plus
    ``grid_exchange`` (import > 0 / export < 0 at the grid connection: it takes the remaining
    imbalance after the run, if enabled), ``residual`` = gen + grid exchange − demand and
    ``soc:<unit>`` (storage energy at the end of each hour, MWh).
    ``total_cost`` includes the grid import (also reported separately under ``grid_exchange``,
    with ``import_share`` = import / total demand); the costs and all ``grid_exchange`` fields are
    ``null`` (CSV ``grid_exchange`` empty) when the central solve is not optimal. Distributed runs always
    report their cost, also over only the units that got a schedule; ``imbalance`` = shortfall /
    surplus MWh of the final ``residual`` (islanded: a renewable surplus stays there, it is never
    curtailed); ``renewable_available_mwh`` = energy of the pre-dispatched units;
    ``storage_simultaneous_mwh`` = MWh charged and discharged in the same hour (central only,
    expected 0; ``null`` elsewhere).
    ``messages``: sent / delivered / lost counts.
    ``inputs``: all CLI args incl. seed and network name; ``network_file``: the prepared ``.nc``
    the run read (its SHA-256 goes into meta).
    """
    network = behavior.network
    units = [cid for cid, c in network.components.items() if c.kind != LOAD]
    schedule = pd.DataFrame(behavior.schedules, index=network.time_index).reindex(
        columns=units
    )
    failed = algorithm == "central" and stop_reason not in ("optimal", RENEWABLES_COVER)
    residual = behavior.balance()
    grid_p = -residual if grid_exchange and not failed else np.zeros_like(residual)
    if grid_exchange:
        schedule[GRID_EXCHANGE] = np.nan if failed else grid_p
    residual = residual + grid_p
    schedule["residual"] = residual
    for cid, c in network.components.items():
        if c.kind == STORAGE:
            p = behavior.schedules.get(cid)
            schedule[f"soc:{cid}"] = np.nan if p is None else storage.soc(c, p)
    schedule.index.name = "time"

    total_cost = 0.0
    for cid, p in behavior.schedules.items():
        c = network.components[cid]
        if c.kind == STORAGE:
            total_cost += storage.cost(c, p)
        else:
            total_cost += float(np.sum(c.cost[0] * p + c.cost[1] * p**2))
    demand = sum(
        float(c.p_max.sum()) for c in network.components.values() if c.kind == LOAD
    )
    grid_import = float(np.maximum(grid_p, 0.0).sum())
    grid_meta = {
        "import_mwh": grid_import,
        "export_mwh": float(np.maximum(-grid_p, 0.0).sum()),
        "import_share": grid_import / demand if demand else 0.0,
        "cost": GRID_IMPORT_PRICE * grid_import,
    }
    total_cost += grid_meta["cost"]
    if failed or not (behavior.schedules or grid_exchange):
        # no valid result (failed solve, or no unit scheduled and no grid import): 0 would
        # look like the cheapest run
        total_cost = None
        grid_meta = dict.fromkeys(grid_meta)

    meta = {
        "inputs": inputs,
        "network_sha256": network_file
        and hashlib.sha256(Path(network_file).read_bytes()).hexdigest(),
        "algorithm": algorithm,
        "total_cost": total_cost,
        GRID_EXCHANGE: grid_meta,
        "imbalance": {
            "shortfall_mwh": float(np.maximum(-residual, 0.0).sum()),
            "surplus_mwh": float(np.maximum(residual, 0.0).sum()),
        },
        "storage_simultaneous_mwh": storage_simultaneous_mwh,
        "renewable_available_mwh": sum(
            float(c.p_max.sum())
            for c in network.components.values()
            if c.kind != LOAD and predispatched(c.kind, c.cost[0])
        ),
        "messages": messages,
        "iterations": iterations,
        "stop_reason": stop_reason,
        "sim_time_s": sim_time_s,
        "wall_time_s": wall_time_s,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_dir / "schedule.csv")
    if iteration_trace is not None:
        iteration_trace.to_csv(out_dir / "trace.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
