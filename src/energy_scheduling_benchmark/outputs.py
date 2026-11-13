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


class IterationTrace:
    """Per-iteration convergence trace of one distributed run (``trace.csv``; audit 2026-11-12
    Q-A3, decided by Jonas: common columns + the algorithm's, recorded in the ESB actors, always
    on). Generic for every iterative algorithm (moved out of ``consensus.py``, audit fix 5).

    Iteration k = the state after k updates (k = 0: the start). Every bus actor adds its mismatch,
    its current cost and its copy of the shared *vector* the algorithm iterates on (consensus: λ);
    a row is kept only once all *n* actors reached it. On sparse topologies the stop reaches the
    actors at different times, so they end 1–3 iterations apart and the final schedule can lie a
    little past the last row (toy grid: 2e-6 relative cost; Jonas 2026-11-13). ``sim_time_s`` = simulated time at which
    the last actor got there (communication only: mango has no computation-time model)."""

    def __init__(self, n: int, now, vector: str = "lambda") -> None:
        self.n, self.now, self.vector = n, now, vector
        # k -> [count, Σ mismatch, Σ vector, min vector, max vector, Σ cost, latest time]
        self.rows: dict[int, list] = {}

    def add(
        self, k: int, mismatch: np.ndarray, cost: float, vector: np.ndarray
    ) -> None:
        row = self.rows.get(k)
        if row is None:
            self.rows[k] = [
                1,
                mismatch.copy(),
                vector.copy(),
                vector.copy(),
                vector.copy(),
                cost,
                self.now(),
            ]
            return
        row[0] += 1
        row[1] += mismatch
        row[2] += vector
        np.minimum(row[3], vector, out=row[3])
        np.maximum(row[4], vector, out=row[4])
        row[5] += cost
        row[6] = max(row[6], self.now())

    def frame(self) -> pd.DataFrame:
        """iteration, sim_time_s, imbalance_mwh (Σ_h |Σ_j mismatch|), cost (€, units without the
        grid exchange), <vector>_mean, <vector>_spread (max_h of max_j − min_j)."""
        return pd.DataFrame(
            [
                {
                    "iteration": k,
                    "sim_time_s": t,
                    "imbalance_mwh": float(np.abs(mis).sum()),
                    "cost": cost,
                    f"{self.vector}_mean": float(v_sum.mean() / self.n),
                    f"{self.vector}_spread": float((hi - lo).max()),
                }
                for k, (count, mis, v_sum, lo, hi, cost, t) in sorted(self.rows.items())
                if count == self.n
            ]
        )


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
    row per iteration, see :meth:`IterationTrace.frame`.

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
