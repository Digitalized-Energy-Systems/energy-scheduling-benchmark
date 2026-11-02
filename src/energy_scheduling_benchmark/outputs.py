"""Run outputs: one folder per run with ``schedule.csv`` and ``meta.json`` (decided Q-B7/Q-B8/Q-B8b).

The cost gap vs. the central optimum and balance summaries are computed by the compare script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mango_energy_environments import CopperPlateBehavior
from mango_energy_environments.environments.scheduling import LOAD

from .world import SLACK


def run_dir(
    root: Path | str,
    network: str,
    algorithm: str,
    topology: str,
    loss_rate: float,
    seed: int,
    start: pd.Timestamp,
    hours: int,
) -> Path:
    """``<root>/<network>/<algorithm>/<topology>_loss<rate>_seed<seed>_from<start>_<hours>h``
    (reruns overwrite)."""
    name = f"{topology}_loss{loss_rate:g}_seed{seed}_from{start:%Y%m%dT%H}_{hours}h"
    return Path(root) / network / algorithm / name


def write_run(
    out_dir: Path | str,
    behavior: CopperPlateBehavior,
    world,
    communication,
    *,
    inputs: dict,
    network_file: Path | None,
    algorithm: str,
    iterations: int,
    stop_reason: str,
    wall_time_s: float,
    slack: bool = False,
) -> None:
    """Write ``schedule.csv`` and ``meta.json`` into *out_dir*.

    ``schedule.csv``: time × non-load unit MW (empty = the unit never set a schedule), plus
    ``slack`` (the difference it takes, if enabled) and ``residual`` = gen + slack − demand.
    ``inputs``: all CLI args incl. seed and network name; ``network_file`` None for built-in nets.
    """
    network = behavior.network
    units = [cid for cid, c in network.components.items() if c.kind != LOAD]
    schedule = pd.DataFrame(behavior.schedules, index=network.time_index).reindex(
        columns=units
    )
    residual = behavior.balance()
    if slack:
        schedule[SLACK] = -residual
        residual = residual + schedule[SLACK]
    schedule["residual"] = residual
    schedule.index.name = "time"

    total_cost = 0.0
    for cid, p in behavior.schedules.items():
        c1, c2 = network.components[cid].cost
        total_cost += float(np.sum(c1 * p + c2 * p**2))

    sent, lost = communication.sent, communication.lost
    meta = {
        "inputs": inputs,
        "network_sha256": network_file
        and hashlib.sha256(Path(network_file).read_bytes()).hexdigest(),
        "algorithm": algorithm,
        "total_cost": total_cost,
        "messages": {
            "sent": sent,
            "delivered": world.messages_delivered_count,
            "lost": lost,
        },
        "iterations": iterations,
        "stop_reason": stop_reason,
        "wall_time_s": wall_time_s,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_dir / "schedule.csv")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
