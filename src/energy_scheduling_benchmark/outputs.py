"""Run outputs: one folder per run with ``schedule.csv`` and ``meta.json`` (decided Q-B7/Q-B8/Q-B8b).

The cost gap vs. the central optimum and balance summaries are computed by the compare script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mango_energy_environments import LOAD, STORAGE, CopperPlateBehavior

from . import storage
from .world import SLACK, SLACK_IMPORT_PRICE


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
    *,
    messages: dict[str, int],
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
    ``slack`` (the difference it takes, if enabled), ``residual`` = gen + slack − demand and
    ``soc:<unit>`` (storage energy at the end of each hour, MWh).
    ``total_cost`` includes the slack import (also reported separately under ``slack``).
    ``messages``: sent / delivered / lost counts.
    ``inputs``: all CLI args incl. seed and network name; ``network_file`` None for built-in nets.
    """
    network = behavior.network
    units = [cid for cid, c in network.components.items() if c.kind != LOAD]
    schedule = pd.DataFrame(behavior.schedules, index=network.time_index).reindex(
        columns=units
    )
    residual = behavior.balance()
    slack_p = -residual if slack else np.zeros_like(residual)
    if slack:
        schedule[SLACK] = slack_p
    schedule["residual"] = residual + slack_p
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
    slack_import = float(np.maximum(slack_p, 0.0).sum())
    total_cost += SLACK_IMPORT_PRICE * slack_import

    meta = {
        "inputs": inputs,
        "network_sha256": network_file
        and hashlib.sha256(Path(network_file).read_bytes()).hexdigest(),
        "algorithm": algorithm,
        "total_cost": total_cost,
        "slack": {
            "import_mwh": slack_import,
            "export_mwh": float(np.maximum(-slack_p, 0.0).sum()),
            "cost": SLACK_IMPORT_PRICE * slack_import,
        },
        "messages": messages,
        "iterations": iterations,
        "stop_reason": stop_reason,
        "wall_time_s": wall_time_s,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(out_dir / "schedule.csv")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
