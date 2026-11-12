"""Rules of the benchmark problem shared by central dispatch and every algorithm.

Own module (audit 2026-11-12 Q-A12b): they used to live in ``world.py`` (the mango world), so
central dispatch and the datasets imported the world for a problem rule.
"""

from __future__ import annotations

import numpy as np
from mango_energy_environments import LOAD, RENEWABLE, STORAGE, THERMAL, Network

#: Grid import price (€/MWh) of ``--grid-exchange`` runs, export is free (Q3: SDAC harmonised
#: maximum clearing price). Was SLACK_IMPORT_PRICE (renamed with the slack agent's removal,
#: audit 2026-11-12 Q-A6).
GRID_IMPORT_PRICE = 4000.0


def predispatched(kind: str, c1: float) -> bool:
    """Runs at its available power p_max before any algorithm: renewables and free thermal units
    (marginal cost 0 → always first; KH12 eq. 27 / HKW15 §II must-take). Decided 2026-11-10
    (supervisor meeting, `docs/renewables-predispatch-plan.md` D1): no curtailment — a surplus is
    exported by the grid exchange or reported."""
    return kind == RENEWABLE or (kind == THERMAL and c1 <= 0)


def no_dispatch(network: Network) -> dict[str, np.ndarray] | None:
    """Schedules when the pre-dispatched units alone cover the load in every hour (no ED needed,
    the same for every algorithm; S1 plan D7): those at p_max, other units at p_min, storage
    idle. None if any hour needs dispatch. (Was in ``run.py``; audit 2026-11-12 fix 5.)"""
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
