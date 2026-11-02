"""Central economic dispatch: the reference optimum (plan: ``docs/central-dispatch-plan.md``).

One cvxpy model over the whole window: units within p_min … p_max at c₁p + c₂p², storage per
:mod:`.storage`, optional slack (import at :data:`SLACK_IMPORT_PRICE`, export free), and per hour
Σ injections = demand. No tie-break terms (Q-C4). HiGHS solves the LP, Clarabel the QP (HiGHS' QP
fails or stalls on PyPSA-Eur networks).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from mango_energy_environments import CopperPlateNetwork
from mango_energy_environments.environments.scheduling import LOAD, STORAGE

from . import storage
from .world import SLACK_IMPORT_PRICE


def solve_central(
    network: CopperPlateNetwork, slack: bool = False
) -> tuple[dict[str, np.ndarray], str]:
    """Return ``({component id: MW per hour}, solver status)``; no schedules unless optimal."""
    steps = len(network.time_index)
    demand = sum(c.p_max for c in network.components.values() if c.kind == LOAD)
    injection, cost, cons, schedules = 0, 0, [], {}
    for cid, c in network.components.items():
        if c.kind == LOAD:
            continue
        if c.kind == STORAGE:
            p_c, p_d = cp.Variable(steps), cp.Variable(steps)
            cons += storage.constraints(c, p_c, p_d)
            p = p_d - p_c
            cost += c.cost[0] * cp.sum(p_d)
        else:
            p = cp.Variable(steps)
            cons += [p >= c.p_min, p <= c.p_max]
            cost += c.cost[0] * cp.sum(p) + c.cost[1] * cp.sum_squares(p)
        injection += p
        schedules[cid] = p
    if slack:
        imp, exp = cp.Variable(steps, nonneg=True), cp.Variable(steps, nonneg=True)
        injection += imp - exp
        cost += SLACK_IMPORT_PRICE * cp.sum(imp)
    problem = cp.Problem(cp.Minimize(cost), [*cons, injection == demand])
    if any(c.cost[1] for c in network.components.values()):
        # Clarabel's equilibration mis-scales the 4000 €/MWh import next to ~10–100 €/MWh units:
        # with it, base_s_1 (168 h) stops 2e-5 above the islanded optimum while reporting optimal.
        problem.solve(solver=cp.CLARABEL, equilibrate_enable=False)
    else:
        problem.solve(solver=cp.HIGHS)
    if problem.status != cp.OPTIMAL:
        return {}, problem.status
    return {cid: np.asarray(p.value) for cid, p in schedules.items()}, problem.status
