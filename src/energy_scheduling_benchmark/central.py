"""Central economic dispatch: the reference optimum (plan: ``docs/central-dispatch-plan.md``).

One cvxpy model over the whole window: units within p_min … p_max at c₁p + c₂p², storage per
:mod:`.storage`, optional slack (import at :data:`SLACK_IMPORT_PRICE`, export free), and per hour
Σ injections = demand. Pre-dispatched units (:func:`.world.predispatched`) are fixed at p_max;
islanded, a free surplus variable takes what nobody can absorb (reported, not curtailed; S1 plan
D6). No tie-break terms (Q-C4). Charging and discharging in the same hour is never optimal with the
ageing cost on discharge and η < 1 (Zhu et al. 2025 Thm 1; Xu et al. 2018 §IV), so it is only
checked and reported, not forbidden by binaries (S2, `docs/storage-cost-plan.md` Q-S4). HiGHS solves the LP, Clarabel the QP (HiGHS' QP
fails or stalls on PyPSA-Eur networks).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from mango_energy_environments import LOAD, STORAGE, Network

from . import storage
from .world import SLACK_IMPORT_PRICE, predispatched


def solve_central(
    network: Network, slack: bool = False
) -> tuple[dict[str, np.ndarray], str, float | None]:
    """Return ``({component id: MW per hour}, solver status, simultaneous MWh)``; no schedules
    (and None) unless optimal. Simultaneous = Σ over storage and hours of min(charge, discharge)."""
    steps = len(network.time_index)
    demand = sum(c.p_max for c in network.components.values() if c.kind == LOAD)
    injection, cost, cons, schedules, fixed, flows = 0, 0, [], {}, {}, []
    for cid, c in network.components.items():
        if c.kind == LOAD:
            continue
        if predispatched(c.kind, c.cost[0]):
            fixed[cid] = c.p_max  # a constant: its cost does not change the optimum
            injection += c.p_max
            continue
        if c.kind == STORAGE:
            p_c, p_d = cp.Variable(steps), cp.Variable(steps)
            cons += storage.constraints(c, p_c, p_d)
            p = p_d - p_c
            flows.append((p_c, p_d))
            cost += c.cost[0] * cp.sum(p_d) + c.cost[1] * cp.sum_squares(p)
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
    else:
        # islanded: renewables are never curtailed, so a surplus must go somewhere to stay feasible
        injection -= cp.Variable(steps, nonneg=True)
    problem = cp.Problem(cp.Minimize(cost), [*cons, injection == demand])
    if any(c.cost[1] for c in network.components.values()):
        # Clarabel's equilibration mis-scales the 4000 €/MWh import next to ~10–100 €/MWh units:
        # with it, base_s_1 (168 h) stops 2e-5 above the islanded optimum while reporting optimal.
        problem.solve(solver=cp.CLARABEL, equilibrate_enable=False)
    else:
        problem.solve(solver=cp.HIGHS)
    if problem.status != cp.OPTIMAL:
        return {}, problem.status, None
    solved = {cid: np.asarray(p.value) for cid, p in schedules.items()}
    both = sum(float(np.minimum(c.value, d.value).clip(0).sum()) for c, d in flows)
    return fixed | solved, problem.status, both
