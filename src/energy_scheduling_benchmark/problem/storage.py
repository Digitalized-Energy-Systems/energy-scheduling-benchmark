"""The benchmark's storage model (Q15), shared by central dispatch and every algorithm.

``E[t+1] = E[t] + η_store·p_c[t] − p_d[t]/η_dispatch``, ``0 ≤ E ≤ p_nom·max_hours``,
``E[0] = state_of_charge_initial``, terminal ``E[T] ≥ E[0]``; charge ≤ −p_min, discharge ≤ p_max.
Cost c₁·p_d (discharge only) + c₂·p² on the net power (curvature band, `datasets.CURVATURE_BAND`).
Schedules are net power p = p_d − p_c.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from mango_energy_environments import Component

from .qp import QP


def soc(c: Component, p: np.ndarray) -> np.ndarray:
    """Energy (MWh) at the end of each hour for net schedule *p*."""
    eta_c, eta_d = c.params["efficiency_store"], c.params["efficiency_dispatch"]
    flow = eta_c * np.maximum(-p, 0.0) - np.maximum(p, 0.0) / eta_d
    return c.params["state_of_charge_initial"] + np.cumsum(flow)


def cost(c: Component, p: np.ndarray) -> float:
    """Cost of the net schedule *p* (the only thing a schedule holds): c₁ on the net discharge.

    2026-11-24 D-244: the optimisations charge c₁ on p_d (Xu 2018 eq. 7), which equals this unless
    a storage charges and discharges in the same hour — never optimal with η < 1 and c₁ > 0 (Zhu
    2025 Thm 1); central reports such hours as ``storage_simultaneous_mwh``. Charging the net in
    the objective instead would make simultaneous charge + discharge a free surplus sink."""
    return float(c.cost[0] * np.maximum(p, 0.0).sum() + c.cost[1] * np.sum(p**2))


def constraints(c: Component, p_c: cp.Variable, p_d: cp.Variable) -> list:
    """Q15 constraints on charge *p_c* and discharge *p_d* (both ≥ 0, one value per hour)."""
    eta_c, eta_d = c.params["efficiency_store"], c.params["efficiency_dispatch"]
    e0 = c.params["state_of_charge_initial"]
    energy = e0 + cp.cumsum(eta_c * p_c - p_d / eta_d)
    e_max = c.params["p_nom"] * c.params["max_hours"]
    return [
        p_c >= 0,
        p_c <= -c.p_min,
        p_d >= 0,
        p_d <= c.p_max,
        energy >= 0,
        energy <= e_max,
        energy[-1] >= e0,
    ]


def response(c: Component):
    """``respond(λ)`` → net schedule P = argmin cost(P) − λᵀP over the Q15 set (HKW15 eq. 19).

    The cvxpy model is canonicalised once (:class:`.qp.QP`); per λ only the linear term q changes
    (−λ·p_d + λ·p_c). The schedules are bit-identical to ``problem.solve(solver=cp.CLARABEL)``
    (test_storage).
    """
    steps = len(c.p_max)
    lam = cp.Parameter(steps, value=np.zeros(steps))
    p_c, p_d = cp.Variable(steps), cp.Variable(steps)
    p = p_d - p_c
    cost = c.cost[0] * cp.sum(p_d) + c.cost[1] * cp.sum_squares(p) - lam @ p
    qp = QP(
        cp.Problem(cp.Minimize(cost), constraints(c, p_c, p_d)), "storage response"
    )  # λ = 0
    i_c, i_d = qp.col[p_c.id], qp.col[p_d.id]

    def respond(prices: np.ndarray) -> np.ndarray:
        q = qp.q0.copy()
        q[i_c : i_c + steps] += prices
        q[i_d : i_d + steps] -= prices
        x = qp.solve(q)
        return x[i_d : i_d + steps] - x[i_c : i_c + steps]

    return respond
