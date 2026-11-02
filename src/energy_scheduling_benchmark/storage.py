"""The benchmark's storage model (Q15), shared by central dispatch and every algorithm.

``E[t+1] = E[t] + η_store·p_c[t] − p_d[t]/η_dispatch``, ``0 ≤ E ≤ p_nom·max_hours``,
``E[0] = state_of_charge_initial``, terminal ``E[T] ≥ E[0]``; charge ≤ −p_min, discharge ≤ p_max.
Only discharge is priced (c₁·p_d). Schedules are net power p = p_d − p_c.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from mango_energy_environments import Component


def soc(c: Component, p: np.ndarray) -> np.ndarray:
    """Energy (MWh) at the end of each hour for net schedule *p*."""
    eta_c, eta_d = c.params["efficiency_store"], c.params["efficiency_dispatch"]
    flow = eta_c * np.maximum(-p, 0.0) - np.maximum(p, 0.0) / eta_d
    return c.params["state_of_charge_initial"] + np.cumsum(flow)


def cost(c: Component, p: np.ndarray) -> float:
    return float(c.cost[0] * np.maximum(p, 0.0).sum())


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
