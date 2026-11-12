"""The benchmark's storage model (Q15), shared by central dispatch and every algorithm.

``E[t+1] = E[t] + η_store·p_c[t] − p_d[t]/η_dispatch``, ``0 ≤ E ≤ p_nom·max_hours``,
``E[0] = state_of_charge_initial``, terminal ``E[T] ≥ E[0]``; charge ≤ −p_min, discharge ≤ p_max.
Cost c₁·p_d (discharge only) + c₂·p² on the net power (curvature band, `datasets.CURVATURE_BAND`).
Schedules are net power p = p_d − p_c.
"""

from __future__ import annotations

import clarabel
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones
from mango_energy_environments import Component


def soc(c: Component, p: np.ndarray) -> np.ndarray:
    """Energy (MWh) at the end of each hour for net schedule *p*."""
    eta_c, eta_d = c.params["efficiency_store"], c.params["efficiency_dispatch"]
    flow = eta_c * np.maximum(-p, 0.0) - np.maximum(p, 0.0) / eta_d
    return c.params["state_of_charge_initial"] + np.cumsum(flow)


def cost(c: Component, p: np.ndarray) -> float:
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

    The cvxpy model is canonicalised once; per λ only the linear term q changes (−λ·p_d + λ·p_c),
    and Clarabel is called directly, updating one solver the way cvxpy's solver cache does.
    Why: cvxpy re-canonicalised on every call — 85–95 % of the sweep time (audit 2026-11-12, T1);
    the schedules are bit-identical to ``problem.solve(solver=cp.CLARABEL)`` (test_storage).
    """
    steps = len(c.p_max)
    lam = cp.Parameter(steps, value=np.zeros(steps))
    p_c, p_d = cp.Variable(steps), cp.Variable(steps)
    p = p_d - p_c
    cost = c.cost[0] * cp.sum(p_d) + c.cost[1] * cp.sum_squares(p) - lam @ p
    problem = cp.Problem(cp.Minimize(cost), constraints(c, p_c, p_d))
    data, _, _ = problem.get_problem_data(cp.CLARABEL)  # at λ = 0
    col = data["param_prob"].var_id_to_col
    i_c, i_d = col[p_c.id], col[p_d.id]
    P, A, b, q0 = sp.triu(data["P"]).tocsc(), data["A"], data["b"], data["c"]
    cones = dims_to_solver_cones(data["dims"])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = None

    def respond(prices: np.ndarray) -> np.ndarray:
        nonlocal solver
        q = q0.copy()
        q[i_c : i_c + steps] += prices
        q[i_d : i_d + steps] -= prices
        if solver is None or not solver.is_data_update_allowed():
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        else:
            solver.update(P=P, q=q, A=A, b=b, settings=solver.get_settings())
        solution = solver.solve()
        if solution.status != clarabel.SolverStatus.Solved:
            raise RuntimeError(f"storage response: {solution.status}")
        x = np.asarray(solution.x)
        return x[i_d : i_d + steps] - x[i_c : i_c + steps]

    return respond
