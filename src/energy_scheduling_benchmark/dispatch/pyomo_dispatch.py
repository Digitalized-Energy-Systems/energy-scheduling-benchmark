"""Centralised economic dispatch via Pyomo.

Minimises generation cost subject to a hard demand-balance constraint and
per-generator capacity bounds.  The Julia original used JuMP + HiGHS; this
module uses Pyomo and defaults to the HiGHS appsi backend, with a fallback
to GLPK when HiGHS is not available.

Usage::

    from energy_scheduling_benchmark import solve_central_dispatch

    result = solve_central_dispatch(
        costs=[10.0, 30.0, 50.0],
        p_max=[100.0, 80.0, 40.0],
        demand=150.0,
    )
    assert result.success
    print(result.dispatch)  # per-generator MW output
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pyomo.environ as pyo

__all__ = ["solve_central_dispatch", "DispatchResult"]


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of a single economic-dispatch solve."""

    success: bool
    dispatch: list[float]
    objective: float
    solver_status: str


def _pick_solver() -> pyo.SolverFactory:
    """Pick an available LP solver.  Prefers HiGHS (``highspy``), falls back to GLPK."""
    for name in ("appsi_highs", "glpk", "cbc"):
        try:
            solver = pyo.SolverFactory(name)
            if solver.available(exception_flag=False):
                return solver
        except Exception:  # noqa: BLE001 — probe failure is fine, try next
            continue
    raise RuntimeError(
        "No Pyomo LP solver available — install `highspy`, GLPK, or CBC."
    )


def solve_central_dispatch(
    costs: Sequence[float],
    p_max: Sequence[float],
    demand: float,
    p_min: Sequence[float] | None = None,
) -> DispatchResult:
    """Solve a copper-plate economic dispatch as an LP.

    Parameters
    ----------
    costs:
        Marginal cost per generator (length *N*).
    p_max:
        Upper bound on power output per generator (length *N*).
    demand:
        Aggregate load that must be met exactly.
    p_min:
        Optional lower bounds (defaults to zero for every generator).

    Returns
    -------
    DispatchResult
        ``success`` is ``True`` when the LP returns an optimal solution;
        otherwise ``dispatch`` is zero-filled and ``objective`` is NaN.
    """
    if len(costs) != len(p_max):
        raise ValueError("costs and p_max must have the same length")
    if p_min is not None and len(p_min) != len(costs):
        raise ValueError("p_min must have the same length as costs")

    n = len(costs)
    if n == 0:
        return DispatchResult(success=False, dispatch=[], objective=float("nan"), solver_status="no_generators")

    p_min_list = list(p_min) if p_min is not None else [0.0] * n

    model = pyo.ConcreteModel()
    model.G = pyo.RangeSet(0, n - 1)
    model.p = pyo.Var(
        model.G,
        domain=pyo.Reals,
        bounds=lambda _m, i: (p_min_list[i], p_max[i]),
    )
    model.balance = pyo.Constraint(expr=sum(model.p[i] for i in model.G) == demand)
    model.cost = pyo.Objective(
        expr=sum(costs[i] * model.p[i] for i in model.G),
        sense=pyo.minimize,
    )

    solver = _pick_solver()
    results = solver.solve(model, tee=False)

    status = str(results.solver.termination_condition)
    success = status in {"optimal", "locallyOptimal"}

    if not success:
        return DispatchResult(
            success=False,
            dispatch=[0.0] * n,
            objective=float("nan"),
            solver_status=status,
        )

    dispatch = [float(pyo.value(model.p[i])) for i in model.G]
    objective = float(pyo.value(model.cost))
    return DispatchResult(
        success=True, dispatch=dispatch, objective=objective, solver_status=status
    )
