"""Centralised economic dispatch via Pyomo.

Minimises generation cost subject to a hard demand-balance constraint and
per-generator capacity bounds. :func:`solve_multiperiod_dispatch` clears a
whole horizon at once so storage units can shift energy between timesteps.
The Julia original used JuMP + HiGHS; this module uses Pyomo and defaults to
the HiGHS appsi backend, with a fallback to GLPK when HiGHS is not available.

Usage::

    from energy_scheduling_benchmark import solve_multiperiod_dispatch

    result = solve_multiperiod_dispatch(
        costs=[10.0, 30.0],
        p_max=[[5.0, 5.0], [5.0, 5.0]],
        demand=[4.0, 8.0],
    )
    assert result.success
    print(result.generation)  # per-generator MW output, shape (N, T)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

__all__ = [
    "solve_multiperiod_dispatch",
    "MultiPeriodDispatchResult",
    "SlackSpec",
    "StorageSpec",
]


_solver_name: str | None = None


def _pick_solver() -> pyo.SolverFactory:
    """Pick an available LP solver.  Prefers HiGHS (``highspy``), falls back to GLPK.

    The winning solver name is cached so repeated solves skip the
    availability probes.
    """
    global _solver_name
    if _solver_name is not None:
        return pyo.SolverFactory(_solver_name)
    for name in ("appsi_highs", "glpk", "cbc"):
        try:
            solver = pyo.SolverFactory(name)
            if solver.available(exception_flag=False):
                _solver_name = name
                return solver
        except Exception:  # noqa: BLE001 — probe failure is fine, try next
            continue
    raise RuntimeError(
        "No Pyomo LP solver available — install `highspy`, GLPK, or CBC."
    )


#: Cost per MWh charged (€/MWh); negligible, only a tie-breaker.
_CYCLING_PENALTY = 1e-6


@dataclass(frozen=True)
class StorageSpec:
    """SOC-constrained storage unit for :func:`solve_multiperiod_dispatch`.

    Same model as the distributed scenarios' storage actors (see
    ``ProximalStorageADMMActor`` in ``distributed_resource_optimization``):
    ``E[t+1] = E[t] - p_d[t] / eta_discharge + p_c[t] * eta_charge`` with
    ``0 <= E <= e_max`` and the terminal condition ``E[T] >= E[0]`` (the
    battery ends at least as full as it started; same rule in every
    algorithm, decision Q15).
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    eta_charge: float
    eta_discharge: float
    #: Initial (and final) state of charge as a fraction of ``e_max``.
    e_initial: float
    #: Cost per MWh discharged. Charging is free, matching the benchmark's
    #: reported cost metric (``compute_overall_cost`` clips charging to 0).
    discharge_cost: float = 0.0


#: Extra cost per MWh exported through the slack (€/MWh), only in the LP
#: objective: makes curtailment win over a free export when both cost the same.
_SLACK_EXPORT_TIEBREAK = 1e-3


@dataclass(frozen=True)
class SlackSpec:
    """Slack node (point of common coupling) that closes the power balance.

    Stands in for the PCC of Bai et al. 2022 (Eq. 3): it supplies any shortfall
    (import) and absorbs any surplus (export) at the given prices (€/MWh).
    The import price should exceed every generator's marginal cost, otherwise
    importing replaces generation.
    """

    import_price: float = 4000.0
    export_price: float = 0.0


@dataclass(frozen=True)
class MultiPeriodDispatchResult:
    """Outcome of a multi-period economic-dispatch solve."""

    success: bool
    #: Generator dispatch, shape ``(n_generators, T)``.
    generation: np.ndarray
    #: Net storage power (discharge − charge), shape ``(n_storage, T)``.
    storage: np.ndarray
    #: Storage energy level, shape ``(n_storage, T + 1)``.
    energy: np.ndarray
    objective: float
    solver_status: str
    #: Slack import / export per timestep (length ``T``); ``None`` without a slack.
    slack_import: np.ndarray | None = None
    slack_export: np.ndarray | None = None


def solve_multiperiod_dispatch(
    costs: Sequence[float],
    p_max: Sequence[Sequence[float]] | np.ndarray,
    demand: Sequence[float] | np.ndarray,
    p_min: Sequence[Sequence[float]] | np.ndarray | None = None,
    storage: Sequence[StorageSpec] = (),
    slack: SlackSpec | None = None,
    costs_quad: Sequence[float] | None = None,
) -> MultiPeriodDispatchResult:
    """Solve a copper-plate economic dispatch over a horizon as one LP
    (a convex QP with *costs_quad*).

    Parameters
    ----------
    costs:
        Marginal cost per generator (length *N*).
    p_max:
        Per-timestep upper bound per generator, shape ``(N, T)``.
    demand:
        Aggregate load per timestep (length *T*), met exactly by generation
        plus net storage discharge.
    p_min:
        Optional per-timestep lower bounds, shape ``(N, T)`` (default zero).
    storage:
        Storage units co-optimised over the horizon (see :class:`StorageSpec`).
    costs_quad:
        Optional quadratic cost coefficient per generator (€/MW²h, ≥ 0), adding
        ``Σ c₂ · p²`` to the objective (``--cost-model quadratic``). The QP is
        solved by Clarabel via cvxpy, the LP by HiGHS via Pyomo.
    slack:
        Optional slack node (see :class:`SlackSpec`). Adds import/export
        variables to every timestep's balance, priced in the objective (export
        plus a tiny tie-break so curtailment is preferred), which also keeps
        the LP feasible when capacity falls short.

    Returns
    -------
    MultiPeriodDispatchResult
        ``success`` is ``True`` when the LP returns an optimal solution;
        otherwise the arrays are zero-filled and ``objective`` is NaN.
        ``objective`` includes the slack terms (with the tie-break).
    """
    demand_arr = np.asarray(demand, dtype=float).ravel()
    T = demand_arr.size
    n = len(costs)
    p_max_arr = np.asarray(p_max, dtype=float).reshape(n, T)
    p_min_arr = (
        np.zeros((n, T))
        if p_min is None
        else np.asarray(p_min, dtype=float).reshape(n, T)
    )
    storage = list(storage)
    k = len(storage)

    def _failure(status: str) -> MultiPeriodDispatchResult:
        return MultiPeriodDispatchResult(
            success=False,
            generation=np.zeros((n, T)),
            storage=np.zeros((k, T)),
            energy=np.zeros((k, T + 1)),
            objective=float("nan"),
            solver_status=status,
        )

    if (n == 0 and k == 0) or T == 0:
        return _failure("no_units")

    quad = None if costs_quad is None else np.asarray(costs_quad, dtype=float)
    if quad is not None and np.any(quad):
        if quad.shape != (n,) or np.any(quad < 0):
            raise ValueError("costs_quad must be non-negative, one value per generator")
        # HiGHS' QP solver fails ("Solve error") or stalls on PyPSA-Eur
        # networks (base_s_5_elec_2020, lower_saxony_2030); Clarabel does not.
        return _solve_qp_clarabel(
            np.asarray(costs, dtype=float),
            quad,
            p_min_arr,
            p_max_arr,
            demand_arr,
            storage,
            slack,
            _failure,
        )

    model = pyo.ConcreteModel()
    model.G = pyo.RangeSet(0, n - 1) if n else pyo.Set(initialize=[])
    model.S = pyo.RangeSet(0, k - 1) if k else pyo.Set(initialize=[])
    model.T = pyo.RangeSet(0, T - 1)
    model.T1 = pyo.RangeSet(0, T)

    model.p = pyo.Var(
        model.G,
        model.T,
        bounds=lambda _m, i, t: (p_min_arr[i, t], p_max_arr[i, t]),
    )
    model.p_c = pyo.Var(
        model.S,
        model.T,
        bounds=lambda _m, s, t: (0.0, storage[s].p_charge_max),
    )
    model.p_d = pyo.Var(
        model.S,
        model.T,
        bounds=lambda _m, s, t: (0.0, storage[s].p_discharge_max),
    )
    model.E = pyo.Var(
        model.S, model.T1, bounds=lambda _m, s, t: (0.0, storage[s].e_max)
    )

    def _e0(s: int) -> float:
        return float(np.clip(storage[s].e_initial, 0.0, 1.0)) * storage[s].e_max

    model.soc_initial = pyo.Constraint(model.S, rule=lambda m, s: m.E[s, 0] == _e0(s))
    model.soc_final = pyo.Constraint(model.S, rule=lambda m, s: m.E[s, T] >= _e0(s))
    model.soc_dynamics = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: (
            m.E[s, t + 1]
            == m.E[s, t]
            - m.p_d[s, t] / storage[s].eta_discharge
            + m.p_c[s, t] * storage[s].eta_charge
        ),
    )
    if slack is not None:
        model.s_imp = pyo.Var(model.T, bounds=(0.0, None))
        model.s_exp = pyo.Var(model.T, bounds=(0.0, None))

    def _slack_net(m: pyo.ConcreteModel, t: int) -> pyo.Expression | float:
        return m.s_imp[t] - m.s_exp[t] if slack is not None else 0.0

    model.balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            sum(m.p[i, t] for i in m.G)
            + sum(m.p_d[s, t] - m.p_c[s, t] for s in m.S)
            + _slack_net(m, t)
            == demand_arr[t]
        ),
    )
    # _CYCLING_PENALTY breaks ties against pointless charge/discharge cycling
    # when storage is lossless and prices are flat (e.g. the toy network).
    cost_expr = sum(costs[i] * model.p[i, t] for i in model.G for t in model.T) + sum(
        storage[s].discharge_cost * model.p_d[s, t] + _CYCLING_PENALTY * model.p_c[s, t]
        for s in model.S
        for t in model.T
    )
    if slack is not None:
        cost_expr += sum(
            slack.import_price * model.s_imp[t]
            + (slack.export_price + _SLACK_EXPORT_TIEBREAK) * model.s_exp[t]
            for t in model.T
        )
    model.cost = pyo.Objective(expr=cost_expr, sense=pyo.minimize)

    solver = _pick_solver()
    # appsi solvers raise RuntimeError instead of reporting failure when asked
    # to auto-load the solution of an infeasible solve — defer loading until
    # the termination condition is known.
    results = solver.solve(model, load_solutions=False, tee=False)
    status = str(results.solver.termination_condition)
    if status not in {"optimal", "locallyOptimal"}:
        return _failure(status)

    model.solutions.load_from(results)
    generation = np.array(
        [[pyo.value(model.p[i, t]) for t in range(T)] for i in range(n)], dtype=float
    ).reshape(n, T)
    net_storage = np.array(
        [
            [pyo.value(model.p_d[s, t]) - pyo.value(model.p_c[s, t]) for t in range(T)]
            for s in range(k)
        ],
        dtype=float,
    ).reshape(k, T)
    energy = np.array(
        [[pyo.value(model.E[s, t]) for t in range(T + 1)] for s in range(k)],
        dtype=float,
    ).reshape(k, T + 1)
    slack_import = slack_export = None
    if slack is not None:
        slack_import = np.array([pyo.value(model.s_imp[t]) for t in range(T)])
        slack_export = np.array([pyo.value(model.s_exp[t]) for t in range(T)])
    return MultiPeriodDispatchResult(
        success=True,
        generation=generation,
        storage=net_storage,
        energy=energy,
        objective=float(pyo.value(model.cost)),
        solver_status=status,
        slack_import=slack_import,
        slack_export=slack_export,
    )


def _solve_qp_clarabel(
    costs: np.ndarray,
    quad: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    demand: np.ndarray,
    storage: list[StorageSpec],
    slack: SlackSpec | None,
    failure,
) -> MultiPeriodDispatchResult:
    """The model of :func:`solve_multiperiod_dispatch` as a cvxpy QP, solved by Clarabel."""
    import cvxpy as cp

    n, T = p_max.shape
    k = len(storage)
    p = cp.Variable((n, T))
    net = cp.sum(p, axis=0)
    cons = [p >= p_min, p <= p_max]
    cost = cp.sum(costs @ p) + cp.sum(quad @ cp.square(p))
    if k:
        col = lambda attr: np.array([getattr(s, attr) for s in storage])[:, None]  # noqa: E731
        e0 = np.clip(col("e_initial"), 0.0, 1.0) * col("e_max")
        p_c, p_d, E = cp.Variable((k, T)), cp.Variable((k, T)), cp.Variable((k, T + 1))
        cons += [
            p_c >= 0, p_c <= col("p_charge_max"),
            p_d >= 0, p_d <= col("p_discharge_max"),
            E >= 0, E <= col("e_max"),
            E[:, :1] == e0, E[:, T:] >= e0,
            E[:, 1:] == E[:, :-1] - cp.multiply(1 / col("eta_discharge"), p_d)
            + cp.multiply(col("eta_charge"), p_c),
        ]  # fmt: skip
        net = net + cp.sum(p_d - p_c, axis=0)
        cost += cp.sum(
            cp.multiply(col("discharge_cost"), p_d)
        ) + _CYCLING_PENALTY * cp.sum(p_c)
    if slack is not None:
        s_imp, s_exp = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
        net = net + s_imp - s_exp
        cost += slack.import_price * cp.sum(s_imp) + (
            slack.export_price + _SLACK_EXPORT_TIEBREAK
        ) * cp.sum(s_exp)
    problem = cp.Problem(cp.Minimize(cost), [*cons, net == demand])
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        return failure(problem.status)
    return MultiPeriodDispatchResult(
        success=True,
        generation=p.value,
        storage=(p_d.value - p_c.value) if k else np.zeros((0, T)),
        energy=E.value if k else np.zeros((0, T + 1)),
        objective=float(problem.value),
        solver_status=problem.status,
        slack_import=s_imp.value if slack is not None else None,
        slack_export=s_exp.value if slack is not None else None,
    )
