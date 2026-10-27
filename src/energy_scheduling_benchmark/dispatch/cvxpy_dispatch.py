"""Centralised economic dispatch via cvxpy.

Minimises generation cost subject to a hard demand-balance constraint and
per-generator capacity bounds. :func:`solve_multiperiod_dispatch` clears a
whole horizon at once so storage units can shift energy between timesteps.
The Julia original used JuMP + HiGHS; this module builds one cvxpy model and
solves the LP with HiGHS, the QP (quadratic costs) with Clarabel.

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

import cvxpy as cp
import numpy as np

__all__ = [
    "solve_multiperiod_dispatch",
    "MultiPeriodDispatchResult",
    "SlackSpec",
    "StorageSpec",
]


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
        ``Σ c₂ · p²`` to the objective (``--cost-model quadratic``), solved by
        Clarabel instead of HiGHS.
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
    if (n == 0 and k == 0) or T == 0:
        return _failure(n, k, T, "no_units")

    quad = np.zeros(n) if costs_quad is None else np.asarray(costs_quad, dtype=float)
    if quad.shape != (n,) or np.any(quad < 0):
        raise ValueError("costs_quad must be non-negative, one value per generator")

    cons = []
    net = np.zeros(T)
    cost = 0.0
    if n:
        p = cp.Variable((n, T))
        cons += [p >= p_min_arr, p <= p_max_arr]
        net = net + cp.sum(p, axis=0)
        cost += cp.sum(np.asarray(costs, dtype=float) @ p)
        if np.any(quad):
            cost += cp.sum(quad @ cp.square(p))
    if k:

        def col(attr: str) -> np.ndarray:
            return np.array([getattr(s, attr) for s in storage], dtype=float)[:, None]

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
        # _CYCLING_PENALTY breaks ties against pointless charge/discharge cycling
        # when storage is lossless and prices are flat (e.g. the toy network).
        cost += cp.sum(cp.multiply(col("discharge_cost"), p_d))
        cost += _CYCLING_PENALTY * cp.sum(p_c)
    if slack is not None:
        s_imp, s_exp = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
        net = net + s_imp - s_exp
        cost += slack.import_price * cp.sum(s_imp)
        cost += (slack.export_price + _SLACK_EXPORT_TIEBREAK) * cp.sum(s_exp)

    problem = cp.Problem(cp.Minimize(cost), [*cons, net == demand_arr])
    # HiGHS' QP solver fails ("Solve error") or stalls on PyPSA-Eur networks
    # (base_s_5_elec_2020, lower_saxony_2030); Clarabel does not.
    problem.solve(solver=cp.CLARABEL if np.any(quad) else cp.HIGHS)
    if problem.status != cp.OPTIMAL:
        return _failure(n, k, T, problem.status)
    return MultiPeriodDispatchResult(
        success=True,
        generation=p.value if n else np.zeros((0, T)),
        storage=(p_d.value - p_c.value) if k else np.zeros((0, T)),
        energy=E.value if k else np.zeros((0, T + 1)),
        objective=float(problem.value),
        solver_status=problem.status,
        slack_import=s_imp.value if slack is not None else None,
        slack_export=s_exp.value if slack is not None else None,
    )


def _failure(n: int, k: int, T: int, status: str) -> MultiPeriodDispatchResult:
    return MultiPeriodDispatchResult(
        success=False,
        generation=np.zeros((n, T)),
        storage=np.zeros((k, T)),
        energy=np.zeros((k, T + 1)),
        objective=float("nan"),
        solver_status=status,
    )
