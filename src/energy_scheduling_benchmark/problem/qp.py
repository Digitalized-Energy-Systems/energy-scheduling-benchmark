"""A cvxpy QP canonicalised once and re-solved by calling Clarabel directly.

Per solve only the linear term q changes (one solver, updated the way cvxpy's solver cache does);
a new P (ADMM's ρ) rebuilds the solver. Why: cvxpy re-canonicalised on every call — 85–95 % of
the sweep time (audit 2026-11-12, T1); results are bit-identical to ``problem.solve(solver=cp.CLARABEL)``
(test_storage).
"""

# 2026-11-17 D-145: one helper for storage.response and the ADMM bus (was duplicated)
from __future__ import annotations

import clarabel
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones


class QP:
    def __init__(self, problem: cp.Problem, name: str) -> None:
        # canonicalised with the parameters at their current value
        data, _, _ = problem.get_problem_data(cp.CLARABEL)
        self.col = data["param_prob"].var_id_to_col  # variable id → first index in x
        self.q0 = data["c"]
        n = self.q0.size
        P = data.get("P")  # absent when the cost is linear
        self.P = sp.csc_matrix((n, n)) if P is None else sp.triu(P).tocsc()
        self._A, self._b = data["A"], data["b"]
        self._cones = dims_to_solver_cones(data["dims"])
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._name, self._solver = name, None

    def solve(self, q: np.ndarray, P: sp.csc_matrix | None = None) -> np.ndarray:
        """x for linear term *q*; a new upper-triangular *P* replaces the old one."""
        if P is not None:
            self.P, self._solver = P, None
        if self._solver is None or not self._solver.is_data_update_allowed():
            self._solver = clarabel.DefaultSolver(
                self.P, q, self._A, self._b, self._cones, self._settings
            )
        else:
            self._solver.update(q=q)
        solution = self._solver.solve()
        if solution.status != clarabel.SolverStatus.Solved:
            raise RuntimeError(f"{self._name}: {solution.status}")
        return np.asarray(solution.x)
