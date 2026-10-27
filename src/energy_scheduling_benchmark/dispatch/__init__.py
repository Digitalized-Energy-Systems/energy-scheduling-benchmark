"""Economic dispatch (centralised LP).

Replaces the Julia ``JuMP + HiGHS`` dispatch block with one cvxpy model
(HiGHS for the LP, Clarabel for the QP).
"""

from .cvxpy_dispatch import (
    MultiPeriodDispatchResult,
    SlackSpec,
    StorageSpec,
    solve_multiperiod_dispatch,
)

__all__ = [
    "solve_multiperiod_dispatch",
    "MultiPeriodDispatchResult",
    "SlackSpec",
    "StorageSpec",
]
