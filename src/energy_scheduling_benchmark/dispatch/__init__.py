"""Economic dispatch (centralised LP).

Replaces the Julia ``JuMP + HiGHS`` dispatch block with a Pyomo model
solved by the HiGHS backend (``highspy``) or any other Pyomo-compatible
solver.
"""

from .pyomo_dispatch import DispatchResult, solve_central_dispatch

__all__ = ["solve_central_dispatch", "DispatchResult"]
