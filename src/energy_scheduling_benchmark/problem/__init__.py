"""The economic-dispatch problem every algorithm and central dispatch solve.

* :mod:`.rules` — rules shared by all (pre-dispatched units, grid import price, no-dispatch window).
* :mod:`.storage` — shared storage model + price response.
* :mod:`.qp` — cvxpy QP canonicalised once, re-solved by Clarabel (storage response, ADMM bus).
"""

# 2026-11-17 D-146: problem rules, storage model and QP helper in their own subpackage
