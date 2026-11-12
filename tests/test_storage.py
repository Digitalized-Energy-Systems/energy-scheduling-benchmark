import numpy as np
from mango_energy_environments import LOAD, STORAGE

from energy_scheduling_benchmark import datasets, storage
from energy_scheduling_benchmark.central import solve_central


def test_response_at_central_prices_is_central_schedule():
    net = datasets.load("toy")
    schedules, *_ = solve_central(net)
    # λ per hour = marginal cost of a unit strictly inside its bounds (KKT of central)
    lam = np.full(len(net.time_index), np.nan)
    for cid, c in net.components.items():
        if c.kind in (LOAD, STORAGE):
            continue
        p = schedules[cid]
        inside = (p > c.p_min + 1e-3) & (p < c.p_max - 1e-3) & np.isnan(lam)
        lam[inside] = (c.cost[0] + 2 * c.cost[1] * p)[inside]
    assert not np.isnan(lam).any()
    battery = net.components["battery"]
    np.testing.assert_allclose(
        storage.response(battery)(lam), schedules["battery"], atol=1e-3
    )


def test_response_equals_cvxpy_solve():
    """The direct Clarabel path (audit T1) gives cvxpy's schedules bit for bit."""
    import cvxpy as cp

    c = datasets.load("toy").components["battery"]
    steps = len(c.p_max)
    lam = cp.Parameter(steps)
    p_c, p_d = cp.Variable(steps), cp.Variable(steps)
    p = p_d - p_c
    cost = c.cost[0] * cp.sum(p_d) + c.cost[1] * cp.sum_squares(p) - lam @ p
    problem = cp.Problem(cp.Minimize(cost), storage.constraints(c, p_c, p_d))
    respond = storage.response(c)
    for prices in np.random.default_rng(0).uniform(0, 80, (5, steps)):
        lam.value = prices
        problem.solve(solver=cp.CLARABEL)
        np.testing.assert_array_equal(respond(prices), p.value)
