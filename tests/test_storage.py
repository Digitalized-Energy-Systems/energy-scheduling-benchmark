import numpy as np
from mango_energy_environments import LOAD, STORAGE

from energy_scheduling_benchmark import datasets, storage
from energy_scheduling_benchmark.central import solve_central


def test_response_at_central_prices_is_central_schedule():
    net = datasets.load("toy")
    schedules, _ = solve_central(net)
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
