import numpy as np
from mango_energy_environments import STORAGE, THERMAL

from energy_scheduling_benchmark.algorithms.dummy import Offer, merit_order


def test_merit_order():
    flat = np.ones(2)
    offers = [
        Offer(THERMAL, 50.0, 1 * flat, 10 * flat),
        Offer(THERMAL, 20.0, 0 * flat, 4 * flat),
        None,
        Offer(STORAGE, 0.0, -5 * flat, 5 * flat),
    ]
    s = merit_order(offers, np.array([3.0, 8.0]))
    np.testing.assert_array_equal(
        s[0], [1.0, 4.0]
    )  # p_min, then what the cheap one leaves
    np.testing.assert_array_equal(s[1], [2.0, 4.0])
    assert s[2] is None
    np.testing.assert_array_equal(s[3], [0.0, 0.0])  # storage idle
