"""Tests for the Pyomo central dispatch."""

from __future__ import annotations

import pytest

from energy_scheduling_benchmark.dispatch import solve_central_dispatch


class TestSolveCentralDispatch:
    def test_merit_order(self):
        result = solve_central_dispatch(
            costs=[10.0, 30.0], p_max=[5.0, 5.0], demand=4.0
        )
        assert result.success
        assert result.dispatch[0] == pytest.approx(4.0)
        assert result.dispatch[1] == pytest.approx(0.0)

    def test_respects_max(self):
        result = solve_central_dispatch(
            costs=[10.0, 30.0], p_max=[2.0, 5.0], demand=4.0
        )
        assert result.success
        assert result.dispatch[0] == pytest.approx(2.0)
        assert result.dispatch[1] == pytest.approx(2.0)

    def test_infeasible_reports_failure(self):
        result = solve_central_dispatch(costs=[10.0], p_max=[1.0], demand=5.0)
        assert not result.success

    def test_empty_reports_failure(self):
        result = solve_central_dispatch(costs=[], p_max=[], demand=0.0)
        assert not result.success

    def test_min_bound(self):
        # Feasible with min=3, max=10, demand=4 → dispatch = 4
        result = solve_central_dispatch(
            costs=[10.0], p_max=[10.0], demand=4.0, p_min=[3.0]
        )
        assert result.success
        assert result.dispatch[0] == pytest.approx(4.0)

    def test_min_bound_infeasible(self):
        # min=3 > demand=1 with only one generator → infeasible
        result = solve_central_dispatch(
            costs=[10.0], p_max=[10.0], demand=1.0, p_min=[3.0]
        )
        assert not result.success
