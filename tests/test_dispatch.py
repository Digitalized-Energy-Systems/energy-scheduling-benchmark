"""Tests for the Pyomo central dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from energy_scheduling_benchmark.dispatch import (
    StorageSpec,
    solve_central_dispatch,
    solve_multiperiod_dispatch,
)


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


class TestSolveMultiperiodDispatch:
    """Full-horizon LP with SOC-constrained storage."""

    def _battery(self, **kwargs) -> StorageSpec:
        params = dict(
            e_max=10.0,
            p_charge_max=5.0,
            p_discharge_max=5.0,
            eta_charge=1.0,
            eta_discharge=1.0,
            e_initial=0.0,
        )
        params.update(kwargs)
        return StorageSpec(**params)

    def test_without_storage_matches_per_timestep_merit_order(self):
        result = solve_multiperiod_dispatch(
            costs=[10.0, 30.0],
            p_max=[[5.0, 5.0], [5.0, 5.0]],
            demand=[4.0, 8.0],
        )
        assert result.success
        np.testing.assert_allclose(
            result.generation, [[4.0, 5.0], [0.0, 3.0]], atol=1e-6
        )
        assert result.objective == pytest.approx(10 * 9 + 30 * 3)

    def test_storage_shifts_cheap_energy(self):
        # Hour 0: only cheap gen available; hour 1: cheap gen curtailed, expensive
        # gen must run unless the battery shifts cheap energy forward.
        result = solve_multiperiod_dispatch(
            costs=[10.0, 50.0],
            p_max=[[10.0, 0.0], [10.0, 10.0]],
            demand=[4.0, 4.0],
            storage=[self._battery()],
        )
        assert result.success
        np.testing.assert_allclose(result.storage[0], [-4.0, 4.0], atol=1e-6)
        np.testing.assert_allclose(result.generation[1], [0.0, 0.0], atol=1e-6)
        assert result.objective == pytest.approx(80.0, abs=1e-3)

    def test_soc_is_cyclic_and_bounded(self):
        result = solve_multiperiod_dispatch(
            costs=[10.0, 50.0],
            p_max=[[10.0, 10.0, 0.0], [10.0, 10.0, 10.0]],
            demand=[2.0, 2.0, 8.0],
            storage=[self._battery(e_initial=0.5)],
        )
        assert result.success
        energy = result.energy[0]
        assert energy[0] == pytest.approx(5.0)
        assert energy[-1] == pytest.approx(5.0)
        assert energy.min() >= -1e-6 and energy.max() <= 10.0 + 1e-6

    def test_power_limits_cap_the_shift(self):
        result = solve_multiperiod_dispatch(
            costs=[10.0, 50.0],
            p_max=[[10.0, 0.0], [10.0, 10.0]],
            demand=[4.0, 4.0],
            storage=[self._battery(p_charge_max=1.5, p_discharge_max=1.5)],
        )
        assert result.success
        np.testing.assert_allclose(result.storage[0], [-1.5, 1.5], atol=1e-6)
        np.testing.assert_allclose(result.generation[1], [0.0, 2.5], atol=1e-6)

    def test_round_trip_losses(self):
        # eta_c = eta_d = 0.5: 4 MWh discharged needs 16 MWh charged.
        result = solve_multiperiod_dispatch(
            costs=[10.0, 50.0],
            p_max=[[20.0, 0.0], [20.0, 10.0]],
            demand=[4.0, 4.0],
            storage=[
                self._battery(
                    e_max=20.0,
                    p_charge_max=20.0,
                    p_discharge_max=20.0,
                    eta_charge=0.5,
                    eta_discharge=0.5,
                )
            ],
        )
        assert result.success
        np.testing.assert_allclose(result.storage[0], [-16.0, 4.0], atol=1e-6)
        assert result.generation[0][0] == pytest.approx(20.0)

    def test_losses_make_shifting_unprofitable(self):
        # Round-trip efficiency 0.25 makes 10 €/MWh energy cost 40 €/MWh at
        # discharge — dearer than running the 30 €/MWh unit directly.
        result = solve_multiperiod_dispatch(
            costs=[10.0, 30.0],
            p_max=[[20.0, 0.0], [20.0, 10.0]],
            demand=[4.0, 4.0],
            storage=[
                self._battery(
                    e_max=20.0,
                    p_charge_max=20.0,
                    p_discharge_max=20.0,
                    eta_charge=0.5,
                    eta_discharge=0.5,
                )
            ],
        )
        assert result.success
        np.testing.assert_allclose(result.storage[0], [0.0, 0.0], atol=1e-6)

    def test_storage_only_network(self):
        result = solve_multiperiod_dispatch(
            costs=[],
            p_max=np.zeros((0, 2)),
            demand=[0.0, 0.0],
            storage=[self._battery()],
        )
        assert result.success
        assert result.generation.shape == (0, 2)

    def test_infeasible_reports_failure(self):
        result = solve_multiperiod_dispatch(
            costs=[10.0],
            p_max=[[1.0, 1.0]],
            demand=[5.0, 5.0],
            storage=[self._battery()],
        )
        assert not result.success
        assert np.isnan(result.objective)

    def test_empty_reports_failure(self):
        result = solve_multiperiod_dispatch(costs=[], p_max=[], demand=[])
        assert not result.success
