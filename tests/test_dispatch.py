"""Tests for the Pyomo central dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from energy_scheduling_benchmark.dispatch import (
    SlackSpec,
    StorageSpec,
    solve_multiperiod_dispatch,
)


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


class TestMultiperiodDispatchSlack:
    """Slack node (import/export) in the full-horizon LP."""

    def test_no_slack_reports_none(self):
        result = solve_multiperiod_dispatch(
            costs=[10.0], p_max=[[5.0, 5.0]], demand=[4.0, 4.0]
        )
        assert result.success
        assert result.slack_import is None and result.slack_export is None

    def test_slack_unused_when_generation_suffices(self):
        kwargs = dict(
            costs=[10.0, 30.0], p_max=[[5.0, 5.0], [5.0, 5.0]], demand=[4.0, 8.0]
        )
        plain = solve_multiperiod_dispatch(**kwargs)
        with_slack = solve_multiperiod_dispatch(**kwargs, slack=SlackSpec())
        assert with_slack.success
        np.testing.assert_allclose(with_slack.generation, plain.generation, atol=1e-6)
        np.testing.assert_allclose(with_slack.slack_import, [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(with_slack.slack_export, [0.0, 0.0], atol=1e-6)
        assert with_slack.objective == pytest.approx(plain.objective)

    def test_shortfall_is_imported_at_import_price(self):
        # Without a slack this LP is infeasible (5 MW capacity, 8 MW demand).
        result = solve_multiperiod_dispatch(
            costs=[10.0],
            p_max=[[5.0, 5.0]],
            demand=[4.0, 8.0],
            slack=SlackSpec(import_price=3000.0),
        )
        assert result.success
        np.testing.assert_allclose(result.generation[0], [4.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(result.slack_import, [0.0, 3.0], atol=1e-6)
        assert result.objective == pytest.approx(10 * 9 + 3000 * 3)

    def test_forced_surplus_is_exported(self):
        # p_min = 6 MW above a 4 MW demand: the 2 MW surplus must go somewhere.
        result = solve_multiperiod_dispatch(
            costs=[10.0],
            p_max=[[10.0, 10.0]],
            p_min=[[6.0, 6.0]],
            demand=[4.0, 4.0],
            slack=SlackSpec(),
        )
        assert result.success
        np.testing.assert_allclose(result.slack_export, [2.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(result.slack_import, [0.0, 0.0], atol=1e-6)

    def test_curtailment_preferred_over_free_export(self):
        # Free wind (cost 0) above demand: exporting at price 0 costs the same
        # as curtailing, the tie-break must pick curtailment.
        result = solve_multiperiod_dispatch(
            costs=[0.0],
            p_max=[[10.0, 10.0]],
            demand=[4.0, 4.0],
            slack=SlackSpec(export_price=0.0),
        )
        assert result.success
        np.testing.assert_allclose(result.generation[0], [4.0, 4.0], atol=1e-6)
        np.testing.assert_allclose(result.slack_export, [0.0, 0.0], atol=1e-6)

    def test_cheap_import_replaces_generation(self):
        # Documents why the import price must exceed every marginal cost.
        result = solve_multiperiod_dispatch(
            costs=[50.0],
            p_max=[[10.0]],
            demand=[4.0],
            slack=SlackSpec(import_price=5.0),
        )
        assert result.success
        np.testing.assert_allclose(result.slack_import, [4.0], atol=1e-6)
        np.testing.assert_allclose(result.generation[0], [0.0], atol=1e-6)


class TestQuadraticDispatch:
    def test_equal_incremental_cost_optimum(self):
        # min 2x + 0.5x² + 3y + 0.1y² s.t. x + y = 8 -> 2 + x = 3 + 0.2y -> x = 13/6
        result = solve_multiperiod_dispatch(
            costs=[2.0, 3.0],
            costs_quad=[0.5, 0.1],
            p_max=[[10.0], [10.0]],
            demand=[8.0],
        )
        assert result.success
        np.testing.assert_allclose(result.generation[:, 0], [13 / 6, 35 / 6], atol=1e-6)

    def test_zero_quadratic_terms_are_the_lp(self):
        kwargs = dict(costs=[2.0, 3.0], p_max=[[10.0], [10.0]], demand=[8.0])
        lp = solve_multiperiod_dispatch(**kwargs)
        qp0 = solve_multiperiod_dispatch(costs_quad=[0.0, 0.0], **kwargs)
        np.testing.assert_array_equal(lp.generation, qp0.generation)

    def test_negative_quadratic_cost_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            solve_multiperiod_dispatch(
                costs=[1.0], costs_quad=[-1.0], p_max=[[5.0]], demand=[1.0]
            )


class TestStorageModelConsistency:
    """Central dispatch and the distributed algorithms use the same storage
    model (decision Q15): central dispatch's battery schedule must be feasible
    for the shared StorageModel built from the same parameters, and the
    model's own exact response must satisfy StorageSpec's constraints."""

    @staticmethod
    def _pair():
        from distributed_resource_optimization.misc.storage import StorageModel

        from energy_scheduling_benchmark.dispatch.pyomo_dispatch import StorageSpec

        params = dict(
            e_max=20.0,
            p_charge_max=6.0,
            p_discharge_max=8.0,
            eta_charge=0.9,
            eta_discharge=0.92,
            e_initial=0.5,
        )
        spec = StorageSpec(**params, discharge_cost=1.0)
        model = StorageModel(**params, discharge_cost=1.0)
        return spec, model

    def test_central_schedule_is_feasible_for_the_shared_model(self):
        spec, model = self._pair()
        demand = np.array([30.0, 30.0, 80.0, 90.0, 40.0, 85.0])
        result = solve_multiperiod_dispatch(
            costs=[10.0, 60.0],
            p_max=[[50.0] * 6, [100.0] * 6],
            demand=demand,
            storage=[spec],
        )
        assert result.success
        net = result.storage[0]
        p_d, p_c = np.maximum(net, 0.0), np.maximum(-net, 0.0)
        assert model.is_feasible(p_d, p_c, tol=1e-5)
        np.testing.assert_allclose(model.energy(p_d, p_c), result.energy[0], atol=1e-5)
        assert np.any(net != 0)  # the battery actually trades

    def test_shared_model_response_is_feasible_for_central_dispatch(self):
        spec, model = self._pair()
        lam = np.array([5.0, 5.0, 80.0, 90.0, 5.0, 85.0])
        p, e = model.price_response(lam, 0.1)
        p_d, p_c = np.maximum(p, 0.0), np.maximum(-p, 0.0)
        e0 = spec.e_initial * spec.e_max
        assert np.all(p_d <= spec.p_discharge_max + 1e-6)
        assert np.all(p_c <= spec.p_charge_max + 1e-6)
        assert np.all(e >= -1e-6) and np.all(e <= spec.e_max + 1e-6)
        assert e[-1] >= e0 - 1e-6  # central dispatch's E_T >= E_0
        np.testing.assert_allclose(
            np.diff(e), spec.eta_charge * p_c - p_d / spec.eta_discharge, atol=1e-6
        )
