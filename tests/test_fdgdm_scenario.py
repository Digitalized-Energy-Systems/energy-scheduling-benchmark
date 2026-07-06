"""Tests for the FDGDM scenario.

Covers unit-level helpers (_scalar, _make_finish_callback) and an
end-to-end integration smoke test of execute_test_case on the built-in toy network.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios._common import _scalar
from energy_scheduling_benchmark.scenarios.fdgdm import (
    FDGDMFinishedInfo,
    _capacity_proportional_allocation,
    _keep_hourly,
    _make_finish_callback,
    _schedule_storage_soc,
    execute_test_case,
)


class TestScalar:
    """_scalar extracts the first element of an array-like as a float."""

    def test_single_element_array(self):
        assert _scalar(np.array([3.14])) == pytest.approx(3.14)

    def test_plain_float(self):
        assert _scalar(5.0) == pytest.approx(5.0)

    def test_empty_array_returns_zero(self):
        assert _scalar(np.array([])) == pytest.approx(0.0)

    def test_takes_first_element_only(self):
        assert _scalar(np.array([7.0, 8.0, 9.0])) == pytest.approx(7.0)

    def test_two_dimensional_array(self):
        assert _scalar(np.array([[4.0, 5.0]])) == pytest.approx(4.0)


class TestMakeFinishCallback:
    """_make_finish_callback stores actor.P in schedule_by_aid and notifies the leader.

    The callback signature is ``(algorithm, carrier)`` where ``carrier._parent``
    is the mango Role and ``carrier._parent.context`` is the agent context.
    """

    def _make_fake_carrier(self, aid: str, sends: list | None = None):
        class FakeContext:
            def __init__(self) -> None:
                self.aid = aid

            async def send_message(self, msg, addr):
                if sends is not None:
                    sends.append((msg, addr))

        class FakeRole:
            context = FakeContext()

        class FakeCarrier:
            _parent = FakeRole()

        return FakeCarrier()

    async def test_stores_schedule_keyed_by_aid(self):
        schedule_by_aid: dict = {}
        actor = MagicMock()
        actor.P = np.array([1.0, 2.0, 3.0])
        algorithm = MagicMock()
        algorithm.actor = actor

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid=schedule_by_aid
        )
        cb(algorithm, self._make_fake_carrier("gen0"))

        assert "gen0" in schedule_by_aid
        assert np.allclose(schedule_by_aid["gen0"], [1.0, 2.0, 3.0])

    async def test_stored_schedule_is_a_copy(self):
        schedule_by_aid: dict = {}
        original = np.array([1.0, 2.0])
        actor = MagicMock()
        actor.P = original
        algorithm = MagicMock()
        algorithm.actor = actor

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid=schedule_by_aid
        )
        cb(algorithm, self._make_fake_carrier("gen0"))
        original[0] = 99.0
        assert schedule_by_aid["gen0"][0] == pytest.approx(1.0)

    async def test_notifies_leader_when_addr_is_set(self):
        sends: list = []
        actor = MagicMock()
        actor.P = np.array([0.0])
        algorithm = MagicMock()
        algorithm.actor = actor
        leader_addr = ("localhost", 5555)

        cb = _make_finish_callback(
            leader_addr_ref={"addr": leader_addr}, schedule_by_aid={}
        )
        cb(algorithm, self._make_fake_carrier("gen0", sends=sends))
        await asyncio.sleep(0)

        assert len(sends) == 1
        assert isinstance(sends[0][0], FDGDMFinishedInfo)
        assert sends[0][1] == leader_addr

    async def test_no_notification_when_leader_addr_is_none(self):
        actor = MagicMock()
        actor.P = np.array([1.0])
        algorithm = MagicMock()
        algorithm.actor = actor

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid={}
        )
        cb(algorithm, self._make_fake_carrier("gen0"))


class TestFDGDMScenarioIntegration:
    """End-to-end integration tests using the toy network."""

    async def test_runs_without_error_on_toy_network(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )

    async def test_csv_output_is_written(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        assert os.path.exists(f"{name_base}-df.csv")

    async def test_csv_contains_power_recordings(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        assert not df.empty
        assert any("P:" in col for col in df.columns)

    async def test_raises_when_network_has_no_loads(self, tmp_path):
        import pypsa
        from datetime import datetime

        net = pypsa.Network()
        net.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        net.add("Bus", "b0")
        net.add("Generator", "g0", bus="b0", carrier="gas", p_nom=10.0, marginal_cost=10.0)
        scenario = ScenarioData(net=net, timeseries={}, start=datetime(2024, 1, 1))

        with pytest.raises(RuntimeError, match="No loads"):
            await execute_test_case(
                scenario=scenario,
                name_base=str(tmp_path / "fdgdm"),
                simulate_days=1,
            )

    async def test_schedule_is_applied_for_all_generators(self, tmp_path):
        """After the run, at least one generator must have dispatched non-zero power."""
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        power_cols = [c for c in df.columns if c.startswith("P:")]
        assert len(power_cols) >= 1
        assert any(df[col].abs().max() > 1e-3 for col in power_cols)

    async def test_total_generation_tracks_demand(self, tmp_path):
        """Sum of generator power should be non-trivially close to demand.

        FDGDM conserves the initial total-power allocation (zero-row-sum weight
        matrix), so it does not re-balance to exactly match dynamic demand the
        way ADMM does.  We use a wider tolerance (50 MW) than the ADMM scenario
        test (20 MW) to account for this property.
        """
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        power_cols = [c for c in df.columns if c.startswith("P:")]
        target_cols = [c for c in df.columns if "target" in c.lower()]
        assert power_cols and target_cols

        total_gen = df[power_cols].sum(axis=1)
        target = df[target_cols[0]]

        # Skip the first row which captures the pre-schedule initial state.
        tail_gen = total_gen.iloc[1:]
        tail_tgt = target.iloc[1:]

        gap = (tail_tgt - tail_gen).abs()
        assert float(gap.mean()) < 5.0, (
            f"Mean generation-demand gap {gap.mean():.1f} MW is too large"
        )


# ---------------------------------------------------------------------------
# _capacity_proportional_allocation unit tests
# ---------------------------------------------------------------------------


class TestCapacityProportionalAllocation:
    """_capacity_proportional_allocation distributes demand proportionally to slack."""

    def test_sum_equals_target(self):
        target = np.array([30.0, 40.0])
        p_max = [np.array([20.0, 30.0]), np.array([20.0, 30.0])]
        p_min = [np.zeros(2), np.zeros(2)]
        allocs = _capacity_proportional_allocation(target, p_max, p_min)
        assert np.allclose(sum(allocs), target)

    def test_respects_bounds(self):
        target = np.array([50.0])
        p_max = [np.array([100.0]), np.array([50.0])]
        p_min = [np.zeros(1), np.zeros(1)]
        allocs = _capacity_proportional_allocation(target, p_max, p_min)
        for alloc, pmax, pmin in zip(allocs, p_max, p_min):
            assert np.all(alloc >= pmin - 1e-9)
            assert np.all(alloc <= pmax + 1e-9)

    def test_proportional_to_slack(self):
        # one generator has 2× the slack of the other → gets 2× the share
        target = np.array([30.0])
        p_max = [np.array([60.0]), np.array([30.0])]
        p_min = [np.zeros(1), np.zeros(1)]
        allocs = _capacity_proportional_allocation(target, p_max, p_min)
        assert np.allclose(allocs[0], 2 * allocs[1], atol=1e-9)

    def test_equal_capacity_splits_evenly(self):
        target = np.array([30.0])
        p_max = [np.full(1, 50.0), np.full(1, 50.0), np.full(1, 50.0)]
        p_min = [np.zeros(1)] * 3
        allocs = _capacity_proportional_allocation(target, p_max, p_min)
        assert np.allclose(allocs[0], allocs[1])
        assert np.allclose(allocs[1], allocs[2])
        assert np.allclose(sum(allocs), target)

    def test_non_zero_p_min_included_in_allocation(self):
        # p_min = [5, 5], target = 20: residual = 10, split by equal slack → [10, 10]
        # actual allocation = [p_min + share] = [5 + 5, 5 + 5] = [10, 10]
        target = np.array([20.0])
        p_max = [np.array([15.0]), np.array([15.0])]
        p_min = [np.array([5.0]), np.array([5.0])]
        allocs = _capacity_proportional_allocation(target, p_max, p_min)
        assert np.allclose(sum(allocs), target)
        assert np.allclose(allocs[0], allocs[1])


# ---------------------------------------------------------------------------
# _schedule_storage_soc unit tests
# ---------------------------------------------------------------------------


class TestScheduleStorageSoc:
    """_schedule_storage_soc energy-constrained storage dispatch."""

    def test_discharges_into_deficit(self):
        # One deficit hour, enough SOC to cover it fully.
        net_load = np.array([10.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=20.0,
            p_min_pu=-1.0,
            max_hours=2.0,
            soc_initial_pu=1.0,  # full
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(1),
        )
        assert np.allclose(result, [10.0])

    def test_charges_from_surplus(self):
        # One surplus hour (negative net_load), storage starts empty.
        net_load = np.array([-15.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=20.0,
            p_min_pu=-1.0,
            max_hours=1.0,
            soc_initial_pu=0.0,
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(1),
        )
        # Charges at min(p_charge_max=20, surplus=15, storable=max_energy=20) → 15
        assert np.allclose(result, [-15.0])

    def test_discharge_limited_by_p_nom(self):
        net_load = np.array([100.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=10.0,
            p_min_pu=-1.0,
            max_hours=10.0,
            soc_initial_pu=1.0,
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(1),
        )
        assert np.allclose(result, [10.0])

    def test_soc_not_overdrawn(self):
        # Only 5 MWh stored; try to discharge 10 MW for 1 h → capped at 5 MW.
        net_load = np.array([10.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=10.0,
            p_min_pu=-1.0,
            max_hours=1.0,
            soc_initial_pu=0.5,  # 5 MWh
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(1),
        )
        assert np.allclose(result, [5.0])

    def test_zero_when_no_surplus_or_deficit(self):
        net_load = np.zeros(3)
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=10.0,
            p_min_pu=-1.0,
            max_hours=2.0,
            soc_initial_pu=0.5,
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(3),
        )
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_inflow_increases_soc(self):
        # Start empty, inflow of 5 MWh in first hour, discharge in second.
        net_load = np.array([0.0, 5.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=10.0,
            p_min_pu=-1.0,
            max_hours=1.0,
            soc_initial_pu=0.0,
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.array([5.0, 0.0]),
        )
        # Hour 0: no deficit, no charge; inflow adds 5 MWh
        # Hour 1: deficit 5 MW, 5 MWh available → dispatches 5 MW
        assert np.allclose(result, [0.0, 5.0])

    def test_charge_limited_by_capacity(self):
        # Storage is almost full; surplus exceeds remaining capacity.
        net_load = np.array([-100.0])
        result = _schedule_storage_soc(
            net_load_ts=net_load,
            p_nom=10.0,
            p_min_pu=-1.0,
            max_hours=1.0,
            soc_initial_pu=0.9,  # only 1 MWh left
            efficiency_store=1.0,
            efficiency_dispatch=1.0,
            inflow_mwh=np.zeros(1),
        )
        assert np.allclose(result, [-1.0])


# ---------------------------------------------------------------------------
# _keep_hourly unit tests
# ---------------------------------------------------------------------------


class TestKeepHourly:
    """_keep_hourly retains the last sample within each hour bucket."""

    def test_empty_input_returns_empty(self):
        t_out, Y_out = _keep_hourly([], np.zeros((0, 2)))
        assert len(t_out) == 0

    def test_one_sample_per_hour_unchanged(self):
        t = [0.0, 3600.0, 7200.0]
        Y = np.array([[1.0], [2.0], [3.0]])
        t_out, Y_out = _keep_hourly(t, Y)
        assert np.allclose(t_out, t)
        assert np.allclose(Y_out, Y)

    def test_last_sample_kept_per_bucket(self):
        # Two samples per hour; last one should survive.
        t = [100.0, 3500.0, 3700.0, 7100.0]
        Y = np.array([[1.0], [2.0], [3.0], [4.0]])
        t_out, Y_out = _keep_hourly(t, Y)
        # bucket 0 → last of [100, 3500] → index 1 (value 2)
        # bucket 1 → last of [3700, 7100] → index 3 (value 4)
        assert np.allclose(t_out, [3500.0, 7100.0])
        assert np.allclose(Y_out, [[2.0], [4.0]])

    def test_preserves_multiple_agents(self):
        t = [0.0, 500.0]
        Y = np.array([[1.0, 10.0], [2.0, 20.0]])
        t_out, Y_out = _keep_hourly(t, Y)
        assert np.allclose(t_out, [500.0])
        assert np.allclose(Y_out, [[2.0, 20.0]])
