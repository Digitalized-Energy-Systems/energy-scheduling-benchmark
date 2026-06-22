"""Tests for the ADMM scenario.

Covers unit-level helpers (_lookup_ts, _scalar, _make_finish_callback) and an
end-to-end integration smoke test of execute_test_case on the built-in toy network.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from distributed_resource_optimization import solve_battery_price_schedule

from energy_scheduling_benchmark.environment import ComponentRef
from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios.admm import (
    FixedScheduleActor,
    _lookup_ts,
    _make_finish_callback,
    _scalar,
    execute_test_case,
)


class TestLookupTs:
    """_lookup_ts falls back to attribute comparison when dict get fails."""

    def _scenario(self, ts_dict: dict) -> object:
        scenario = MagicMock()
        scenario.timeseries = ts_dict
        return scenario

    def test_direct_match(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0, 2.0])
        assert _lookup_ts(self._scenario({ref: ts}), ref) is ts

    def test_attribute_fallback_when_key_does_not_hash_equal(self):
        """Covers the case where the stored key and the lookup ref are not equal
        (e.g. ComponentRef vs. a subclass), but share element_type/component_id."""

        class AltRef:
            """Stand-in for a ComponentRef subclass that breaks equality."""

            def __init__(self, et: str, cid: str) -> None:
                self.element_type = et
                self.component_id = cid

            def __eq__(self, other: object) -> bool:
                return False

            def __hash__(self) -> int:
                return id(self)

        real_ref = ComponentRef("thermal", "g0")
        ts = pd.Series([3.0, 4.0])
        alt_ref = AltRef("thermal", "g0")
        # dict.get uses hash+eq, so this won't find real_ref
        assert self._scenario({real_ref: ts}).timeseries.get(alt_ref) is None
        # _lookup_ts should still find it via attribute scan
        assert _lookup_ts(self._scenario({real_ref: ts}), alt_ref) is ts

    def test_missing_key_returns_none(self):
        assert _lookup_ts(self._scenario({}), ComponentRef("thermal", "missing")) is None

    def test_element_type_mismatch_returns_none(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0])
        assert _lookup_ts(self._scenario({ref: ts}), ComponentRef("load", "g0")) is None

    def test_component_id_mismatch_returns_none(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0])
        assert _lookup_ts(self._scenario({ref: ts}), ComponentRef("thermal", "g1")) is None


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
    """_make_finish_callback stores schedules and sends notifications."""

    async def test_stores_schedule_keyed_by_aid(self):
        schedule_by_aid: dict = {}
        algorithm = MagicMock()
        algorithm.x = np.array([1.0, 2.0, 3.0])

        class FakeRole:
            class context:
                aid = "gen0"

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid=schedule_by_aid
        )
        cb(algorithm, FakeRole(), "gen0")

        assert "gen0" in schedule_by_aid
        assert np.allclose(schedule_by_aid["gen0"], [1.0, 2.0, 3.0])

    async def test_stored_schedule_is_a_copy(self):
        schedule_by_aid: dict = {}
        original = np.array([1.0, 2.0])
        algorithm = MagicMock()
        algorithm.x = original

        class FakeRole:
            class context:
                aid = "gen0"

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid=schedule_by_aid
        )
        cb(algorithm, FakeRole(), "gen0")
        original[0] = 99.0
        assert schedule_by_aid["gen0"][0] == pytest.approx(1.0)

    async def test_notifies_leader_when_addr_is_set(self):
        sends: list = []

        class FakeContext:
            aid = "gen0"

            async def send_message(self, msg, addr):
                sends.append((msg, addr))

        class FakeRole:
            context = FakeContext()

        algorithm = MagicMock()
        algorithm.x = np.array([0.0])
        leader_addr = ("localhost", 5555)

        cb = _make_finish_callback(
            leader_addr_ref={"addr": leader_addr}, schedule_by_aid={}
        )
        cb(algorithm, FakeRole(), "gen0")
        await asyncio.sleep(0)  # let the task execute

        assert len(sends) == 1
        assert sends[0][1] == leader_addr

    async def test_no_notification_when_leader_addr_is_none(self):
        algorithm = MagicMock()
        algorithm.x = np.array([1.0])

        class FakeRole:
            class context:
                aid = "gen0"

        cb = _make_finish_callback(
            leader_addr_ref={"addr": None}, schedule_by_aid={}
        )
        # Must not raise even when no leader address is available yet
        cb(algorithm, FakeRole(), "gen0")


class TestADMMScenarioIntegration:
    """End-to-end integration tests using the toy network."""

    async def test_runs_without_error_on_toy_network(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )

    async def test_csv_output_is_written(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
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
        name_base = str(tmp_path / "admm")
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
                name_base=str(tmp_path / "admm"),
                simulate_days=1,
            )

    async def test_schedule_is_applied_for_all_generators(self, tmp_path):
        """After the run, recorded power values should be non-trivially non-zero
        for at least the thermal generators."""
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
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
        # At least one generator must have dispatched some power
        assert any(df[col].abs().max() > 1e-3 for col in power_cols)

    async def test_thermal_generators_dispatch_non_trivially(self, tmp_path):
        """Thermals must produce positive power — verifies the merit-order fix
        that ensures thermals are not stuck at zero due to a wrong actor type."""
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_percent=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        thermal_cols = [c for c in df.columns if "thermal" in c.lower()]
        assert thermal_cols, "No thermal columns in CSV"
        # At least the cheapest thermal must dispatch substantially
        assert any(df[col].max() > 10.0 for col in thermal_cols)

    async def test_total_generation_tracks_demand(self, tmp_path):
        """Sum of all generator power should approximate recorded demand
        (after the first timestep where initial state is recorded)."""
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
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

        # After the schedule is applied, generation must be within 20 MW of demand.
        gap = (tail_tgt - tail_gen).abs()
        assert float(gap.mean()) < 20.0, f"Mean generation-demand gap {gap.mean():.1f} MW is too large"


class TestFixedScheduleActor:
    """FixedScheduleActor always replies with its pre-computed schedule."""

    async def test_replies_with_fixed_schedule(self):
        from unittest.mock import MagicMock, AsyncMock
        from distributed_resource_optimization import ADMMMessage

        sched = np.array([1.0, 2.0, 3.0])
        actor = FixedScheduleActor(sched)

        replies = []

        class FakeCarrier:
            def reply_to_other(self, msg, meta):
                replies.append(msg)

        msg = MagicMock(spec=ADMMMessage)
        await actor.on_exchange_message(FakeCarrier(), msg, {})

        assert len(replies) == 1
        assert np.allclose(replies[0].x, sched)

    def test_x_is_a_copy(self):
        original = np.array([1.0, 2.0])
        actor = FixedScheduleActor(original)
        original[0] = 99.0
        assert actor.x[0] == pytest.approx(1.0)

    async def test_ignores_non_admm_messages(self):
        actor = FixedScheduleActor(np.array([1.0]))
        calls = []

        class FakeCarrier:
            def reply_to_other(self, msg, meta):
                calls.append(msg)

        await actor.on_exchange_message(FakeCarrier(), "not-an-admm-message", {})
        assert calls == []


class TestSolveBatteryPriceSchedule:
    """solve_battery_price_schedule solves an SOC-constrained LP."""

    def test_discharges_when_price_exceeds_cost(self):
        # High uniform price, zero discharge cost → should discharge
        pi = np.full(6, 10.0)
        sched = solve_battery_price_schedule(
            horizon=6, pi=pi,
            e_max=20.0, p_charge_max=5.0, p_discharge_max=5.0,
            eta_charge=1.0, eta_discharge=1.0,
            e_initial=0.5, e_final=0.5,
        )
        assert sched is not None
        assert float(np.max(sched)) > 0.1

    def test_soc_never_goes_negative(self):
        pi = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
        sched = solve_battery_price_schedule(
            horizon=6, pi=pi,
            e_max=10.0, p_charge_max=3.0, p_discharge_max=3.0,
            eta_charge=0.95, eta_discharge=0.95,
            e_initial=0.5, e_final=0.5,
        )
        e = 0.5 * 10.0
        for p in sched:
            if p >= 0:
                e -= p / 0.95
            else:
                e += (-p) * 0.95
            assert e >= -1e-4

    def test_terminal_soc_is_respected(self):
        pi = np.full(4, 1.0)
        sched = solve_battery_price_schedule(
            horizon=4, pi=pi,
            e_max=10.0, p_charge_max=5.0, p_discharge_max=5.0,
            e_initial=0.3, e_final=0.3,
        )
        e = 0.3 * 10.0
        for p in sched:
            if p >= 0:
                e -= p
            else:
                e += (-p)
        assert abs(e - 0.3 * 10.0) < 0.5
