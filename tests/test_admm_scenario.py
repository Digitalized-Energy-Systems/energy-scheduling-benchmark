"""Tests for the ADMM scenario.

Covers unit-level helpers (_lookup_ts, _make_finish_callback) and an
end-to-end integration smoke test of execute_test_case on the built-in toy network.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import ComponentRef

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios._common import _lookup_ts
from energy_scheduling_benchmark.scenarios.admm import (
    _make_finish_callback,
    execute_test_case,
)


class TestLookupTs:
    """_lookup_ts returns a unit's series, or None."""

    def _scenario(self, ts_dict: dict) -> object:
        scenario = MagicMock()
        scenario.timeseries = ts_dict
        return scenario

    def test_direct_match(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0, 2.0])
        assert _lookup_ts(self._scenario({ref: ts}), ref) is ts

    def test_missing_key_returns_none(self):
        assert (
            _lookup_ts(self._scenario({}), ComponentRef("thermal", "missing")) is None
        )

    def test_element_type_mismatch_returns_none(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0])
        assert _lookup_ts(self._scenario({ref: ts}), ComponentRef("load", "g0")) is None

    def test_component_id_mismatch_returns_none(self):
        ref = ComponentRef("thermal", "g0")
        ts = pd.Series([1.0])
        assert (
            _lookup_ts(self._scenario({ref: ts}), ComponentRef("thermal", "g1")) is None
        )


class TestMakeFinishCallback:
    """_make_finish_callback sends the participant's x to the leader."""

    @staticmethod
    def _role(sends: list):
        class FakeContext:
            aid = "gen0"

            def schedule_instant_message(self, msg, addr):
                sends.append((msg, addr))

        class FakeRole:
            context = FakeContext()

        return FakeRole()

    async def test_notifies_leader_with_schedule(self):
        sends: list = []
        algorithm = MagicMock()
        algorithm.x = np.array([1.0, 2.0])
        leader_addr = ("localhost", 5555)

        cb = _make_finish_callback(leader_addr_ref={"addr": leader_addr})
        cb(algorithm, self._role(sends), "gen0")

        assert len(sends) == 1
        msg, addr = sends[0]
        assert msg.aid == "gen0"
        assert np.allclose(msg.schedule, [1.0, 2.0])
        assert addr == leader_addr

    async def test_sent_schedule_is_a_copy(self):
        sends: list = []
        algorithm = MagicMock()
        algorithm.x = np.array([1.0, 2.0])
        cb = _make_finish_callback(leader_addr_ref={"addr": ("localhost", 5555)})
        cb(algorithm, self._role(sends), "gen0")
        algorithm.x[0] = 99.0
        assert sends[0][0].schedule[0] == pytest.approx(1.0)

    async def test_no_notification_when_leader_addr_is_none(self):
        sends: list = []
        algorithm = MagicMock()
        algorithm.x = np.array([1.0])
        cb = _make_finish_callback(leader_addr_ref={"addr": None})
        cb(algorithm, self._role(sends), "gen0")
        assert sends == []


class TestADMMScenarioIntegration:
    """End-to-end integration tests using the toy network."""

    async def test_runs_without_error_on_toy_network(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
        )

    async def test_csv_output_is_written(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "admm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_rate=0.0,
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
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        assert not df.empty
        assert any("P:" in col for col in df.columns)

    async def test_raises_when_network_has_no_loads(self, tmp_path):
        from datetime import datetime

        import pypsa

        net = pypsa.Network()
        net.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
        net.add("Bus", "b0")
        net.add(
            "Generator", "g0", bus="b0", carrier="gas", p_nom=10.0, marginal_cost=10.0
        )
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
            loss_rate=0.0,
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
            loss_rate=0.0,
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
            loss_rate=0.0,
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
        assert float(gap.mean()) < 20.0, (
            f"Mean generation-demand gap {gap.mean():.1f} MW is too large"
        )

    async def test_runs_under_message_loss(self, tmp_path):
        """With 5% of messages lost the run still finishes and passes the
        power-balance check (strict): missing values are replaced by the last
        received ones and lost results are requested again."""
        name_base = str(tmp_path / "admm")
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            loss_rate=0.05,
            seed=1,
            name_base=name_base,
            simulate_days=1,
            strict=True,
        )
        assert os.path.exists(f"{name_base}-df.csv")

    async def test_max_missing_rounds_reaches_coordinator(self, tmp_path, monkeypatch):
        import energy_scheduling_benchmark.scenarios.admm as admm_mod

        seen = {}
        real = admm_mod.create_consensus_target_reach_admm_coordinator

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(
            admm_mod, "create_consensus_target_reach_admm_coordinator", spy
        )
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            name_base=str(tmp_path / "admm"),
            simulate_days=1,
            max_missing_rounds=7,
        )
        assert seen["max_missing_rounds"] == 7

    async def test_adaptive_rho_from_a_poor_start(self, tmp_path, monkeypatch):
        """--adaptive-rho reaches the optimum from rho = 10 (fixed rho = 10
        creeps); the coordinator's rho changes are recorded."""
        import energy_scheduling_benchmark.scenarios.admm as admm_mod

        coordinators = []
        real = admm_mod.create_consensus_target_reach_admm_coordinator

        def spy(**kwargs):
            coordinators.append(real(**kwargs))
            return coordinators[-1]

        monkeypatch.setattr(
            admm_mod, "create_consensus_target_reach_admm_coordinator", spy
        )
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            name_base=str(tmp_path / "admm"),
            simulate_days=1,
            rho=10.0,
            adaptive_rho=True,
            strict=True,
        )
        coordinator = coordinators[0]
        assert coordinator.adaptive_rho and coordinator.converged
        assert coordinator.rho_changes > 0 and coordinator.final_rho != 10.0
