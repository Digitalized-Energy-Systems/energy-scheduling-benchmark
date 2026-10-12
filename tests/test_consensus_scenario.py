"""Tests for the consensus scenario.

End-to-end integration smoke tests of execute_test_case on the built-in
toy network.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios.consensus import execute_test_case


class TestConsensusScenarioIntegration:
    """End-to-end integration tests using the toy network."""

    async def test_runs_and_writes_outputs_on_toy_network(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "consensus")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        assert os.path.exists(f"{name_base}-df.csv")
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
                delay_s=0.0,
                loss_rate=0.0,
                name_base=str(tmp_path / "consensus"),
                simulate_days=1,
            )


class TestConsensusUnderLoss:
    """Consensus is the one scenario that accepts a non-zero loss rate."""

    async def _run(self, tmp_path, name, **kwargs):
        name_base = str(tmp_path / name)
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            delay_s=0.02,
            name_base=name_base,
            simulate_days=1,
            **kwargs,
        )
        return pd.read_csv(f"{name_base}-df.csv", index_col=0)

    async def test_same_seed_gives_identical_results(self, tmp_path):
        a = await self._run(tmp_path, "a", loss_rate=0.05, seed=1)
        b = await self._run(tmp_path, "b", loss_rate=0.05, seed=1)
        pd.testing.assert_frame_equal(a, b)

    async def test_unfinished_run_is_reported(self, tmp_path, caplog):
        # With every message lost the kick-off never arrives, so no generator
        # finishes; this used to write outputs without any error.
        await self._run(tmp_path, "lost", loss_rate=1.0)
        assert "never finished" in caplog.text
        assert "thermal0" in caplog.text

    async def test_unfinished_run_raises_under_strict(self, tmp_path):
        with pytest.raises(RuntimeError, match="never verified"):
            await self._run(tmp_path, "strict", loss_rate=1.0, strict=True)

    async def test_deprecated_loss_percent_keyword(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="loss_percent"):
            await self._run(tmp_path, "pct", loss_percent=0.0)
