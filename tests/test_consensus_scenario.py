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
            loss_percent=0.0,
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
                loss_percent=0.0,
                name_base=str(tmp_path / "consensus"),
                simulate_days=1,
            )
