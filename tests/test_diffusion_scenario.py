"""Tests for the diffusion scenario.

End-to-end integration smoke tests of execute_test_case on the built-in toy
network, mirroring test_deed_admm_scenario.py.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios.diffusion import execute_test_case


class TestExecuteTestCase:
    """End-to-end integration tests using the toy network."""

    async def test_run_writes_balanced_dispatch(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "diffusion")
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

        power_cols = [c for c in df.columns if c.startswith("P:")]
        target_cols = [c for c in df.columns if c.startswith("target:")]
        assert power_cols, "No generator power recordings in CSV"
        assert target_cols, "No demand target recording in CSV"

        # At least one generator must have dispatched some power.
        assert any(df[col].abs().max() > 1e-3 for col in power_cols)

        # Per-timestep power balance: Σ generation ≈ demand target.
        total = df[power_cols].fillna(0.0).sum(axis=1).to_numpy()
        target = df[target_cols[0]].fillna(0.0).to_numpy()
        mask = np.abs(target) > 1e-6
        gap = np.abs(total[mask] - target[mask])
        tol = np.maximum(2.0, 0.05 * np.abs(target[mask]))
        assert np.all(gap <= tol), (
            f"Power balance violated: max gap {gap.max():.2f} MW "
            f"(demand ~{np.abs(target[mask]).mean():.1f} MW)"
        )

    async def test_raises_when_network_has_no_loads(self, tmp_path):
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
                name_base=str(tmp_path / "diffusion"),
                simulate_days=1,
            )

    async def test_raises_on_lossy_transport(self, tmp_path):
        """Diffusion's fixed-round message accounting requires lossless comms."""
        scenario = build_toy_network(periods=24)
        with pytest.raises(ValueError, match="lossless"):
            await execute_test_case(
                scenario=scenario,
                loss_percent=5.0,
                name_base=str(tmp_path / "diffusion"),
                simulate_days=1,
            )
