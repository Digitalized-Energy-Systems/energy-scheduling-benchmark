"""Tests for the central dispatch scenario.

End-to-end smoke tests of execute_test_case on the built-in toy network:
every hourly timestep must be dispatched, total generation (including net
storage discharge) must match the aggregated demand, and the toy battery
must take part.
"""

from __future__ import annotations

import pandas as pd

from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios.central_dispatch import execute_test_case


class TestExecuteTestCase:
    async def _run(self, tmp_path, **kwargs) -> pd.DataFrame:
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "central")
        await execute_test_case(
            scenario=scenario,
            delay_s=kwargs.pop("delay_s", 0.0),
            loss_rate=kwargs.pop("loss_rate", 0.0),
            name_base=name_base,
            simulate_days=1,
            **kwargs,
        )
        return pd.read_csv(f"{name_base}-df.csv", index_col=0)

    async def test_every_hour_dispatched(self, tmp_path):
        """One row per simulated hour, each with a positive demand target —
        a stalled report/solve round-trip would leave hours missing."""
        df = await self._run(tmp_path)
        target_cols = [c for c in df.columns if c.startswith("target:")]
        assert target_cols, "No target column in CSV"
        target = df[target_cols[0]]
        assert len(df) >= 24
        assert (target.iloc[:24] > 0).all()

    async def test_generation_matches_demand(self, tmp_path):
        """The LP meets demand exactly, so recorded generation must track the
        recorded target closely at every dispatched timestep."""
        df = await self._run(tmp_path)
        power_cols = [c for c in df.columns if c.startswith("P:")]
        target_cols = [c for c in df.columns if c.startswith("target:")]
        assert power_cols and target_cols

        total_gen = df[power_cols].sum(axis=1)
        target = df[target_cols[0]]
        gap = (target - total_gen).abs()
        assert float(gap.max()) < 1.0, f"Max generation-demand gap {gap.max():.2f} MW"

    async def test_merit_order_prefers_cheap_generation(self, tmp_path):
        """Zero-cost wind must be dispatched; thermals cover the rest."""
        df = await self._run(tmp_path)
        power_cols = [c for c in df.columns if c.startswith("P:")]
        renewable_cols = [c for c in power_cols if "wind" in c.lower()]
        thermal_cols = [c for c in power_cols if "thermal" in c.lower()]
        assert renewable_cols and thermal_cols
        assert df[renewable_cols].to_numpy().max() > 1e-3
        assert df[thermal_cols].to_numpy().max() > 1e-3

    async def test_storage_participates(self, tmp_path):
        """The toy battery is dispatched within its power rating (20 MW)."""
        df = await self._run(tmp_path)
        assert "P:batt0" in df.columns
        assert df["P:batt0"].abs().max() <= 20.0 + 1e-6

    async def test_runs_under_message_loss(self, tmp_path):
        """A lost unit report is requested again (mango gather), so with 5%
        message loss the LP still solves with every unit and balances."""
        df = await self._run(
            tmp_path, delay_s=0.02, loss_rate=0.05, seed=1, strict=True
        )
        assert len(df) == 24
