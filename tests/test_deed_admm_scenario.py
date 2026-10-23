"""Tests for the DEED-ADMM scenario.

End-to-end integration smoke tests of execute_test_case on the built-in toy
network, mirroring test_admm_scenario.py.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios.deed_admm import execute_test_case


class TestExecuteTestCase:
    """End-to-end integration tests using the toy network."""

    async def test_run_writes_balanced_dispatch(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "deed_admm")
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
        assert mask.any(), "demand target is zero in every row"
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
                name_base=str(tmp_path / "deed_admm"),
                simulate_days=1,
            )

    async def test_runs_under_message_loss(self, tmp_path):
        """With 5% of messages lost the run still finishes and passes the
        power-balance check (strict): missing values are replaced by the last
        received ones and lost results are requested again."""
        name_base = str(tmp_path / "deed_admm")
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            loss_rate=0.05,
            seed=1,
            name_base=name_base,
            simulate_days=1,
            strict=True,
        )
        assert os.path.exists(f"{name_base}-df.csv")


class TestConvergenceStop:
    async def test_default_stops_early_with_the_same_cost(self, tmp_path, caplog):
        import logging
        import re

        def run(name, **kwargs):
            return execute_test_case(
                scenario=build_toy_network(periods=24),
                name_base=str(tmp_path / name),
                simulate_days=1,
                **kwargs,
            )

        pattern = r"DEED-ADMM finished .*iterations=(\d+)"
        caplog.set_level(logging.INFO)
        await run("early")
        early_iters = {int(m) for m in re.findall(pattern, caplog.text)}
        caplog.clear()
        await run("full", tol_ratio=None)
        full_iters = {int(m) for m in re.findall(pattern, caplog.text)}

        assert full_iters == {500}
        assert early_iters and max(early_iters) < 500
        early = pd.read_csv(tmp_path / "early-df.csv", index_col=0)["cost:total"].sum()
        full = pd.read_csv(tmp_path / "full-df.csv", index_col=0)["cost:total"].sum()
        assert early == pytest.approx(full, rel=1e-3)

    def test_cli_default_and_zero_disables(self):
        from energy_scheduling_benchmark.scenarios._common import (
            build_scenario_argparser,
        )
        from energy_scheduling_benchmark.scenarios.deed_admm import _add_deed_admm_args
        from energy_scheduling_benchmark.scenarios.fdgdm import PAPER_TOL_RATIO

        parser = build_scenario_argparser(
            None, default_name_base="x", extra_args=_add_deed_admm_args
        )
        assert parser.parse_args([]).tol_ratio == PAPER_TOL_RATIO
        assert parser.parse_args(["--tol-ratio", "0"]).tol_ratio == 0.0
