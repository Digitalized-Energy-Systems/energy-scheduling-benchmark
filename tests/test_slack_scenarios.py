"""End-to-end tests of the optional slack node (``--slack``) in the scenarios.

The slack closes the power balance after the fact: per hour it imports
``demand − Σ schedules`` when positive and exports the surplus otherwise,
priced into ``cost:total`` and written as ``slack:import``/``slack:export``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.dispatch import SlackSpec
from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios import central_dispatch, consensus, fdgdm

SLACK_COLS = ["slack:import", "slack:export"]


async def _run(module, tmp_path, name: str, scenario=None, **kwargs) -> pd.DataFrame:
    name_base = str(tmp_path / name)
    await module.execute_test_case(
        scenario=scenario or build_toy_network(periods=24),
        delay_s=0.0,
        loss_rate=0.0,
        name_base=name_base,
        simulate_days=1,
        **kwargs,
    )
    return pd.read_csv(f"{name_base}-df.csv", index_col=0)


class TestSlackOffByDefault:
    async def test_no_slack_columns_without_slack(self, tmp_path):
        df = await _run(central_dispatch, tmp_path, "cd")
        assert not set(SLACK_COLS) & set(df.columns)


class TestCentralDispatchSlack:
    async def test_feasible_network_leaves_slack_unused_and_cost_unchanged(
        self, tmp_path
    ):
        plain = await _run(central_dispatch, tmp_path, "plain")
        with_slack = await _run(central_dispatch, tmp_path, "slack", slack=SlackSpec())
        assert set(SLACK_COLS) <= set(with_slack.columns)
        assert with_slack[SLACK_COLS].abs().to_numpy().max() < 1e-6
        np.testing.assert_allclose(
            with_slack["cost:total"].to_numpy(), plain["cost:total"].to_numpy()
        )


class TestConsensusSlack:
    async def test_slack_cost_added_to_total(self, tmp_path):
        """Consensus is deterministic without packet loss, so the dispatch is
        the same with and without the slack; only the slack cost is added."""
        prices = SlackSpec(import_price=3000.0, export_price=7.0)
        plain = await _run(consensus, tmp_path, "plain")
        with_slack = await _run(consensus, tmp_path, "slack", slack=prices)
        expected = (
            plain["cost:total"]
            + with_slack["slack:import"] * prices.import_price
            + with_slack["slack:export"] * prices.export_price
        )
        np.testing.assert_allclose(with_slack["cost:total"], expected)

    async def test_slack_closes_recorded_balance(self, tmp_path):
        df = await _run(consensus, tmp_path, "slack", slack=SlackSpec())
        p_cols = [c for c in df.columns if c.startswith("P:")]
        t_cols = [c for c in df.columns if c.startswith("target:")]
        rows = df[t_cols[0]] > 0
        closed = (
            df.loc[rows, p_cols].sum(axis=1)
            + df.loc[rows, "slack:import"]
            - df.loc[rows, "slack:export"]
        )
        np.testing.assert_allclose(closed, df.loc[rows, t_cols[0]], atol=1e-6)


class TestFDGDMSlack:
    async def test_renewable_surplus_is_exported(self, tmp_path):
        """With 400 MW of wind against ~75 MW of load, FDGDM's pre-scheduled
        renewables exceed demand in every hour; the slack must absorb it."""
        scenario = build_toy_network(periods=24)
        scenario.net.generators.loc["wind0", "p_nom"] = 400.0
        df = await _run(fdgdm, tmp_path, "fdgdm", scenario=scenario, slack=SlackSpec())
        assert df["slack:export"].max() > 1.0
        assert df["slack:import"].max() < 1e-6


class TestSlackCli:
    def test_slack_with_strict_is_rejected(self):
        with pytest.raises(SystemExit):
            central_dispatch.main(["--slack", "--strict"])
