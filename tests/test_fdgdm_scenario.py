"""Tests for the FDGDM scenario.

Covers unit-level helpers (make_finish_callback) and an
end-to-end integration smoke test of execute_test_case on the built-in toy network.
"""

from __future__ import annotations

import os
from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import ScenarioData, build_toy_network
from energy_scheduling_benchmark.scenarios._common import (
    GeneratorResultRole,
    make_finish_callback,
)
from energy_scheduling_benchmark.scenarios.fdgdm import (
    FDGDMFinishedInfo,
    _capacity_proportional_allocation,
    execute_test_case,
)

# fdgdm.py calls _common.make_finish_callback inline with these fixed kwargs
# rather than through its own wrapper; pin them here so the tests below still
# exercise exactly FDGDM's finish-callback shape.
_make_finish_callback = partial(
    make_finish_callback,
    finished_message_type=FDGDMFinishedInfo,
    algorithm_label="FDGDM",
    schedule_attr="actor.P",
)


class TestMakeFinishCallback:
    """make_finish_callback sends actor.P to the leader and keeps it on the
    agent's GeneratorResultRole (no shared dict between agents).

    The callback signature is ``(algorithm, carrier)`` where ``carrier._parent``
    is the mango Role and ``carrier._parent.context`` is the agent context.
    """

    def _make_fake_carrier(self, aid: str, sends: list | None = None, result_role=None):
        class FakeContext:
            def __init__(self) -> None:
                self.aid = aid

            def schedule_instant_message(self, msg, addr):
                if sends is not None:
                    sends.append((msg, addr))

            def get_role(self, cls):
                return result_role

        class FakeRole:
            context = FakeContext()

        class FakeCarrier:
            _parent = FakeRole()

        return FakeCarrier()

    @staticmethod
    def _algorithm(p):
        actor = MagicMock()
        actor.P = p
        algorithm = MagicMock()
        algorithm.actor = actor
        return algorithm

    async def test_schedule_is_sent_to_the_leader(self):
        sends: list = []
        leader_addr = ("localhost", 5555)
        cb = _make_finish_callback(leader_addr_ref={"addr": leader_addr})
        cb(
            self._algorithm(np.array([1.0, 2.0, 3.0])),
            self._make_fake_carrier("gen0", sends),
        )

        assert len(sends) == 1
        msg, addr = sends[0]
        assert isinstance(msg, FDGDMFinishedInfo)
        assert msg.aid == "gen0"
        assert np.allclose(msg.schedule, [1.0, 2.0, 3.0])
        assert addr == leader_addr

    async def test_sent_schedule_is_a_copy(self):
        sends: list = []
        original = np.array([1.0, 2.0])
        cb = _make_finish_callback(leader_addr_ref={"addr": ("localhost", 5555)})
        cb(self._algorithm(original), self._make_fake_carrier("gen0", sends))
        original[0] = 99.0
        assert sends[0][0].schedule[0] == pytest.approx(1.0)

    async def test_schedule_is_kept_on_the_result_role(self):
        result_role = GeneratorResultRole()
        cb = _make_finish_callback(leader_addr_ref={"addr": None})
        cb(
            self._algorithm(np.array([4.0])),
            self._make_fake_carrier("gen0", result_role=result_role),
        )
        assert np.allclose(result_role.schedule, [4.0])

    async def test_no_notification_when_leader_addr_is_none(self):
        sends: list = []
        cb = _make_finish_callback(leader_addr_ref={"addr": None})
        cb(self._algorithm(np.array([1.0])), self._make_fake_carrier("gen0", sends))
        assert sends == []


class TestFDGDMScenarioIntegration:
    """End-to-end integration tests using the toy network."""

    async def test_runs_without_error_on_toy_network(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
        )

    async def test_csv_output_is_written(self, tmp_path):
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
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
        name_base = str(tmp_path / "fdgdm")
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
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
        )
        df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
        power_cols = [c for c in df.columns if c.startswith("P:")]
        assert len(power_cols) >= 1
        assert any(df[col].abs().max() > 1e-3 for col in power_cols)

    async def test_balance_verification_runs_and_passes(self, tmp_path, caplog):
        """The aggregator must verify power balance after FDGDM finishes
        (demand_target/balance_label wiring, like the sibling scenarios)."""
        import logging

        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        with caplog.at_level(logging.INFO):
            await execute_test_case(
                scenario=scenario,
                delay_s=0.0,
                loss_rate=0.0,
                name_base=name_base,
                simulate_days=1,
                cost_model="linear",  # quadratic toy: Eq. 16 clipping breaks ΣP (deviations.md)
            )
        balance_msgs = [r for r in caplog.records if "power balance" in r.getMessage()]
        assert balance_msgs, "expected a balance verification log entry"
        assert all(r.levelno < logging.ERROR for r in balance_msgs), (
            "balance verification reported a violation on the toy network"
        )

    async def test_total_generation_tracks_demand(self, tmp_path):
        """Sum of generator power should track demand closely (mean gap < 5 MW).

        FDGDM conserves the initial total-power allocation (zero-row-sum weight
        matrix); the demand-feasible proportional initial allocation makes the
        conserved total match demand, so the gap should stay small.
        """
        scenario = build_toy_network(periods=24)
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=scenario,
            delay_s=0.0,
            loss_rate=0.0,
            name_base=name_base,
            simulate_days=1,
            cost_model="linear",  # quadratic toy: Eq. 16 clipping breaks ΣP (deviations.md)
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

    async def test_runs_under_message_loss(self, tmp_path):
        """With 5% of messages lost the run still finishes and passes the
        power-balance check (strict): missing values are replaced by the last
        received ones and lost results are requested again."""
        name_base = str(tmp_path / "fdgdm")
        await execute_test_case(
            scenario=build_toy_network(periods=24),
            loss_rate=0.05,
            seed=1,
            name_base=name_base,
            simulate_days=1,
            cost_model="linear",  # quadratic toy: Eq. 16 clipping breaks ΣP (deviations.md)
            strict=True,
        )
        assert os.path.exists(f"{name_base}-df.csv")


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


class TestEq24Stopping:
    async def test_default_stops_early_with_the_same_cost(self, tmp_path, caplog):
        import logging
        import re

        from energy_scheduling_benchmark.scenarios.fdgdm import execute_test_case

        def run(name, **kwargs):
            return execute_test_case(
                scenario=build_toy_network(periods=24),
                name_base=str(tmp_path / name),
                simulate_days=1,
                **kwargs,
            )

        caplog.set_level(logging.INFO)
        await run("early")
        early_iters = {
            int(m)
            for m in re.findall(r"FDGDM finished .*iterations=(\d+)", caplog.text)
        }
        caplog.clear()
        await run("full", tol_ratio=None)
        full_iters = {
            int(m)
            for m in re.findall(r"FDGDM finished .*iterations=(\d+)", caplog.text)
        }
        assert full_iters == {300}

        assert early_iters and max(early_iters) < 300
        early = pd.read_csv(tmp_path / "early-df.csv", index_col=0)["cost:total"].sum()
        full = pd.read_csv(tmp_path / "full-df.csv", index_col=0)["cost:total"].sum()
        assert early == pytest.approx(full, rel=1e-3)

    def test_cli_default_is_paper_ratio_and_zero_disables(self):
        from energy_scheduling_benchmark.scenarios._common import (
            build_scenario_argparser,
        )
        from energy_scheduling_benchmark.scenarios.fdgdm import (
            PAPER_TOL_RATIO,
            _add_fdgdm_args,
        )

        parser = build_scenario_argparser(
            None, default_name_base="x", extra_args=_add_fdgdm_args
        )
        assert parser.parse_args([]).tol_ratio == PAPER_TOL_RATIO
        assert parser.parse_args(["--tol-ratio", "0"]).tol_ratio == 0.0
