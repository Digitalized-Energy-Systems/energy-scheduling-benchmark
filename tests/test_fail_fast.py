"""An exception inside an agent aborts the run (TODO Later: robustness)."""

from __future__ import annotations

import time

import pytest

from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios._common import AgentError


async def test_failing_central_dispatch_solver_aborts(tmp_path, monkeypatch):
    """Seen 2026-10-16: appsi_highs rejected a QP inside the leader and the
    simulation kept running instead of failing."""
    import energy_scheduling_benchmark.scenarios.central_dispatch as cd

    def broken(*args, **kwargs):
        raise ValueError("solver exploded")

    monkeypatch.setattr(cd, "solve_multiperiod_dispatch", broken)
    with pytest.raises(AgentError, match="solver exploded"):
        await cd.execute_test_case(
            scenario=build_toy_network(periods=72),
            name_base=str(tmp_path / "cd"),
            simulate_days=3,
        )
    assert not (tmp_path / "cd-df.csv").exists()


async def test_failing_actor_in_peer_algorithm_aborts(tmp_path, monkeypatch):
    import energy_scheduling_benchmark.scenarios.consensus as cons

    calls = {"n": 0}
    original = cons.LinearCostEconomicDispatchConsensusActor.project_power

    def flaky(self, lam, data):
        calls["n"] += 1
        if calls["n"] > 20:
            raise ZeroDivisionError("actor broke mid-run")
        return original(self, lam, data)

    monkeypatch.setattr(
        cons.LinearCostEconomicDispatchConsensusActor, "project_power", flaky
    )
    start = time.perf_counter()
    with pytest.raises(AgentError, match="actor broke mid-run"):
        await cons.execute_test_case(
            scenario=build_toy_network(periods=24),
            name_base=str(tmp_path / "c"),
            simulate_days=1,
        )
    assert time.perf_counter() - start < 60


async def test_healthy_run_unaffected(tmp_path):
    import energy_scheduling_benchmark.scenarios.central_dispatch as cd

    await cd.execute_test_case(
        scenario=build_toy_network(periods=24),
        name_base=str(tmp_path / "cd"),
        simulate_days=1,
    )
    assert (tmp_path / "cd-df.csv").exists()


async def test_caller_task_is_not_left_cancelled(tmp_path, monkeypatch):
    """The watch cancels the stepping child task only, never the caller."""
    import asyncio

    import energy_scheduling_benchmark.scenarios.central_dispatch as cd

    def broken(*args, **kwargs):
        raise ValueError("solver exploded")

    monkeypatch.setattr(cd, "solve_multiperiod_dispatch", broken)
    with pytest.raises(AgentError):
        await cd.execute_test_case(
            scenario=build_toy_network(periods=24),
            name_base=str(tmp_path / "cd"),
            simulate_days=1,
        )
    assert asyncio.current_task().cancelling() == 0
    await asyncio.sleep(0)  # no pending cancellation fires here
