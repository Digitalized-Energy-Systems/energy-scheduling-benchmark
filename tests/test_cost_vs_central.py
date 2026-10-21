"""The benchmark's primary metric: a balanced distributed run cannot beat the
central-dispatch optimum (same network, horizon and bounds).

A run may serve slightly less than demand within the 1 % balance tolerance
and look cheaper for it (decision Q16; consensus on the toy network: 5 MWh
unmet, 0.28 % below central, 2026-10-21). Valued at the dearest unit's
marginal cost, that unmet energy must cover the whole saving, otherwise the
run used looser unit bounds than central dispatch. The upper limit only
catches runaway runs (FDGDM: +5.5 %).
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import build_toy_network

SCENARIOS = ["consensus", "admm", "diffusion", "exact_diffusion", "fdgdm", "deed_admm"]


async def _run(tmp_path, scenario: str) -> tuple[float, float]:
    """``(total cost, unmet demand in MWh)`` of *scenario* on the toy network."""
    module = importlib.import_module(
        f"energy_scheduling_benchmark.scenarios.{scenario}"
    )
    name_base = tmp_path / scenario
    await module.execute_test_case(
        scenario=build_toy_network(periods=24),
        name_base=str(name_base),
        simulate_days=1,
        strict=True,
    )
    df = pd.read_csv(f"{name_base}-df.csv", index_col=0)
    gap = df.filter(like="target:").iloc[:, 0] - df.filter(like="P:").sum(axis=1)
    return float(df["cost:total"].sum()), float(gap.clip(lower=0.0).sum())


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_cost_is_not_below_central_dispatch(tmp_path, scenario):
    central, _ = await _run(tmp_path, "central_dispatch")
    cost, unmet_mwh = await _run(tmp_path, scenario)
    dearest = float(build_toy_network(periods=1).net.generators.marginal_cost.max())
    assert cost + unmet_mwh * dearest >= central * (1 - 1e-6), (
        f"{scenario}: {cost:.1f} + {unmet_mwh:.2f} MWh unmet vs central {central:.1f}"
    )
    assert cost <= central * 1.10, f"{scenario}: {cost:.1f} vs central {central:.1f}"
