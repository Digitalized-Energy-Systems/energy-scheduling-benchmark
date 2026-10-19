"""Storage acceptance (TODO Phase 2, Q15): with quadratic generator costs the
optimal storage schedule is unique, so every converged algorithm's battery
schedule must match central dispatch's.

Tolerances (relative L1 distance Σ|P − P_central| / Σ|P_central|) reflect each
algorithm's known stopping point, measured on the toy network (2026-10-19):
ADMM / DEED-ADMM 0.000, exact diffusion 0.007, diffusion 0.10 (constant-step
bias), consensus 0.21 (stops at the 1 % power-balance tolerance). FDGDM is not
included: its storage is scheduled outside the algorithm, and quadratic FDGDM
needs ``--slack`` (deviations.md). Linear costs are not tested: their optimal
storage schedule is not unique.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark.networks import build_toy_network

TOLERANCE = {
    "admm": 0.02,
    "deed_admm": 0.02,
    "exact_diffusion": 0.02,
    "diffusion": 0.15,
    "consensus": 0.30,
}


async def _battery(tmp_path, scenario: str) -> np.ndarray:
    module = importlib.import_module(
        f"energy_scheduling_benchmark.scenarios.{scenario}"
    )
    name_base = tmp_path / scenario
    await module.execute_test_case(
        scenario=build_toy_network(periods=24),
        name_base=str(name_base),
        simulate_days=1,
        cost_model="quadratic",
    )
    return pd.read_csv(f"{name_base}-df.csv", index_col=0)["P:batt0"].to_numpy()


@pytest.mark.parametrize("scenario", sorted(TOLERANCE))
async def test_battery_schedule_matches_central_dispatch(tmp_path, scenario):
    central = await _battery(tmp_path, "central_dispatch")
    schedule = await _battery(tmp_path, scenario)
    n = min(len(central), len(schedule))
    distance = np.abs(schedule[:n] - central[:n]).sum() / np.abs(central[:n]).sum()
    assert distance <= TOLERANCE[scenario], f"{scenario}: relative L1 {distance:.3f}"
    # The battery is actually used, not left idle.
    assert schedule[schedule > 0].sum() > 0.5 * central[central > 0].sum()
