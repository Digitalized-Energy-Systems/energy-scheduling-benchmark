import json

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.outputs import run_dir, write_run
from energy_scheduling_benchmark.world import build_world


def test_run_dir():
    start = pd.Timestamp("2019-01-14")
    assert run_dir(
        "out", "toy", "admm", "ring", 0.1, 3, start, 168, True
    ).as_posix() == ("out/toy/admm/ring_loss0.1_seed3_from20190114T00_168h_storage")
    assert run_dir(
        "o", "toy", "consensus", "ring", 0, 0, start, 24, False, "paper"
    ).name == ("ring_loss0_seed0_from20190114T00_24h_nostorage_paper")


@pytest.mark.parametrize("grid", [False, True])
async def test_write_run(tmp_path, grid):
    net = datasets.load("toy")
    world, behavior, agents, communication, _ = build_world(net)
    steps = len(net.time_index)
    behavior.act(agents["bus0"].aid, "set_schedule", np.full(steps, 50.0), "gas")
    behavior.act(agents["bus1"].aid, "set_schedule", np.full(steps, 10.0), "coal")
    behavior.act(agents["bus0"].aid, "set_schedule", np.full(steps, -1.0), "battery")
    messages = {"sent": 3, "delivered": 2, "lost": 1}

    write_run(
        tmp_path,
        behavior,
        messages=messages,
        inputs={"network": "toy", "seed": 0},
        network_file=None,
        algorithm="dummy",
        iterations=1,
        stop_reason="done",
        wall_time_s=0.5,
        grid_exchange=grid,
    )

    csv = pd.read_csv(tmp_path / "schedule.csv", index_col="time")
    units = ["gas", "coal", "wind", "battery"]
    extra = ["grid_exchange"] if grid else []
    assert list(csv.columns) == units + extra + ["residual", "soc:battery"]
    assert csv["wind"].isna().all()  # never set a schedule
    battery = net.components["battery"]
    e0, eta = (
        battery.params["state_of_charge_initial"],
        battery.params["efficiency_store"],
    )
    np.testing.assert_allclose(csv["soc:battery"], e0 + eta * np.arange(1, steps + 1))
    if grid:
        np.testing.assert_allclose(csv["residual"], 0.0)
        np.testing.assert_allclose(csv["grid_exchange"], -behavior.balance())
    else:
        np.testing.assert_allclose(csv["residual"], behavior.balance())

    meta = json.loads((tmp_path / "meta.json").read_text())
    gas, coal = net.components["gas"].cost, net.components["coal"].cost
    per_step = gas[0] * 50 + gas[1] * 50**2 + coal[0] * 10 + coal[1] * 10**2
    per_step += battery.cost[1] * 1.0**2  # curvature band on the net power
    imported = np.maximum(-behavior.balance(), 0).sum() if grid else 0.0
    assert meta["grid_exchange"]["import_mwh"] == pytest.approx(imported)
    assert meta["grid_exchange"]["cost"] == pytest.approx(4000 * imported)
    demand = sum(c.p_max.sum() for c in net.components.values() if c.kind == "load")
    assert meta["grid_exchange"]["import_share"] == pytest.approx(imported / demand)
    final = csv["residual"]
    assert meta["imbalance"]["shortfall_mwh"] == pytest.approx(
        np.maximum(-final, 0).sum()
    )
    assert meta["imbalance"]["surplus_mwh"] == pytest.approx(np.maximum(final, 0).sum())
    assert meta["total_cost"] == pytest.approx(steps * per_step + 4000 * imported)
    assert meta["messages"] == messages
    assert meta["network_sha256"] is None
    assert meta["inputs"] == {"network": "toy", "seed": 0}


def _write_empty(tmp_path, algorithm, stop_reason, grid=True):
    _, behavior, _, _, _ = build_world(datasets.load("toy"))
    write_run(
        tmp_path,
        behavior,
        messages={},
        inputs={},
        network_file=None,
        algorithm=algorithm,
        iterations=0,
        stop_reason=stop_reason,
        wall_time_s=0.0,
        grid_exchange=grid,
    )
    csv = pd.read_csv(tmp_path / "schedule.csv", index_col="time")
    return json.loads((tmp_path / "meta.json").read_text()), csv, behavior


async def test_failed_central_writes_null_cost_and_grid_exchange(tmp_path):
    meta, csv, behavior = _write_empty(tmp_path, "central", "infeasible")
    assert meta["total_cost"] is None
    assert all(v is None for v in meta["grid_exchange"].values())
    assert csv["grid_exchange"].isna().all()
    np.testing.assert_allclose(csv["residual"], behavior.balance())  # = −demand, not 0


async def test_lossy_run_without_schedules_keeps_grid_cost(tmp_path):
    meta, csv, behavior = _write_empty(tmp_path, "dummy", "timeout")
    demand = -behavior.balance().sum()
    assert meta["grid_exchange"]["import_mwh"] == pytest.approx(demand)
    assert meta["grid_exchange"]["import_share"] == pytest.approx(1.0)
    assert meta["total_cost"] == pytest.approx(4000 * demand)
    assert meta["imbalance"] == {"shortfall_mwh": 0.0, "surplus_mwh": 0.0}


async def test_stalled_run_without_schedules_has_no_cost(tmp_path):
    """A diverged distributed run (nothing scheduled, no grid exchange) must not look like cost 0."""
    meta, _, _ = _write_empty(tmp_path, "consensus", "stalled", grid=False)
    assert meta["total_cost"] is None
