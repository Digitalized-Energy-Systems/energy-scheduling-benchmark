import json

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.outputs import run_dir, write_run
from energy_scheduling_benchmark.world import build_world


def test_run_dir():
    start = pd.Timestamp("2019-01-14")
    assert run_dir("out", "toy", "admm", "ring", 0.1, 3, start, 168).as_posix() == (
        "out/toy/admm/ring_loss0.1_seed3_from20190114T00_168h"
    )


@pytest.mark.parametrize("slack", [False, True])
async def test_write_run(tmp_path, slack):
    net = datasets.load("toy")
    world, behavior, agents, communication, _ = build_world(net, slack=slack)
    steps = len(net.time_index)
    behavior.act(agents["gas"].aid, "set_schedule", np.full(steps, 50.0))
    behavior.act(agents["coal"].aid, "set_schedule", np.full(steps, 10.0))
    behavior.act(agents["battery"].aid, "set_schedule", np.full(steps, -1.0))
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
        slack=slack,
    )

    csv = pd.read_csv(tmp_path / "schedule.csv", index_col="time")
    units = ["gas", "coal", "wind", "battery"]
    extra = ["slack"] if slack else []
    assert list(csv.columns) == units + extra + ["residual", "soc:battery"]
    assert csv["wind"].isna().all()  # never set a schedule
    battery = net.components["battery"]
    e0, eta = (
        battery.params["state_of_charge_initial"],
        battery.params["efficiency_store"],
    )
    np.testing.assert_allclose(csv["soc:battery"], e0 + eta * np.arange(1, steps + 1))
    if slack:
        np.testing.assert_allclose(csv["residual"], 0.0)
        np.testing.assert_allclose(csv["slack"], -behavior.balance())
    else:
        np.testing.assert_allclose(csv["residual"], behavior.balance())

    meta = json.loads((tmp_path / "meta.json").read_text())
    gas, coal = net.components["gas"].cost, net.components["coal"].cost
    per_step = gas[0] * 50 + gas[1] * 50**2 + coal[0] * 10 + coal[1] * 10**2
    imported = np.maximum(-behavior.balance(), 0).sum() if slack else 0.0
    assert meta["slack"]["import_mwh"] == pytest.approx(imported)
    assert meta["slack"]["cost"] == pytest.approx(4000 * imported)
    assert meta["total_cost"] == pytest.approx(steps * per_step + 4000 * imported)
    assert meta["messages"] == messages
    assert meta["network_sha256"] is None
    assert meta["inputs"] == {"network": "toy", "seed": 0}


async def test_no_schedules_writes_null_cost(tmp_path):
    _, behavior, _, _, _ = build_world(datasets.load("toy"))
    write_run(
        tmp_path,
        behavior,
        messages={},
        inputs={},
        network_file=None,
        algorithm="central",
        iterations=0,
        stop_reason="infeasible",
        wall_time_s=0.0,
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["total_cost"] is None and meta["slack"]["cost"] is None
