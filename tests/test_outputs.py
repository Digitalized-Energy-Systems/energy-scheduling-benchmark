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

    write_run(
        tmp_path,
        behavior,
        world,
        communication,
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
    assert list(csv.columns) == units + (["slack"] if slack else []) + ["residual"]
    assert csv["wind"].isna().all()  # never set a schedule
    if slack:
        np.testing.assert_allclose(csv["residual"], 0.0)
        np.testing.assert_allclose(csv["slack"], -behavior.balance())
    else:
        np.testing.assert_allclose(csv["residual"], behavior.balance())

    meta = json.loads((tmp_path / "meta.json").read_text())
    gas, coal = net.components["gas"].cost, net.components["coal"].cost
    per_step = gas[0] * 50 + gas[1] * 50**2 + coal[0] * 10 + coal[1] * 10**2
    assert meta["total_cost"] == pytest.approx(steps * per_step)
    assert meta["messages"] == {"sent": 0, "delivered": 0, "lost": 0}
    assert meta["network_sha256"] is None
    assert meta["inputs"] == {"network": "toy", "seed": 0}
