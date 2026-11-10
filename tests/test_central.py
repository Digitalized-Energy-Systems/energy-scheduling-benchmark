import json

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import LOAD, STORAGE, THERMAL, Component, Network

from energy_scheduling_benchmark import datasets, storage
from energy_scheduling_benchmark.central import solve_central
from energy_scheduling_benchmark.run import main, window

IDX = pd.date_range("2024-01-01", periods=2, freq="h")


def flat(v):
    return np.full(2, float(v))


def net(*extra, demand=(4.0, 8.0)):
    components = {
        "cheap": Component(THERMAL, "b", flat(0), flat(5), cost=(10.0, 0.0)),
        "dear": Component(THERMAL, "b", flat(0), flat(5), cost=(30.0, 0.0)),
        "load": Component(LOAD, "b", np.array(demand), np.array(demand)),
    }
    return Network(IDX, components | dict(extra), [])


def test_merit_order_lp():
    schedules, status, _ = solve_central(net())
    assert status == "optimal"
    np.testing.assert_allclose(schedules["cheap"], [4, 5], atol=1e-6)
    np.testing.assert_allclose(schedules["dear"], [0, 3], atol=1e-6)


def test_storage_shifts_cheap_energy():
    params = dict(
        p_nom=5.0,
        max_hours=2.0,
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
        state_of_charge_initial=0.0,
    )
    bat = Component(STORAGE, "b", flat(-5), flat(5), params=params)
    schedules, _, simultaneous = solve_central(net(("bat", bat)))
    # store the cheap unit's 1 MW surplus (losses make storing 'dear' energy not worth it)
    np.testing.assert_allclose(schedules["bat"], [-1, 0.81], atol=1e-6)
    # losses: never charge and discharge in the same hour
    assert simultaneous == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(storage.soc(bat, schedules["bat"]), [0.9, 0], atol=1e-6)


def test_islanded_infeasible_and_slack():
    schedules, status, _ = solve_central(net(demand=(4.0, 12.0)))
    assert status == "infeasible" and schedules == {}
    schedules, status, _ = solve_central(net(demand=(4.0, 12.0)), slack=True)
    assert status == "optimal"
    np.testing.assert_allclose(schedules["dear"], [0, 5], atol=1e-6)


def test_quadratic_toy_balances():
    toy = window(datasets.load("toy"), None, 24)
    schedules, status, _ = solve_central(toy)
    assert status == "optimal"
    demand = sum(c.p_max for c in toy.components.values() if c.kind == LOAD)
    np.testing.assert_allclose(sum(schedules.values()), demand, atol=1e-5)


def test_esb_run_central(tmp_path):
    main(["--algorithm", "central", "--hours", "24", "--out", str(tmp_path)])
    (out,) = tmp_path.glob("toy/central/*")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == "optimal"
    assert meta["messages"] == {"sent": 0, "delivered": 0, "lost": 0}
    assert meta["inputs"]["start"] == "2024-01-01 00:00:00"
    csv = pd.read_csv(out / "schedule.csv", index_col="time")
    np.testing.assert_allclose(csv["residual"], 0.0, atol=1e-5)
    assert not csv["soc:battery"].isna().any()


def test_window_bad_start():
    with pytest.raises(ValueError, match="not a snapshot"):
        window(datasets.load("toy"), "2030-01-01", 2)


def test_slack_never_worse_on_real_network(tmp_path):
    costs = {}
    for flags in ([], ["--slack"]):
        out = tmp_path / str(len(flags))
        args = ["--algorithm", "central", "--network", "base_s_1_elec_2019"]
        main([*args, "--hours", "168", "--out", str(out), *flags])
        (meta,) = out.rglob("meta.json")
        costs[len(flags)] = json.loads(meta.read_text())["total_cost"]
    assert costs[1] <= costs[0] * (1 + 1e-9)
