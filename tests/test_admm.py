"""Exchange ADMM (S6a, docs/admm-exchange-plan.md) against central dispatch."""

import json

import numpy as np
import pandas as pd
import pytest
from distributed_resource_optimization import next_hops
from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    Component,
    Network,
    load_network,
    save_network,
)

from energy_scheduling_benchmark import admm, datasets
from energy_scheduling_benchmark.run import main
from energy_scheduling_benchmark.world import topology_graph

TIGHT = ["--set", "eps_abs=1e-7", "--set", "eps_rel=1e-7"]
FIXED = ["--set", "adaptive=false", "--set", "rho=0.1"]


def _meta(root, network: str, algorithm: str) -> dict:
    return json.loads(next(root.glob(f"{network}/{algorithm}/*/meta.json")).read_text())


def _hops(network, topology: str) -> int:
    """Σ over the bus agents of the leader's hops to it (its own messages to itself: 1)."""
    routes = next_hops(topology_graph(network, topology))  # aids = bus names
    host = min(routes)  # world.leader: the first bus
    total = 0
    for aid in routes:
        n, a = 0, aid
        while a != host:
            a, n = routes[a][host], n + 1
        total += max(n, 1)
    return total


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_admm_reaches_central_cost_on_toy(tmp_path, topology):
    common = ["--hours", "24", "--topology", topology, "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "admm", "--tuning", "paper", *common, *TIGHT])
    central, run = _meta(tmp_path, "toy", "central"), _meta(tmp_path, "toy", "admm")
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-6)
    # one request + one reply per bus and iteration, each hop counted, plus the stop message
    it = run["iterations"]
    hops = _hops(datasets.load("toy"), topology)
    assert run["messages"]["sent"] == (2 * it + 1) * hops
    assert run["messages"]["missing_replies"] == 0
    assert run["sim_time_s"] >= 2 * 0.01 * it
    trace = pd.read_csv(next(tmp_path.glob("toy/admm/*/trace.csv")))
    assert list(trace.iteration) == list(range(it))
    assert trace.cost.iloc[-1] == pytest.approx(run["total_cost"], rel=1e-6)
    assert trace.lambda_spread.iloc[-1] < 1e-3  # every bus at the same marginal cost


def test_admm_reaches_central_cost_on_synthetic_10(tmp_path):
    common = [
        "--network",
        "synthetic_10",
        "--start",
        "2019-01-09",
        "--out",
        str(tmp_path),
    ]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "admm", "--tuning", "paper", *common, *FIXED, *TIGHT])
    central, run = (_meta(tmp_path, "synthetic_10", a) for a in ("central", "admm"))
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-6)


def _two_bus(tmp_path, battery: bool) -> str:
    """Bus a: gas (c₁ 10, c₂ 0.1, 0…10 MW) + load 4 / 8 MW; bus b: wind 6 / 1 MW (hour 0 has a
    2 MWh surplus) and optionally a battery that can move it into hour 1."""
    a = lambda *v: np.array(v, dtype=float)  # noqa: E731
    comps = {
        "gas": Component(
            THERMAL, "a", a(0, 0), a(10, 10), (10.0, 0.1), {"p_nom": 10.0}
        ),
        "load": Component(LOAD, "a", a(4, 8), a(4, 8)),
        "wind": Component(RENEWABLE, "b", a(0, 0), a(6, 1), (0.0, 0.0), {"p_nom": 9.0}),
    }
    if battery:
        params = {"p_nom": 5.0, "max_hours": 1.0, "efficiency_store": 0.9}
        params |= {"efficiency_dispatch": 0.9, "state_of_charge_initial": 0.0}
        comps["battery"] = Component(
            STORAGE, "b", a(-5, -5), a(5, 5), (1.0, 0.01), params
        )
    path = tmp_path / f"two_bus_{battery}.nc"
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    save_network(Network(idx, comps, [("a", "b")]), path)
    return str(path)


@pytest.mark.parametrize("battery", [False, True])
def test_admm_surplus_and_storage_match_central(tmp_path, battery):
    common = ["--network", _two_bus(tmp_path, battery), "--hours", "2"]
    common += ["--topology", "complete", "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "admm", "--tuning", "paper", *common, *FIXED, *TIGHT])
    name = f"two_bus_{battery}"
    central, run = _meta(tmp_path, name, "central"), _meta(tmp_path, name, "admm")
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-5)
    surplus = run["imbalance"]["surplus_mwh"]
    assert surplus == pytest.approx(central["imbalance"]["surplus_mwh"], abs=1e-3)
    assert surplus == pytest.approx(
        0.0 if battery else 2.0, abs=0.1
    )  # stored, else reported


def test_lossy_admm_finishes_and_repeats(tmp_path):
    # 20 % of the ADMM messages lost (D-134): a missing reply keeps the bus's last x, the round
    # ends at 4·hops·delay_s (D-135); the same seed gives the same run
    common = ["--algorithm", "admm", "--hours", "2", "--topology", "ring"]
    common += ["--loss-rate", "0.2", "--set", "max_iter=600", "--tuning", "paper"]
    metas = []
    for out in ("a", "b"):
        main([*common, "--out", str(tmp_path / out)])
        metas.append(_meta(tmp_path / out, "toy", "admm"))
    a, b = metas
    assert a["stop_reason"] in ("converged", "max_iter")  # not "stalled"
    assert a["messages"]["lost"] > 0 and a["messages"]["missing_replies"] > 0
    assert a["total_cost"] == b["total_cost"] and a["messages"] == b["messages"]


def test_lossy_admm_on_a_one_bus_network(tmp_path):
    # review 2026-11-16: 0 hops gave a round length of 0, every reply "missing", and the flat
    # start then passed the stop test at x = 0
    common = ["--network", "base_s_1_elec_2019", "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(
        [
            "--algorithm",
            "admm",
            "--tuning",
            "paper",
            *common,
            *FIXED,
            "--loss-rate",
            "0.05",
        ]
    )
    central, run = (
        _meta(tmp_path, "base_s_1_elec_2019", a) for a in ("central", "admm")
    )
    assert run["stop_reason"] == "converged" and run["iterations"] > 1
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-3)


def test_sweep_grid_is_32_fixed_rho_around_the_curvature():
    net = load_network(datasets.prepared_path("synthetic_10"))
    grid, note = admm.sweep_grid(net, net, "complete")
    rho = np.array([c["rho"] for c in grid])
    assert len(grid) == 32 and not any(c["adaptive"] for c in grid)
    assert rho[-1] / rho[0] == pytest.approx(1e6, rel=1e-3)
    assert "median 2c₂" in note
