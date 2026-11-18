"""DEED-ADMM (S6b, docs/deed-admm-plan.md) against central dispatch."""

import json

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import load_network

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.algorithms import deed_admm
from energy_scheduling_benchmark.run import main

from .test_admm import _two_bus

TIGHT = ["--set", "eps_abs=1e-7", "--set", "eps_rel=1e-7"]


def _meta(root, network: str, algorithm: str) -> dict:
    return json.loads(next(root.glob(f"{network}/{algorithm}/*/meta.json")).read_text())


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_deed_admm_reaches_central_cost_on_toy(tmp_path, topology):
    common = ["--hours", "24", "--topology", topology, "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "deed_admm", "--tuning", "paper", *common, *TIGHT])
    central, run = (
        _meta(tmp_path, "toy", "central"),
        _meta(tmp_path, "toy", "deed_admm"),
    )
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-6)
    trace = pd.read_csv(next(tmp_path.glob("toy/deed_admm/*/trace.csv")))
    assert trace.iteration.iloc[-1] >= run["iterations"] - 3  # actors stop 1–3 apart
    assert trace.cost.iloc[-1] == pytest.approx(run["total_cost"], rel=1e-5)
    assert trace.lambda_spread.iloc[-1] < 1e-3  # one common price


def test_deed_admm_reaches_central_cost_on_synthetic_10(tmp_path):
    common = [
        "--network",
        "synthetic_10",
        "--start",
        "2019-01-09",
        "--out",
        str(tmp_path),
    ]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "deed_admm", "--tuning", "paper", *common, *TIGHT])
    central, run = (
        _meta(tmp_path, "synthetic_10", a) for a in ("central", "deed_admm")
    )
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-6)


@pytest.mark.parametrize("battery", [False, True])
def test_deed_admm_surplus_and_storage_match_central(tmp_path, battery, monkeypatch):
    buses = []
    finish = deed_admm.BusParticipant.finish
    monkeypatch.setattr(
        deed_admm.BusParticipant,
        "finish",
        lambda self: (buses.append(self), finish(self)),
    )
    common = ["--network", _two_bus(tmp_path, battery), "--hours", "2"]
    common += ["--topology", "complete", "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "deed_admm", "--tuning", "paper", *common, *TIGHT])
    name = f"two_bus_{battery}"
    central, run = _meta(tmp_path, name, "central"), _meta(tmp_path, name, "deed_admm")
    assert run["stop_reason"] == "converged"
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-5)
    surplus = run["imbalance"]["surplus_mwh"]
    assert surplus == pytest.approx(central["imbalance"]["surplus_mwh"], abs=1e-3)
    assert surplus == pytest.approx(
        0.0 if battery else 2.0, abs=0.1
    )  # stored, else reported
    # Thm 1 (η_c·η_d < 1): never charge and discharge in the same hour
    for bus in buses:
        for _, ch, dis in bus.stores.values():
            assert np.all(np.minimum(bus.p[ch], bus.p[dis]) < 1e-4)
    assert any(bus.stores for bus in buses) == battery


def test_lossy_deed_admm_finishes_and_repeats(tmp_path):
    # 20 % of the (λ, ξ) messages lost: round deadline 2·delay_s (Q-Z6 A); same seed, same run
    common = ["--algorithm", "deed_admm", "--hours", "2", "--topology", "ring"]
    common += ["--loss-rate", "0.2", "--set", "max_iter=600", "--tuning", "paper"]
    metas = []
    for out in ("a", "b"):
        main([*common, "--out", str(tmp_path / out)])
        metas.append(_meta(tmp_path / out, "toy", "deed_admm"))
    a, b = metas
    assert a["stop_reason"] in ("converged", "max_iter")  # not "stalled"
    assert a["messages"]["lost"] > 0 and a["messages"]["missing_edges"] > 0
    assert a["total_cost"] == b["total_cost"] and a["messages"] == b["messages"]


def test_sweep_grid_is_4_beta_by_8_gamma():
    net = load_network(datasets.prepared_path("synthetic_10"))
    grid, note = deed_admm.sweep_grid(net, net, "ring")
    beta = sorted({c["beta"] for c in grid})
    gamma = sorted({c["gamma"] for c in grid})
    assert len(grid) == 32 and len(beta) == 4 and len(gamma) == 8
    assert beta[0] == pytest.approx(0.6 / 4)  # ring: λ_max(L) = 4
    assert beta[-1] == pytest.approx(1.9 / 4)  # D-181: g up to 1.9 < 2
    assert gamma[-1] / gamma[0] == pytest.approx(1e6, rel=1e-3)
    assert "median 2c₂" in note
    assert deed_admm.SWEEP_RUN["max_iter"] == 10_000  # D-178
