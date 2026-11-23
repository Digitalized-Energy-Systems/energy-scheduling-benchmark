"""Diffusion after de Azevedo et al. 2017 (S8a, docs/diffusion-plan.md) against central dispatch."""

import json

import networkx as nx
import pytest

from energy_scheduling_benchmark.algorithms import diffusion
from energy_scheduling_benchmark.run import main

COMMON = ["--network", "toy", "--hours", "24", "--topology", "grid"]


def _meta(root, algorithm: str) -> dict:
    return json.loads(next(root.glob(f"toy/{algorithm}/*/meta.json")).read_text())


def test_eq12_weights_as_printed():
    # a_ij = 1/max(n_i, n_j), no +1 (D-216): the path's middle nodes keep no self-weight
    path = nx.path_graph(["a", "b", "c", "d"])
    assert diffusion.weights(path, "a") == {"b": 0.5}
    assert diffusion.weights(path, "b") == {"a": 0.5, "c": 0.5}


def test_sweep_grid_has_32_candidates():
    from energy_scheduling_benchmark import datasets
    from energy_scheduling_benchmark.run import load_network

    net = load_network(datasets.prepared_path("toy"))
    grid, note = diffusion.sweep_grid(net, net, "grid")
    assert len(grid) == 32 and {"mu", "penalty", "mu_is"} == set(grid[0])
    assert len({(c["penalty"], c["mu_is"]) for c in grid}) == 4


def test_printed_nudge_diverges_on_toy(tmp_path):
    # eq. (21) as printed (D-226) with eq. (12) weights: the estimate's sum error grows as
    # (A − I)^k (toy grid: ×~1.45 per round) until the bus QP fails — a result, "diverged"
    main(
        [
            "--algorithm",
            "diffusion",
            "--tuning",
            "paper",
            *COMMON,
            "--out",
            str(tmp_path),
        ]
    )
    assert _meta(tmp_path, "diffusion")["stop_reason"] == "diverged"


def test_lazy_weights_reach_central(tmp_path, monkeypatch):
    # evidence that the bus QP, nudge and IS pass are right: with (I + A)/2 (eigenvalues ≥ 0)
    # the printed nudge is stable; the gap is the O(μ) offset + the penalty slack λ/(2ηα)
    eq12 = diffusion.weights
    monkeypatch.setattr(
        diffusion, "weights", lambda g, aid: {j: a / 2 for j, a in eq12(g, aid).items()}
    )
    out = ["--out", str(tmp_path)]
    main(["--algorithm", "central", *COMMON, *out])
    sets = ["--set", "mu=0.005", "--set", "penalty=1000", "--set", "max_iter=4000"]
    main(["--algorithm", "diffusion", "--tuning", "paper", *COMMON, *sets, *out])
    central, run = _meta(tmp_path, "central"), _meta(tmp_path, "diffusion")
    assert run["stop_reason"] == "max_iter" and run["iterations"] == 4000
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=0.02)
    assert run["imbalance"]["shortfall_mwh"] + run["imbalance"]["surplus_mwh"] < 25
