"""Diffusion after de Azevedo et al. 2017 (S8a, docs/diffusion-plan.md) against central dispatch."""

import json

import numpy as np
import pytest
from distributed_resource_optimization.algorithm.firstorder.diffusion import (
    diffusion as dro,
)

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.algorithms import diffusion
from energy_scheduling_benchmark.run import load_network, main

COMMON = ["--network", "toy", "--hours", "24", "--topology", "grid"]


def _meta(root, algorithm: str) -> dict:
    return json.loads(next(root.glob(f"toy/{algorithm}/*/meta.json")).read_text())


class _Observe:
    """Minimal behaviour: one bus of the prepared toy network."""

    def __init__(self, net, bus):
        self.c = {k: c for k, c in net.components.items() if c.bus == bus}

    def observe(self, aid, what, cid=None):
        if what == "components":
            return list(self.c)
        c = self.c[cid]
        if what == "statics":
            return {"kind": c.kind, "bus": c.bus, "cost": c.cost, **c.params}
        return {"p_min": c.p_min, "p_max": c.p_max, "demand": c.p_max}[what]


def test_sweep_grid_has_32_candidates():
    net = load_network(datasets.prepared_path("toy"))
    grid, note = diffusion.sweep_grid(net, net, "grid")
    assert len(grid) == 32 and {"mu", "penalty", "mu_is"} == set(grid[0])
    assert len({(c["penalty"], c["mu_is"]) for c in grid}) == 4


def test_gradient_is_the_marginal_cost_inside_and_penalised_outside():
    # eq. (23) (D-231): inside what the bus can produce ∇F = the exact dispatch's marginal cost
    # (finite differences of its cost), outside + 2ηα·(w − w̄)
    net = load_network(datasets.prepared_path("toy"))
    bus = net.components["gas"].bus
    actor = diffusion.BusActor(
        _Observe(net, bus), bus, len(net.time_index), penalty=10.0
    )
    w = np.full(len(net.time_index), 30.0)
    p, lam, w_bar = actor._dispatch(w)
    np.testing.assert_allclose(w_bar, w, atol=1e-5)  # producible: no penalty
    h = 1e-3
    up = actor._cost(actor._dispatch(w + h)[0]) - actor._cost(p)
    assert up / h == pytest.approx(lam.sum(), rel=1e-2)
    far = np.full_like(w, 1e4)
    _, lam_far, w_bar_far = actor._dispatch(far)
    np.testing.assert_allclose(
        actor.gradient(far), lam_far + 2 * 10.0 * (far - w_bar_far), rtol=1e-9
    )
    assert np.all(w_bar_far < far)
    # at a limit λ is the marginal cost there (coal: p_min = 0 → c₁ = 50 €/MWh); np.clip put
    # w̄ exactly on the limit, where the balance dual is not unique (audit 2026-11-24: −1424)
    coal = diffusion.BusActor(_Observe(net, "bus1"), "bus1", len(net.time_index), 10.0)
    _, lam_min, _ = coal._dispatch(np.zeros(len(net.time_index)))
    np.testing.assert_allclose(lam_min, net.components["coal"].cost[0], rtol=1e-4)


def test_printed_nudge_diverges_on_toy(tmp_path):
    # eq. (21) as printed (D-226) with eq. (12) weights: the estimate's sum error grows as
    # (A − I)^k until the bus QP fails — a result, "diverged"
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
    # evidence that the bus dispatch, nudge and information sharing are right: with (I + A)/2
    # (eigenvalues ≥ 0) the printed nudge is stable; the gap is the O(μ) constant-step offset
    monkeypatch.setattr(dro, "metropolis", lambda n_i, n_j: 0.5 / max(n_i, n_j))
    out = ["--out", str(tmp_path)]
    main(["--algorithm", "central", *COMMON, *out])
    sets = ["--set", "mu=0.005", "--set", "penalty=1000", "--set", "max_iter=4000"]
    main(["--algorithm", "diffusion", "--tuning", "paper", *COMMON, *sets, *out])
    central, run = _meta(tmp_path, "central"), _meta(tmp_path, "diffusion")
    assert run["stop_reason"] == "max_iter" and run["iterations"] == 4000
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=0.02)
    assert run["imbalance"]["shortfall_mwh"] + run["imbalance"]["surplus_mwh"] < 25


def test_information_sharing_never_converged_is_reported(tmp_path):
    # toy ring = an even ring: eq. (12) has eigenvalue −1, plain averaging (μ_IS = 0) never
    # converges, so the optimization never starts and nothing is dispatched (D-232)
    ring = ["--network", "toy", "--hours", "24", "--topology", "ring"]
    sets = ["--set", "mu_is=0", "--set", "max_iter=200"]
    main(
        [
            "--algorithm",
            "diffusion",
            "--tuning",
            "paper",
            *ring,
            *sets,
            "--out",
            str(tmp_path),
        ]
    )
    assert _meta(tmp_path, "diffusion")["stop_reason"] == diffusion.IS_NOT_CONVERGED
