"""Exact diffusion after Yuan et al. 2019 Alg. 1 on the ED dual (S8b, docs/exact-diffusion-plan.md)
against central dispatch."""

import json

import numpy as np
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.algorithms import exact_diffusion
from energy_scheduling_benchmark.run import load_network, main


def _meta(root, algorithm: str) -> dict:
    return json.loads(next(root.glob(f"toy/{algorithm}/*/meta.json")).read_text())


def test_sweep_grid_has_32_step_sizes():
    net = load_network(datasets.prepared_path("toy"))
    grid, note = exact_diffusion.sweep_grid(net, net, "grid")
    mus = [c["mu"] for c in grid]
    assert len(grid) == 32 and set(grid[0]) == {"mu"} and mus == sorted(mus)
    # μσ² < 2 at the stiffest toy bus (35 MW per €/MWh), the grid's scale (cf. Lemma 6)
    assert mus[-1] * 35.15 < 2
    # D-261: 16 pilot steps over four decades; a pilot top caps the grid
    pilot, _ = exact_diffusion.sweep_grid(net, net, "grid", pilot=True)
    assert len(pilot) == 16 and pilot[-1]["mu"] / pilot[0]["mu"] == pytest.approx(
        1e4, rel=1e-2
    )
    capped, _ = exact_diffusion.sweep_grid(net, net, "grid", top={None: 0.3})
    assert len(capped) == 32 and capped[-1]["mu"] == pytest.approx(0.3)


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_toy_reaches_central(tmp_path, topology):
    # the correction removes plain diffusion's O(μ) offset: the exact optimum, exact balance
    # (S8a diffusion on the same toy: −0.5 … −1.6 %, 15–25 MWh shortfall)
    common = ["--network", "toy", "--hours", "24", "--topology", topology]
    out = ["--out", str(tmp_path)]
    main(["--algorithm", "central", *common, *out])
    sets = ["--set", "mu=0.01", "--set", "max_iter=4000"]
    main(["--algorithm", "exact_diffusion", "--tuning", "paper", *common, *sets, *out])
    central, run = _meta(tmp_path, "central"), _meta(tmp_path, "exact_diffusion")
    assert run["stop_reason"] == "max_iter" and run["iterations"] == 4000
    assert run["total_cost"] == pytest.approx(central["total_cost"], rel=1e-6)
    assert run["imbalance"]["shortfall_mwh"] + run["imbalance"]["surplus_mwh"] < 1e-4
    # the trace ends at the reported schedule: rows 0 … max_iter, the last = the final λ's answer
    folder = next(tmp_path.glob("toy/exact_diffusion/*"))
    last = (folder / "trace.csv").read_text().splitlines()[-1].split(",")
    assert int(last[0]) == 4000
    assert float(last[3]) == pytest.approx(run["imbalance"]["shortfall_mwh"], abs=1e-9)
    assert float(last[4]) == pytest.approx(run["imbalance"]["surplus_mwh"], abs=1e-9)


def test_gradient_is_the_innovation():
    # D-257: ∇J_k(λ) = P_k(λ) − D_k (Danskin), so λ rises while demand exceeds supply
    class Observe:
        def observe(self, aid, what, cid=None):
            if what == "components":
                return ["g", "l"]
            if what == "statics":
                kind = "gen" if cid == "g" else "load"
                return {"kind": kind, "bus": "b", "cost": (10.0, 0.5), "p_nom": 50.0}
            return {"p_min": 0.0, "p_max": 50.0, "demand": 30.0}[what]

    actor = exact_diffusion.BusActor(Observe(), "b", 2, trace=None)
    lam = np.array([0.0, 40.0])  # gen answers 0 and (40 − 10)/(2·0.5) = 30 MW
    np.testing.assert_allclose(actor.gradient(lam), [-30.0, 0.0])
