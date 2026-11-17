"""Consensus + innovations (HKW15) against central dispatch; its step-size helpers."""

import json

import pandas as pd
import pytest

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.algorithms import consensus
from energy_scheduling_benchmark.run import main, without_storage


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_consensus_reaches_central_cost(tmp_path, topology):
    common = ["--hours", "24", "--topology", topology, "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(
        ["--algorithm", "consensus", *common]
    )  # toy's manual set (tuned_parameters.toml)
    central, consensus = (
        json.loads(next(tmp_path.glob(f"toy/{a}/*/meta.json")).read_text())
        for a in ("central", "consensus")
    )
    assert consensus["stop_reason"] == "converged"
    assert consensus["inputs"]["consensus"]["tuning"] == "manual"
    assert consensus["total_cost"] == pytest.approx(central["total_cost"], rel=1e-3)
    # Q-A3: time to solution (≥ one 10 ms hop per iteration) and the per-iteration trace, whose
    # last row is near the final state (cost without grid exchange = total_cost here, islanded;
    # sparse topologies: buses end 1–3 iterations apart, so not exact — Jonas 2026-11-13)
    assert central["sim_time_s"] is None
    assert consensus["sim_time_s"] >= 0.01 * consensus["iterations"]
    trace = pd.read_csv(next(tmp_path.glob("toy/consensus/*/trace.csv")))
    assert list(trace.iteration) == list(range(len(trace)))
    assert trace.iteration.iloc[-1] >= consensus["iterations"] - 1
    assert trace.cost.iloc[-1] == pytest.approx(consensus["total_cost"], rel=1e-4)
    assert trace.imbalance_mwh.iloc[-1] < trace.imbalance_mwh.iloc[0]


def test_consensus_per_agent_steps_reach_central_cost(tmp_path):
    # S3: α and β scaled per bus keep the optimum (scaling only α would bias it, plan §2)
    common = ["--hours", "24", "--topology", "ring", "--out", str(tmp_path)]
    main(["--algorithm", "central", *common])
    main(["--algorithm", "consensus", *common, "--set", "agent_scale=capacity"])
    central, scaled = (
        json.loads(next(tmp_path.glob(f"toy/{a}/*/meta.json")).read_text())
        for a in ("central", "consensus")
    )
    assert scaled["inputs"]["consensus"]["agent_scale"] == "capacity"
    assert scaled["stop_reason"] == "converged"
    assert scaled["total_cost"] == pytest.approx(central["total_cost"], rel=1e-3)


def test_consensus_settings_get_own_folders(tmp_path):
    common = ["--algorithm", "consensus", "--hours", "2", "--out", str(tmp_path)]
    main([*common, "--tuning", "paper", "--set", "max_iter=2"])
    main([*common, "--tuning", "paper", "--set", "max_iter=3"])
    names = sorted(p.name for p in tmp_path.glob("toy/consensus/*"))
    assert len(names) == 2 and all("_paper_set" in n for n in names)
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        main([*common, "--set", "kappa"])


def test_lossy_consensus_finishes_with_deadline_rounds(tmp_path):
    # S5: 20 % of the λ messages lost — without round deadlines one lost λ stalled the whole run;
    # control messages always arrive; the same seed gives the same run
    common = ["--algorithm", "consensus", "--hours", "2", "--topology", "ring"]
    common += ["--loss-rate", "0.2", "--set", "max_iter=600"]
    metas = []
    for out in ("a", "b"):
        main([*common, "--out", str(tmp_path / out)])
        metas.append(
            json.loads(
                next(tmp_path.glob(f"{out}/toy/consensus/*/meta.json")).read_text()
            )
        )
    a, b = metas
    assert a["stop_reason"] in ("converged", "max_iter")  # not "stalled"
    assert a["messages"]["lost"] > 0
    assert (
        a["messages"]["rounds_by_deadline"] > 0 and a["messages"]["missing_edges"] > 0
    )
    assert a["total_cost"] == b["total_cost"] and a["messages"] == b["messages"]


def test_max_sensitivity_toy():

    # bus0 gas 1/(2·0.033) + battery 1/(2·0.025); the wind bus (40) is pre-dispatched (S1)
    toy = datasets.load("toy")
    assert consensus.max_sensitivity(toy) == pytest.approx(1 / 0.066 + 20)
    # per-agent steps (S3): bus0 (s = 0.75) still the worst scaled bus; coal bus1 1.5 · 5.45
    assert consensus.max_sensitivity(toy, "capacity") == pytest.approx(
        0.75 * (1 / 0.066 + 20)
    )


def test_sweep_grid_toy():

    toy = datasets.load("toy")
    # bus graph: complete K4 → 4, ring C4 → 4, grid = path P4 → 2 + √2
    assert consensus.laplacian_max(toy, "complete") == pytest.approx(4)
    assert consensus.laplacian_max(toy, "grid") == pytest.approx(2 + 2**0.5)
    candidates, note = consensus.sweep_grid(toy, toy, "grid")
    assert len(candidates) == 32  # Q-A5: 32 per algorithm
    assert sum(c["steps"] == "constant" for c in candidates) == 16
    assert {c["alpha_exp"] for c in candidates if c["steps"] == "decaying"} == {
        0.6,
        0.98,
    }
    assert "λ_max(L)" in note


def test_agent_factors_toy():

    toy = datasets.load("toy")
    # dispatchable capacity: bus0 gas 100 + battery 20, bus1 coal 60 → median 90; bus2 (wind
    # only) and bus3 (load only) get the median factor of the others
    f = consensus.agent_factors(toy, "capacity")
    assert f == pytest.approx({"bus0": 0.75, "bus1": 1.5, "bus2": 1.125, "bus3": 1.125})
    assert set(consensus.agent_factors(toy, "none").values()) == {1.0}
    with pytest.raises(ValueError, match="agent_scale"):
        consensus.agent_factors(toy, "size")


def test_factors_come_from_the_network_with_storage():

    toy = datasets.load("toy")
    # without the battery bus0 would get 0.8 (median 80 of 100/60); the tuned set uses 0.75
    no_storage = consensus.max_sensitivity(without_storage(toy), "capacity", toy)
    assert no_storage == pytest.approx(0.75 / 0.066)
