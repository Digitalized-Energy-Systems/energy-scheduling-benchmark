import shutil
import tomllib

import pytest

from energy_scheduling_benchmark import consensus, tune


def test_write_section_replaces_only_its_table(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    shutil.copy(consensus.PARAMETERS, path)
    monkeypatch.setattr(consensus, "PARAMETERS", path)
    name = "consensus.toy.ring.sweep"
    tune.write_section(name, {"steps": "constant", "kappa": 0.01, "note": "a"})
    tune.write_section(name, {"steps": "constant", "kappa": 0.02, "note": "b"})
    table = tomllib.loads(path.read_text())["consensus"]
    assert table["toy"]["ring"]["sweep"] == {
        "steps": "constant",
        "kappa": 0.02,
        "note": "b",
    }
    assert table["toy"]["ring"]["manual"]["kappa"] == 0.03  # neighbours untouched
    assert table["alpha0"] == 0.055
    assert path.read_text().count(f"[{name}]") == 1


async def test_sweep_picks_a_converged_candidate():
    args = tune.argparse.Namespace(
        network="toy", topology="ring", steps="both", start=None, hours=6,
        alpha0="200", beta0="0.2", kappa="1.2,8", gamma="4", max_iter=1000,
        write=False, storage=True,
    )  # fmt: skip
    best = await tune.sweep(args)
    # toy max scaled bus dP/dλ = 0.75 · 35.15 (bus0, S3) → κ = 1.2/26.36 stable, 8/26.36 oscillates
    assert best["steps"] == "constant" and best["agent_scale"] == "capacity"
    assert best["kappa"] == pytest.approx(1.2 / (0.75 * 35.15), rel=1e-3)
    assert "best: constant" in best["note"]


def test_max_sensitivity_toy():
    from energy_scheduling_benchmark import datasets

    # bus0 gas 1/(2·0.033) + battery 1/(2·0.025); the wind bus (40) is pre-dispatched (S1)
    toy = datasets.load("toy")
    assert tune.max_sensitivity(toy) == pytest.approx(1 / 0.066 + 20)
    # per-agent steps (S3): bus0 (s = 0.75) still the worst scaled bus; coal bus1 1.5 · 5.45
    assert tune.max_sensitivity(toy, "capacity") == pytest.approx(
        0.75 * (1 / 0.066 + 20)
    )


def test_parameter_routes():
    import pytest

    assert (
        consensus.parameters("toy", "ring")["tuning"] == "manual"
    )  # default if present
    paper = consensus.parameters("toy", "ring", "paper")
    assert (paper["steps"], paper["alpha0"]) == ("decaying", 0.055)
    assert consensus.parameters("base_s_1_elec_2019", "grid")["tuning"] == "paper"
    with pytest.raises(ValueError):
        consensus.parameters("base_s_1_elec_2019", "grid", "sweep")
    assert consensus.parameters("toy", "ring", overrides={"kappa": 1.0})["kappa"] == 1.0
    with pytest.raises(ValueError, match="kapa"):
        consensus.parameters("toy", "ring", overrides={"kapa": 1.0})  # typo


async def test_sweep_writes_nothing_if_no_candidate_runs_through():
    args = tune.argparse.Namespace(
        network="toy", topology="ring", steps="constant", start=None, hours=6,
        alpha0="1", beta0="0.2", kappa="8", gamma="4", max_iter=500, write=True,
        storage=True,
    )  # fmt: skip
    before = consensus.PARAMETERS.read_text()
    assert await tune.sweep(args) is None  # κ = 0.2 diverges → "stalled"
    assert consensus.PARAMETERS.read_text() == before


async def test_sweep_writes_the_best_whatever_its_gap(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    shutil.copy(consensus.PARAMETERS, path)
    monkeypatch.setattr(consensus, "PARAMETERS", path)
    args = tune.argparse.Namespace(
        network="toy", topology="ring", steps="constant", start=None, hours=6,
        alpha0="1", beta0="0.2", kappa="0.01", gamma="4", max_iter=20, write=True,
        storage=True,
    )  # fmt: skip
    best = await tune.sweep(args)  # κ tiny: λ barely moves from 0 → gap far from 0
    gap = float(best["note"].split(": ")[-1].split("%")[0])
    assert abs(gap) > 1  # once (--max-gap 1 %) this was not written
    written = tomllib.loads(path.read_text())["consensus"]["toy"]["ring"]["sweep"]
    assert written["kappa"] == best["kappa"] and written["note"] == best["note"]


def test_score_zero_cost_window_and_unavoidable_surplus():
    # a window the renewables cover at 0 €/MWh: central and run both cost 0 → gap 0, no crash
    covered = {"stop_reason": "renewables_cover", "total_cost": 0.0}
    covered["imbalance"] = {"shortfall_mwh": 0.0, "surplus_mwh": 5.0}
    assert (
        tune.score(covered, covered, 100.0) == 0.0
    )  # central's surplus is not penalised
    run = {**covered, "stop_reason": "max_iter"}
    run["imbalance"] = {
        "shortfall_mwh": 0.0,
        "surplus_mwh": 3.0,
    }  # 2 MWh off the optimum
    assert tune.score(run, covered, 100.0) == pytest.approx(0.02)


def test_agent_factors_toy():
    from energy_scheduling_benchmark import datasets

    toy = datasets.load("toy")
    # dispatchable capacity: bus0 gas 100 + battery 20, bus1 coal 60 → median 90; bus2 (wind
    # only) and bus3 (load only) get the median factor of the others
    f = consensus.agent_factors(toy, "capacity")
    assert f == pytest.approx({"bus0": 0.75, "bus1": 1.5, "bus2": 1.125, "bus3": 1.125})
    assert set(consensus.agent_factors(toy, "none").values()) == {1.0}
    with pytest.raises(ValueError, match="agent_scale"):
        consensus.agent_factors(toy, "size")


def test_factors_come_from_the_network_with_storage():
    from energy_scheduling_benchmark import datasets
    from energy_scheduling_benchmark.run import without_storage

    toy = datasets.load("toy")
    # without the battery bus0 would get 0.8 (median 80 of 100/60); the tuned set uses 0.75
    no_storage = tune.max_sensitivity(without_storage(toy), "capacity", toy)
    assert no_storage == pytest.approx(0.75 / 0.066)
