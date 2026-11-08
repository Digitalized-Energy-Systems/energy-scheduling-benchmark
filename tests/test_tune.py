import shutil
import tomllib

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
        alpha0="200", beta0="0.2", kappa="1.2,8", gamma="4", max_iter=2000, write=False,
    )  # fmt: skip
    best = await tune.sweep(args)
    # toy max bus dP/dλ = 40 → κ = 1.2/40 = 0.03 (stable), 8/40 = 0.2 oscillates
    assert best["steps"] == "constant" and best["kappa"] == 0.03
    assert "best: constant" in best["note"]


def test_max_sensitivity_toy():
    from energy_scheduling_benchmark import datasets

    assert tune.max_sensitivity(datasets.load("toy")) == 40.0  # wind: 1/(2·0.0125)


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
