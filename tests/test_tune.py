import shutil
import tomllib

import pytest

from energy_scheduling_benchmark import consensus, parameters, tune


def test_write_section_replaces_only_its_table(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    shutil.copy(parameters.PARAMETERS, path)
    monkeypatch.setattr(parameters, "PARAMETERS", path)
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


def _args(**kw):
    defaults = dict(
        algorithm="consensus", network="toy", topology="ring", start=None, hours=6,
        max_iter=1000, jobs=2, write=False, storage=True,
    )  # fmt: skip
    return tune.argparse.Namespace(**defaults | kw)


def _grid(monkeypatch, alpha, beta, decaying=True):
    """A small consensus grid; decaying candidates (if kept) use the paper exponent."""
    monkeypatch.setattr(consensus, "SWEEP_ALPHA", alpha)
    monkeypatch.setattr(consensus, "SWEEP_BETA_CONSTANT", beta)
    monkeypatch.setattr(consensus, "SWEEP_BETA_DECAYING", (0.3,) if decaying else ())
    monkeypatch.setattr(consensus, "SWEEP_ALPHA_EXP", (0.98,))


def test_sweep_picks_a_converged_candidate(monkeypatch, capsys):
    # toy max scaled bus dP/dλ = 0.75 · 35.15 (bus0, S3) → κ = 1.2/26.36 stable, 8/26.36 oscillates
    # ring of 4 buses: λ_max(L) = 4, β = 0.73/4 ≈ 4·κ as before; β = 4.8/4 > 2/λ_max diverges
    # → λ abort (T4)
    _grid(monkeypatch, (1.2, 8), (0.73, 4.8))
    best = tune.sweep(_args())
    assert best["steps"] == "constant" and best["agent_scale"] == "capacity"
    assert best["kappa"] == pytest.approx(1.2 / (0.75 * 35.15), rel=1e-3)
    assert best["gamma"] * best["kappa"] == pytest.approx(0.73 / 4, rel=1e-3)
    assert "best: constant" in best["note"]
    assert "diverged" in capsys.readouterr().out


def test_sweep_prunes_without_changing_the_winner(monkeypatch, capsys):
    # the good candidate first: the two tiny-step ones score worse on window 1 and are cut
    _grid(monkeypatch, (1.2, 0.05, 0.02), (0.73,), decaying=False)
    starts = ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 12:00"]
    best = tune.sweep(_args(start=starts, max_iter=300, jobs=1))
    assert best["kappa"] == pytest.approx(1.2 / (0.75 * 35.15), rel=1e-3)
    assert "2 pruned" in capsys.readouterr().out


def test_max_sensitivity_toy():
    from energy_scheduling_benchmark import datasets

    # bus0 gas 1/(2·0.033) + battery 1/(2·0.025); the wind bus (40) is pre-dispatched (S1)
    toy = datasets.load("toy")
    assert consensus.max_sensitivity(toy) == pytest.approx(1 / 0.066 + 20)
    # per-agent steps (S3): bus0 (s = 0.75) still the worst scaled bus; coal bus1 1.5 · 5.45
    assert consensus.max_sensitivity(toy, "capacity") == pytest.approx(
        0.75 * (1 / 0.066 + 20)
    )


def test_sweep_grid_toy():
    from energy_scheduling_benchmark import datasets

    toy = datasets.load("toy")
    # bus graph without the leader: complete K4 → 4, ring C4 → 4, grid = path P4 → 2 + √2
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


def test_parameter_routes(capsys):
    import pytest

    assert (
        parameters.parameters("consensus", "toy", "ring")["tuning"] == "manual"
    )  # default if present
    # Q-A10: no manual set → the sweep set (synthetic_10 complete has one)
    swept = parameters.parameters("consensus", "synthetic_10", "complete")
    assert swept["tuning"] == "sweep" and swept["agent_scale"] == "capacity"
    paper = parameters.parameters("consensus", "toy", "ring", "paper")
    assert (paper["steps"], paper["alpha0"]) == ("decaying", 0.055)
    assert "warning" not in capsys.readouterr().err  # explicit/tuned routes are quiet
    assert (
        parameters.parameters("consensus", "base_s_1_elec_2019", "grid")["tuning"]
        == "paper"
    )
    assert (
        "no tuned consensus set for base_s_1_elec_2019/grid" in capsys.readouterr().err
    )
    with pytest.raises(ValueError):
        parameters.parameters("consensus", "base_s_1_elec_2019", "grid", "sweep")
    assert (
        parameters.parameters("consensus", "toy", "ring", overrides={"kappa": 1.0})[
            "kappa"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="kapa"):
        parameters.parameters(
            "consensus", "toy", "ring", overrides={"kapa": 1.0}
        )  # typo


def test_sweep_writes_nothing_if_no_candidate_runs_through(monkeypatch):
    _grid(monkeypatch, (8,), (4.8,), decaying=False)
    before = parameters.PARAMETERS.read_text()
    assert tune.sweep(_args(max_iter=500, write=True)) is None  # β > 2/λ_max diverges
    assert parameters.PARAMETERS.read_text() == before


def test_sweep_writes_the_best_whatever_its_gap(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    shutil.copy(parameters.PARAMETERS, path)
    monkeypatch.setattr(parameters, "PARAMETERS", path)
    _grid(monkeypatch, (0.26,), (0.16,), decaying=False)  # κ ≈ 0.01
    best = tune.sweep(_args(max_iter=20, write=True))  # λ barely moves → gap far from 0
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
    no_storage = consensus.max_sensitivity(without_storage(toy), "capacity", toy)
    assert no_storage == pytest.approx(0.75 / 0.066)
