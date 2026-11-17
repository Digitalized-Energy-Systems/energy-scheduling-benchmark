import json
import shutil
import tempfile
import tomllib

import pandas as pd
import pytest

from energy_scheduling_benchmark import parameters, tune
from energy_scheduling_benchmark.algorithms import consensus


def test_write_section_replaces_only_its_table(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    shutil.copy(parameters.PARAMETERS, path)
    monkeypatch.setattr(parameters, "PARAMETERS", path)
    name = "consensus.toy.ring.sweep"
    tune.write_section(name, {"steps": "constant", "kappa": 0.01, "note": "a"})
    path.write_text(path.read_text() + "# a comment after the section\n")
    tune.write_section(name, {"steps": "constant", "kappa": 0.02, "note": "b"})
    table = tomllib.loads(path.read_text())["consensus"]
    assert table["toy"]["ring"]["sweep"] == {
        "steps": "constant",
        "kappa": 0.02,
        "note": "b",
    }
    assert table["toy"]["ring"]["manual"]["kappa"] == 0.03  # neighbours untouched
    assert table["alpha0"] == 0.055
    text = path.read_text()
    assert text.count(f"[{name}]") == 1
    assert text.index("# a comment after") > text.index('note = "b"')  # kept, in place


def _args(**kw):
    defaults = dict(
        algorithm="consensus", network="toy", topology="ring", start=None, hours=6,
        max_iter=1000, jobs=2, write=False, storage=True, out=tempfile.mkdtemp(),
        replace=False,
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


def test_sweep_keeps_existing_records_unless_replace(tmp_path):
    # D-154: earlier records are never deleted unasked
    store = tmp_path / "consensus" / "toy" / "ring_storage"
    store.mkdir(parents=True)
    (store / "sweep.csv").write_text("old")
    with pytest.raises(SystemExit, match="--replace"):
        tune.sweep(_args(out=tmp_path))
    assert (store / "sweep.csv").read_text() == "old"


def test_sweep_prunes_without_changing_the_winner(monkeypatch, capsys, tmp_path):
    # the good candidate first: the two tiny-step ones score worse on window 1 and are cut
    _grid(monkeypatch, (1.2, 0.05, 0.02), (0.73,), decaying=False)
    starts = ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 12:00"]
    best = tune.sweep(_args(start=starts, max_iter=300, jobs=1, out=tmp_path))
    assert best["kappa"] == pytest.approx(1.2 / (0.75 * 35.15), rel=1e-3)
    assert "2 pruned" in capsys.readouterr().out
    # S4b records (D-147/D-148/D-150): one row per candidate × day, pruned days skipped
    store = tmp_path / "consensus" / "toy" / "ring_storage"
    table = pd.read_csv(store / "sweep.csv")
    assert len(table) == 3 * 3
    assert list(table.groupby("candidate")["status"].apply(list)) == [
        ["ok"] * 3,
        ["ok", "skipped", "skipped"],
        ["ok", "skipped", "skipped"],
    ]
    for folder in table["folder"].dropna():
        assert (store / folder / "meta.json").exists()
        assert len(pd.read_csv(store / folder / "trace.csv")) > 1  # convergence curve
    record = json.loads((store / "sweep.json").read_text())
    assert record["chosen"] == best and len(record["central"]) == 3
    for c in record["central"].values():
        assert (store / c["folder"] / "meta.json").exists()


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


def test_rank_converged_by_iterations_before_budget_runs():
    # A3 (D-140): at the noise floor scores tie, so a converged run ranks by iterations
    fast = tune.rank({"stop_reason": "converged", "iterations": 200}, 1e-9, True)
    slow = tune.rank({"stop_reason": "converged", "iterations": 900}, 1e-12, True)
    budget = tune.rank({"stop_reason": "max_iter", "iterations": 3000}, 0.0, True)
    covered = tune.rank({"stop_reason": "renewables_cover", "iterations": 0}, 0.0, True)
    assert covered < fast < slow < budget
    assert max(fast, budget) == budget  # a candidate counts with its worst window


def test_rank_by_score_ignores_early_convergence():
    # D-143 (consensus): a fast but inaccurate stop loses to an accurate budget run
    early = tune.rank({"stop_reason": "converged", "iterations": 20}, 0.9, False)
    budget = tune.rank({"stop_reason": "max_iter", "iterations": 3000}, 0.01, False)
    assert budget < early
