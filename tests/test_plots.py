import pytest

# the [plot] extra; ROSA's venv has no kaleido (plots run locally)
pytest.importorskip("kaleido")

from energy_scheduling_benchmark import plots, tune  # noqa: E402
from energy_scheduling_benchmark.algorithms import admm  # noqa: E402
from energy_scheduling_benchmark.run import main as esb_run  # noqa: E402

from .test_tune import _args, _grid  # noqa: E402


def _written(folder, *names):
    for name in names:
        for ext in ("html", "pdf"):
            assert (folder / f"{name}.{ext}").stat().st_size > 0, f"{name}.{ext}"


def test_sweep_plots(monkeypatch, tmp_path):
    # consensus: a stable, a pruned (tiny step) and a diverging (β > 2/λ_max) candidate + one
    # decaying; admm: 4 ρ — both variants of the grid figure, failed/pruned rows included
    _grid(monkeypatch, (1.2, 0.02), (0.73, 4.8))
    monkeypatch.setattr(admm, "SWEEP_CANDIDATES", 4)
    starts = ["2024-01-01 00:00", "2024-01-01 06:00"]
    for alg in ("consensus", "admm"):
        tune.sweep(_args(algorithm=alg, start=starts, max_iter=300, out=tmp_path))
    plots.main(["sweeps", str(tmp_path)])
    for alg in ("consensus", "admm"):
        _written(
            tmp_path / alg / "toy" / "ring_storage" / "plots", "convergence", "grid"
        )
    _written(tmp_path / "plots", "overview_consensus", "overview_admm")


def test_candidates_status_and_chosen(monkeypatch, tmp_path):
    _grid(monkeypatch, (1.2, 0.05, 0.02), (0.73,), decaying=False)
    starts = ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 12:00"]
    best = tune.sweep(_args(start=starts, max_iter=300, jobs=1, out=tmp_path))
    store = tmp_path / "consensus" / "toy" / "ring_storage"
    table = plots.pd.read_csv(store / "sweep.csv")
    record = plots.json.loads((store / "sweep.json").read_text())
    cands = plots.candidates(table, record)
    # the same picture as test_tune's pruning test: candidate 0 wins, 1 and 2 are cut
    assert list(cands.status) == ["ok", "pruned", "pruned"]
    assert plots.chosen(record) == 0 and cands.kappa[0] == best["kappa"]
    assert cands.label[0].startswith("constant, kappa=")


def test_run_plots(tmp_path):
    common = ["--network", "toy", "--topology", "complete", "--hours", "6"]
    common += ["--out", str(tmp_path)]
    for alg in ("central", "consensus", "admm", "dummy"):  # dummy: no trace.csv
        esb_run([*common, "--algorithm", alg])
    plots.main(["runs", str(tmp_path)])
    _written(tmp_path / "plots", "gap", "effort", "convergence_toy_20240101_storage")


def test_gap_pct_edges():
    g = plots.gap_pct([0.0, 2.0, 1.0], 1.0)
    assert list(g) == [-100.0, 100.0, 0.0]
    assert plots.gap_pct([0.0], 0.0)[0] == 0.0  # renewables cover: equal costs, gap 0
    assert plots.np.isnan(plots.gap_pct([1.0], 0.0)[0])
