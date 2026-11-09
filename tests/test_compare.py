from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.compare import compare, load_runs, main
from energy_scheduling_benchmark.run import main as esb_run


def test_compare_pairs_runs_with_central(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(datasets.STANDARD_DAYS, "toy", {"winter": "2024-01-01"})
    common = ["--network", "toy", "--topology", "complete", "--out", str(tmp_path)]
    esb_run([*common, "--algorithm", "central", "--hours", "24"])
    esb_run([*common, "--algorithm", "dummy", "--hours", "24"])
    esb_run([*common, "--algorithm", "dummy", "--hours", "6"])  # no central for 6 h

    rows, missing = compare(load_runs(tmp_path))
    assert len(rows) == 1  # the 6 h dummy run is skipped
    row = rows.iloc[0]
    assert row.season == "winter" and row.hours == 24
    assert row.gap_pct == (row.total_cost / row.central_cost - 1) * 100
    assert row.gap_pct >= 0  # dummy cannot beat the optimum
    assert any("no central run" in m and "6h" in m for m in missing)
    assert any(
        m.startswith("toy central complete_loss0_seed0:") and " 6 h " in m
        for m in missing
    )

    capsys.readouterr()
    main(["--root", str(tmp_path)])
    out = capsys.readouterr()
    assert "| toy | dummy |" in out.out and "missing:" in out.err
    assert (tmp_path / "compare.csv").read_text().startswith("network,algorithm,")
