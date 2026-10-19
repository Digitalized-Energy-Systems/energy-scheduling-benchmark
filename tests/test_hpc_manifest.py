"""hpc/make_manifest.py: sweep size, pruning, unique run ids, CLI flags."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "hpc" / "make_manifest.py"
_spec = importlib.util.spec_from_file_location("make_manifest", _PATH)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def test_full_product_size_per_network():
    runs = list(mm.runs(["net.nc"]))
    # Distributed: 3 topologies x 2 cost models x 2 slack x 3 days x (2 lossless + 3 x 5 seeds).
    per_distributed = 3 * 2 * 2 * 3 * (2 + 3 * 5)
    # Central dispatch: complete topology and loss 0 only (x 2 for the determinism repeat).
    per_central = 2 * 2 * 3 * 2
    assert len(runs) == 6 * per_distributed + per_central


def test_central_dispatch_pruned_to_complete_and_lossless():
    central = [r for r in mm.runs(["net.nc"]) if r["scenario"] == "central-dispatch"]
    for r in central:
        args = r["args"]
        assert args[args.index("--topology") + 1] == "complete"
        assert args[args.index("--loss-rate") + 1] == "0"


def test_run_ids_unique_and_seeds_follow_q9():
    runs = list(mm.runs(["a.nc", "b.nc"]))
    keys = [(r["network"], r["run_id"]) for r in runs]
    assert len(keys) == len(set(keys))
    lossless = [r for r in runs if "_loss0_" in r["run_id"]]
    assert {r["run_id"].rsplit("_", 1)[1] for r in lossless} == {"s0", "s0-rep"}
    lossy = [r for r in runs if "_loss0.05_" in r["run_id"]]
    assert {r["run_id"].rsplit("_s", 1)[1] for r in lossy} == {"1", "2", "3", "4", "5"}


def test_flags_parse_with_the_scenario_cli():
    from energy_scheduling_benchmark.scenarios._common import build_scenario_argparser

    parser = build_scenario_argparser(None, default_name_base="x")
    for r in mm.runs(
        ["toy"], scenarios=["consensus"], days=[1], loss_rates=[0.0, 0.1], seeds=1
    ):
        args = parser.parse_args(["--network", r["network"], *r["args"]])
        assert args.topology in mm.TOPOLOGIES
        assert args.slack == ("slack-on" in r["run_id"])


def test_write_and_cli(tmp_path):
    assert (
        mm.main(["toy", "--days", "1", "--loss-rates", "0", "--out", str(tmp_path)])
        == 0
    )
    lines = (tmp_path / "consensus.tsv").read_text().splitlines()
    assert len(lines) == 3 * 2 * 2 * 2
    run_id, network, *flags = lines[0].split("\t")
    assert network == "toy" and "--topology" in flags


def test_unknown_network_rejected(tmp_path):
    with pytest.raises(SystemExit):
        mm.main(["nope.nc", "--out", str(tmp_path)])
