import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    THERMAL,
    Component,
    Network,
    save_network,
)

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.run import main, run, window


def test_window():
    net = window(datasets.load("toy"), "2024-01-01 06:00", 5)
    assert net.time_index[0] == pd.Timestamp("2024-01-01 06:00")
    assert all(len(c.p_max) == 5 for c in net.components.values())
    with pytest.raises(ValueError):
        window(datasets.load("toy"), None, 49)


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_end_to_end_lossless(tmp_path, topology):
    main(
        [
            "--algorithm",
            "dummy",
            "--topology",
            topology,
            "--hours",
            "6",
            "--out",
            str(tmp_path),
        ]
    )
    (out,) = tmp_path.glob("toy/dummy/*")
    csv = pd.read_csv(out / "schedule.csv", index_col="time")
    np.testing.assert_allclose(
        csv["residual"], 0.0, atol=1e-9
    )  # toy has enough capacity
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == "complete"
    assert meta["messages"]["lost"] == 0
    assert meta["messages"]["delivered"] == meta["messages"]["sent"] > 0
    assert meta["total_cost"] > 0


async def test_end_to_end_all_lost_times_out(tmp_path):
    args = argparse.Namespace(
        network="toy",
        algorithm="dummy",
        topology="grid",
        loss_rate=1.0,
        delay_s=0.01,
        seed=0,
        grid_exchange=True,
        start=None,
        hours=3,
        day=None,
        storage=True,
        out=str(tmp_path),
        trace=False,
        tuning=None,
        set=[],
    )
    out = await run(args)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == "timeout"
    # every offer request is lost; the leader's finish messages are control messages and arrive (S5)
    assert 0 < meta["messages"]["lost"] < meta["messages"]["sent"]
    csv = pd.read_csv(out / "schedule.csv", index_col="time")
    assert csv[["gas", "coal", "wind", "battery"]].isna().all().all()
    np.testing.assert_allclose(
        csv["residual"], 0.0
    )  # the grid exchange takes all demand


def test_no_storage(tmp_path):
    for flag in ([], ["--no-storage"]):
        main(["--algorithm", "central", "--hours", "24", "--out", str(tmp_path), *flag])
    on, off = (tmp_path / "toy/central/grid_loss0_seed0_from20240101T00_24h_storage",
               tmp_path / "toy/central/grid_loss0_seed0_from20240101T00_24h_nostorage")  # fmt: skip
    meta_on, meta_off = (json.loads((d / "meta.json").read_text()) for d in (on, off))
    assert "battery" not in pd.read_csv(off / "schedule.csv").columns
    assert meta_off["inputs"]["storage"] is False
    # storage can only help — up to the QP solver's tolerance (Clarabel ~1e-8 relative): with the
    # ageing cost the battery is almost idle on toy, and both optima agree to ~1e-9
    assert meta_off["total_cost"] >= meta_on["total_cost"] * (1 - 1e-8)


def test_prepare_once_run_from_file(tmp_path):
    path = datasets.prepare("toy", k=0.0, out=tmp_path / "toy_linear.nc")
    main(["--network", str(path), "--algorithm", "central", "--out", str(tmp_path)])
    (out,) = tmp_path.glob("toy_linear/central/*")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["network_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["stop_reason"] == "optimal"
    with pytest.raises(FileNotFoundError, match="esb-prepare --network nope"):
        main(["--network", "nope"])


def test_set_values_parse_python_booleans():
    from energy_scheduling_benchmark.run import _value

    # esb-tune writes candidates as f"{key}={value}": a Python bool must not become a string
    assert (
        _value("False") is False and _value("True") is True and _value("false") is False
    )
    assert _value("0.1") == 0.1 and _value("constant") == "constant"


def test_trace_lands_in_run_folder(tmp_path):
    main(["--algorithm", "dummy", "--hours", "2", "--trace", "--out", str(tmp_path)])
    (folder,) = tmp_path.glob("toy/dummy/*")
    names = {p.name for p in folder.iterdir()}
    assert {"mango_trace.jsonl", "mango_trace.html", "meta.json"} <= names


def _two_bus(tmp_path, wind):
    """Bus a: gas (c₁ 10, c₂ 0.1, 0…10 MW) + load 4 / 8 MW; bus b: wind (pre-dispatched)."""
    a = lambda *v: np.array(v, dtype=float)  # noqa: E731
    comps = {
        "gas": Component(
            THERMAL, "a", a(0, 0), a(10, 10), (10.0, 0.1), {"p_nom": 10.0}
        ),
        "load": Component(LOAD, "a", a(4, 8), a(4, 8)),
        "wind": Component(
            RENEWABLE, "b", a(0, 0), a(*wind), (0.0, 0.0), {"p_nom": 9.0}
        ),
    }
    path = tmp_path / "two_bus.nc"
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    save_network(Network(idx, comps, [("a", "b")]), path)
    return path


def _metas(tmp_path, path):
    common = ["--network", str(path), "--hours", "2", "--topology", "complete"]
    common += ["--out", str(tmp_path / "out")]
    main(["--algorithm", "central", *common])
    steps = ["--tuning", "paper", "--set", "steps=constant", "--set", "kappa=0.1"]
    main(["--algorithm", "consensus", *common, *steps, "--set", "gamma=4"])
    return [
        json.loads(
            next((tmp_path / "out").glob(f"two_bus/{alg}/*/meta.json")).read_text()
        )
        for alg in ("central", "consensus")
    ]


def test_surplus_hour_is_reported_not_curtailed(tmp_path):
    # hour 0: wind 6 > load 4 → 2 MWh surplus (islanded), gas off; hour 1: gas covers 8 − 1 = 7
    central, consensus = _metas(tmp_path, _two_bus(tmp_path, (6, 1)))
    assert central["stop_reason"] == "optimal"
    assert consensus["stop_reason"] == "converged"  # λ floor 0 in the surplus hour
    for meta in (central, consensus):
        assert meta["imbalance"]["surplus_mwh"] == pytest.approx(2.0, abs=1e-3)
        assert meta["renewable_available_mwh"] == 7.0
    assert consensus["total_cost"] == pytest.approx(central["total_cost"], rel=1e-3)


def test_renewables_cover_every_hour_needs_no_dispatch(tmp_path):
    central, consensus = _metas(tmp_path, _two_bus(tmp_path, (6, 9)))
    for meta in (central, consensus):
        assert meta["stop_reason"] == "renewables_cover"
        assert meta["iterations"] == 0 and meta["messages"]["sent"] == 0
        assert (
            meta["total_cost"] == 0.0
        )  # gas idle, wind free; a valid result, not null
        assert meta["imbalance"]["surplus_mwh"] == pytest.approx(3.0)


def test_control_messages_are_never_lost():
    from distributed_resource_optimization.algorithm.firstorder.consensus.innovations import (
        ConsensusMessage,
        ConsensusReport,
    )
    from mango.simulation.communication import MessagePackage

    from energy_scheduling_benchmark.world import CountingCommunication

    sim = CountingCommunication(loss_percent=1.0, default_delay_s=0.01)
    report = ConsensusReport("a", True, 1)
    packages = [
        MessagePackage("a", "b", 0.0, (ConsensusMessage(np.zeros(1), 1), {})),
        MessagePackage("a", "b", 0.0, (report, {})),
    ]
    reached = [
        r.reached for r in sim.calculate_communication(0.0, packages).package_results
    ]
    assert reached == [False, True] and (sim.sent, sim.lost) == (2, 1)


def test_algorithm_interface(tmp_path):
    from energy_scheduling_benchmark import tune
    from energy_scheduling_benchmark.algorithms import ALGORITHMS
    from energy_scheduling_benchmark.run import build_parser

    # audit 2026-11-12 Q-A4: one dict; every distributed algorithm module has setup(...)
    assert build_parser().parse_args([]).algorithm == "central"  # Q-A12a
    assert all(callable(m.setup) for m in ALGORITHMS.values() if m is not None)
    assert set(tune.ALGORITHMS) == {
        "consensus",
        "admm",
        "deed_admm",
        "diffusion",
        "exact_diffusion",
    }  # the tunable ones (sweep_grid)
    with pytest.raises(SystemExit, match="no parameters"):
        main(["--algorithm", "dummy", "--set", "kappa=1", "--out", str(tmp_path)])
