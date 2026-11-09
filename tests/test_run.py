import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import STORAGE, THERMAL

from energy_scheduling_benchmark import datasets
from energy_scheduling_benchmark.dummy import Offer, merit_order
from energy_scheduling_benchmark.run import main, run, window


def test_merit_order():
    flat = np.ones(2)
    offers = [
        Offer(THERMAL, 50.0, 1 * flat, 10 * flat),
        Offer(THERMAL, 20.0, 0 * flat, 4 * flat),
        None,
        Offer(STORAGE, 0.0, -5 * flat, 5 * flat),
    ]
    s = merit_order(offers, np.array([3.0, 8.0]))
    np.testing.assert_array_equal(
        s[0], [1.0, 4.0]
    )  # p_min, then what the cheap one leaves
    np.testing.assert_array_equal(s[1], [2.0, 4.0])
    assert s[2] is None
    np.testing.assert_array_equal(s[3], [0.0, 0.0])  # storage idle


def test_window():
    net = window(datasets.load("toy"), "2024-01-01 06:00", 5)
    assert net.time_index[0] == pd.Timestamp("2024-01-01 06:00")
    assert all(len(c.p_max) == 5 for c in net.components.values())
    with pytest.raises(ValueError):
        window(datasets.load("toy"), None, 49)


@pytest.mark.parametrize("topology", ["grid", "ring", "complete"])
def test_end_to_end_lossless(tmp_path, topology):
    main(["--topology", topology, "--hours", "6", "--out", str(tmp_path)])
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
        slack=True,
        start=None,
        hours=3,
        day=None,
        storage=True,
        out=str(tmp_path),
    )
    out = await run(args)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == "timeout"
    assert meta["messages"]["sent"] == meta["messages"]["lost"] > 0
    csv = pd.read_csv(out / "schedule.csv", index_col="time")
    assert csv[["gas", "coal", "wind", "battery"]].isna().all().all()
    np.testing.assert_allclose(csv["residual"], 0.0)  # the slack takes all demand


def test_no_storage(tmp_path):
    for flag in ([], ["--no-storage"]):
        main(["--algorithm", "central", "--hours", "24", "--out", str(tmp_path), *flag])
    on, off = (tmp_path / "toy/central/grid_loss0_seed0_from20240101T00_24h_storage",
               tmp_path / "toy/central/grid_loss0_seed0_from20240101T00_24h_nostorage")  # fmt: skip
    meta_on, meta_off = (json.loads((d / "meta.json").read_text()) for d in (on, off))
    assert "battery" not in pd.read_csv(off / "schedule.csv").columns
    assert meta_off["inputs"]["storage"] is False
    assert meta_off["total_cost"] >= meta_on["total_cost"]  # storage can only help


def test_prepare_once_run_from_file(tmp_path):
    path = datasets.prepare("toy", k=0.0, out=tmp_path / "toy_linear.nc")
    main(["--network", str(path), "--algorithm", "central", "--out", str(tmp_path)])
    (out,) = tmp_path.glob("toy_linear/central/*")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["network_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["stop_reason"] == "optimal"
    with pytest.raises(FileNotFoundError, match="esb-prepare --network nope"):
        main(["--network", "nope"])


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


def test_day_sets_window(tmp_path):
    main(["--network", "synthetic_10", "--algorithm", "central", "--day", "summer",
          "--out", str(tmp_path)])  # fmt: skip
    (out,) = tmp_path.glob("synthetic_10/central/*")
    assert out.name == "grid_loss0_seed0_from20190613T00_24h_storage"
    with pytest.raises(SystemExit):  # --start and --day exclude each other
        main(["--start", "2019-01-01", "--day", "winter"])
    with pytest.raises(SystemExit, match="no standard days for toy"):
        main(["--day", "winter", "--out", str(tmp_path)])


def test_consensus_settings_get_own_folders(tmp_path):
    common = ["--algorithm", "consensus", "--hours", "2", "--out", str(tmp_path)]
    main([*common, "--tuning", "paper", "--set", "max_iter=2"])
    main([*common, "--tuning", "paper", "--set", "max_iter=3"])
    names = sorted(p.name for p in tmp_path.glob("toy/consensus/*"))
    assert len(names) == 2 and all("_paper_set" in n for n in names)
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        main([*common, "--set", "kappa"])
