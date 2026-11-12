import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    Component,
    Network,
    save_network,
)

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
        trace=False,
    )
    out = await run(args)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["stop_reason"] == "timeout"
    # every offer request is lost; the leader's finish messages are control messages and arrive (S5)
    assert 0 < meta["messages"]["lost"] < meta["messages"]["sent"]
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
    # last row is the final state (cost without slack = total_cost here, islanded)
    assert central["sim_time_s"] is None
    assert consensus["sim_time_s"] >= 0.01 * consensus["iterations"]
    trace = pd.read_csv(next(tmp_path.glob("toy/consensus/*/trace.csv")))
    assert list(trace.iteration) == list(range(len(trace)))
    assert trace.iteration.iloc[-1] >= consensus["iterations"] - 1
    assert trace.cost.iloc[-1] == pytest.approx(consensus["total_cost"], rel=1e-6)
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


def test_trace_lands_in_run_folder(tmp_path):
    main(["--hours", "2", "--trace", "--out", str(tmp_path)])
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
