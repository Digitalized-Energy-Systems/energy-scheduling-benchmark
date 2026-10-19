"""hpc/verify_inputs.py (Q19 acceptance test) and the expected inputs of a manifest."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "hpc" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vi = _load("verify_inputs")
mm = _load("make_manifest")

FLAGS = ["--topology", "ring", "--cost-model", "quadratic", "--simulate-days", "3",
         "--loss-rate", "0.05", "--seed", "2", "--slack"]  # fmt: skip
EXPECTED = {
    "networks": {"net.nc": "abc"},
    "git": {"energy_scheduling_benchmark": {"sha": "s1", "dirty": False}},
}


def _meta(**over):
    meta = {
        "dimensions": {
            "scenario": "deed_admm",
            "network": "/x/resources/networks/net.nc",
            "topology": "ring",
            "cost_model": "quadratic",
            "simulate_days": 3,
            "loss_rate": 0.05,
            "seed": 2,
            "slack": True,
        },
        "inputs": {"network_sha256": "abc"},
        "git": {"energy_scheduling_benchmark": {"sha": "s1", "dirty": False}},
    }
    meta.update(over)
    return meta


def test_expected_dims():
    assert vi.expected_dims(FLAGS) == {
        "slack": True, "topology": "ring", "cost_model": "quadratic",
        "simulate_days": 3, "loss_rate": 0.05, "seed": 2,
    }  # fmt: skip


def test_matching_run_passes():
    assert vi.check_run(_meta(), "deed-admm", "net.nc", FLAGS, EXPECTED) == []


def test_wrong_network_file_detected():
    problems = vi.check_run(
        _meta(inputs={"network_sha256": "zzz"}), "deed-admm", "net.nc", FLAGS, EXPECTED
    )
    assert any("sha256" in p for p in problems)


def test_wrong_code_detected():
    meta = _meta(git={"energy_scheduling_benchmark": {"sha": "other", "dirty": False}})
    problems = vi.check_run(meta, "deed-admm", "net.nc", FLAGS, EXPECTED)
    assert any("energy_scheduling_benchmark" in p for p in problems)


def test_wrong_scenario_detected():
    problems = vi.check_run(_meta(), "admm", "net.nc", FLAGS, EXPECTED)
    assert any("scenario" in p for p in problems)


def test_verify_reports_missing_runs(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "expected.json").write_text(json.dumps(EXPECTED))
    (tmp_path / "m" / "deed-admm.tsv").write_text(
        "\t".join(["r1", "net.nc", *FLAGS]) + "\n"
    )
    checked, missing, problems = vi.verify(tmp_path / "m", tmp_path / "results")
    assert (checked, missing, problems) == (1, ["net/r1"], [])
    (tmp_path / "results" / "net").mkdir(parents=True)
    (tmp_path / "results" / "net" / "r1-meta.json").write_text(json.dumps(_meta()))
    assert vi.verify(tmp_path / "m", tmp_path / "results") == (1, [], [])
    assert vi.main([str(tmp_path / "m"), "--results", str(tmp_path / "results")]) == 0


def test_expected_inputs_hash_network_files(tmp_path):
    net = tmp_path / "n.nc"
    net.write_bytes(b"network")
    expected = mm.expected_inputs([str(net), "toy"])
    import hashlib

    assert expected["networks"][str(net)] == hashlib.sha256(b"network").hexdigest()
    assert expected["networks"]["toy"] is None
    assert set(expected["git"]) == set(mm.REPOS)
