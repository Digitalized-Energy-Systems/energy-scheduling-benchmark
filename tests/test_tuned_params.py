"""Tuned-parameter file: precedence, CLI integration, packaged defaults."""

from __future__ import annotations

import importlib
import inspect

import pytest

from energy_scheduling_benchmark import tuned_params

SCENARIOS = ["consensus", "admm", "diffusion", "exact_diffusion", "fdgdm", "deed_admm"]


def _capture(monkeypatch, scenario: str, argv: list[str]) -> dict:
    module = importlib.import_module(f"energy_scheduling_benchmark.scenarios.{scenario}")
    seen: dict = {}

    async def fake(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(module, "execute_test_case", fake)
    module.main(argv)
    return seen


class TestResolve:
    CONFIG = {
        "consensus": {"max_iter": 1, "alpha": 0.1, "topology": {"ring": {"max_iter": 2}}},
        "network": {
            "net": {
                "consensus": {"alpha": 0.3, "topology": {"ring": {"alpha": 0.4}}},
            }
        },
    }

    def test_scenario_defaults(self):
        values, sources = tuned_params.resolve(self.CONFIG, "consensus", "other", "complete")
        assert values == {"max_iter": 1, "alpha": 0.1}
        assert sources["max_iter"] == "consensus"

    def test_precedence(self):
        values, sources = tuned_params.resolve(self.CONFIG, "consensus", "dir/net.nc", "ring")
        assert values == {"max_iter": 2, "alpha": 0.4}
        assert sources["max_iter"] == "consensus.topology.ring"
        assert sources["alpha"] == 'network."net".consensus.topology.ring'

    def test_network_key(self):
        assert tuned_params.network_key("/a/b/base_s_2.nc") == "base_s_2"
        assert tuned_params.network_key("toy") == "toy"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_packaged_defaults_equal_library_defaults(monkeypatch, tmp_path, scenario):
    """On the complete topology the packaged file reproduces execute_test_case's
    own defaults, so moving the values into the file changes no result."""
    module = importlib.import_module(f"energy_scheduling_benchmark.scenarios.{scenario}")
    defaults = {
        name: p.default
        for name, p in inspect.signature(module.execute_test_case).parameters.items()
    }
    seen = _capture(monkeypatch, scenario, ["--out-dir", str(tmp_path)])
    config, _ = tuned_params.resolve(tuned_params.load(), scenario, "toy", "complete")
    assert config, f"no [{scenario}] section in tuned_parameters.toml"
    for key, value in config.items():
        assert seen[key] == value == defaults[key], key


def test_topology_section_applies(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, "consensus", ["--topology", "ring", "--out-dir", str(tmp_path)])
    assert seen["max_iter"] == 3000


def test_cli_flag_overrides_file(monkeypatch, tmp_path):
    seen = _capture(
        monkeypatch,
        "consensus",
        ["--topology", "ring", "--max-iter", "7", "--out-dir", str(tmp_path)],
    )
    assert seen["max_iter"] == 7


def test_custom_file(monkeypatch, tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('[admm]\nrho = 0.5\n[network."toy".admm]\nmax_iter = 9\n')
    seen = _capture(
        monkeypatch, "admm", ["--tuned-params", str(path), "--out-dir", str(tmp_path)]
    )
    assert seen["rho"] == 0.5 and seen["max_iter"] == 9


def test_unknown_key_rejected(monkeypatch, tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[admm]\nrhoo = 0.5\n")
    with pytest.raises(SystemExit):
        _capture(monkeypatch, "admm", ["--tuned-params", str(path), "--out-dir", str(tmp_path)])
