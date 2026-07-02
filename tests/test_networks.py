"""Tests for the PyPSA network loaders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from mango_energy_environments import LOAD, RENEWABLE, THERMAL, ComponentRef

from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_network,
    load_scenario,
)


class TestAvailableExamples:
    def test_lists_known_examples(self):
        names = available_examples()
        assert "ac-dc-meshed" in names
        assert "storage-hvdc" in names
        assert "scigrid-de" in names


class TestBuildToyNetwork:
    def test_returns_scenario_data(self):
        result = build_toy_network()
        assert isinstance(result, ScenarioData)
        assert result.start == datetime(2024, 1, 1)
        assert result.label == "toy-5bus"

    def test_has_expected_components(self):
        result = build_toy_network()
        net = result.net
        assert set(net.generators.index) == {"thermal0", "thermal1", "wind0"}
        assert set(net.loads.index) == {"load0", "load1"}
        assert set(net.storage_units.index) == {"batt0"}

    def test_timeseries_keys(self):
        result = build_toy_network()
        assert ComponentRef(RENEWABLE, "wind0") in result.timeseries
        assert ComponentRef(LOAD, "load0") in result.timeseries
        assert ComponentRef(LOAD, "load1") in result.timeseries


class TestLoadNetwork:
    def test_passes_through_pypsa_network(self):
        import pypsa

        original = pypsa.Network()
        assert load_network(original) is original

    def test_callable_source_invoked(self):
        import pypsa

        sentinel = pypsa.Network()
        assert load_network(lambda: sentinel) is sentinel

    def test_example_name_dispatches_to_loader(self):
        """``load_network`` recognises example short names as non-paths."""
        import pypsa

        sentinel = pypsa.Network()

        # Monkey-patch pypsa.examples temporarily
        class _FakeExamples:
            @staticmethod
            def ac_dc_meshed(**_kw):
                return sentinel

        original = getattr(pypsa, "examples", None)
        pypsa.examples = _FakeExamples()
        try:
            net = load_network("ac-dc-meshed")
            assert net is sentinel
        finally:
            if original is not None:
                pypsa.examples = original
            else:
                del pypsa.examples

    def test_unsupported_source_raises(self):
        with pytest.raises(ValueError):
            load_network(Path("/does/not/exist.xyz"))


class TestLoadScenario:
    def test_from_pypsa_network(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-03-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "g0", bus="b0", carrier="gas", p_nom=10.0)

        scenario = load_scenario(net, label="custom")
        assert scenario.net is net
        assert scenario.label == "custom"
        assert scenario.start == datetime(2024, 3, 1)

    def test_explicit_timeseries_override(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.5]}, index=snaps)

        overrides = {ComponentRef(THERMAL, "foo"): pd.Series([1.0], index=snaps[:1])}
        scenario = load_scenario(net, timeseries=overrides)
        assert scenario.timeseries is overrides
