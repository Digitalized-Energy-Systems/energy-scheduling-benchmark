"""Tests for the PyPSA network loaders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    extract_timeseries,
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


class TestExtractTimeseries:
    def test_no_timeseries_returns_empty(self):
        import pypsa

        net = pypsa.Network()
        net.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
        net.add("Bus", "b0")
        net.add("Generator", "g0", bus="b0", carrier="gas", p_nom=10.0)
        assert extract_timeseries(net) == {}

    def test_p_max_pu_extracted_as_renewable(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(RENEWABLE, "w0") in ts
        assert list(ts[ComponentRef(RENEWABLE, "w0")].values) == [0.5, 0.6, 0.7]

    def test_loads_t_extracted(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Load", "l0", bus="b0", p_set=3.0)
        net.loads_t.p_set = pd.DataFrame({"l0": [3.0, 4.0]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(LOAD, "l0") in ts

    def test_storage_units_t_extracted(self):
        import pypsa

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=2, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("StorageUnit", "s0", bus="b0", p_nom=5.0, max_hours=2.0)
        net.storage_units_t.p_set = pd.DataFrame({"s0": [0.0, 1.0]}, index=snaps)
        ts = extract_timeseries(net)
        assert ComponentRef(STORAGE, "s0") in ts


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


class TestPyPSABehaviorFromScenario:
    def test_from_scenario_carries_start_and_timeseries(self):
        from energy_scheduling_benchmark import PyPSABehavior

        scenario = build_toy_network(periods=24)
        behavior = PyPSABehavior.from_scenario(scenario)

        assert behavior.net is scenario.net
        assert behavior.start_datetime == scenario.start

    def test_from_network_auto_extracts_timeseries(self):
        import pypsa

        from energy_scheduling_benchmark import PyPSABehavior

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)

        behavior = PyPSABehavior.from_network(net)
        assert ComponentRef(RENEWABLE, "w0") in behavior._timeseries

    def test_from_network_auto_timeseries_disabled(self):
        import pypsa

        from energy_scheduling_benchmark import PyPSABehavior

        net = pypsa.Network()
        snaps = pd.date_range("2024-01-01", periods=3, freq="h")
        net.set_snapshots(snaps)
        net.add("Bus", "b0")
        net.add("Generator", "w0", bus="b0", carrier="wind", p_nom=10.0)
        net.generators_t.p_max_pu = pd.DataFrame({"w0": [0.5, 0.6, 0.7]}, index=snaps)

        behavior = PyPSABehavior.from_network(
            net, auto_timeseries=False, start_datetime=datetime(2024, 1, 1)
        )
        assert behavior._timeseries == {}
