"""Unit tests for the shared scenario helpers in ``scenarios/_common.py``."""

from __future__ import annotations

import numpy as np
import pytest

from energy_scheduling_benchmark import ComponentRef
from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios._common import (
    _clip_scenario,
    build_group_map,
    carrier_style,
    compute_overall_cost,
    require_lossless_transport,
)


class TestClipScenario:
    def test_trims_timeseries_to_window(self):
        scenario = build_toy_network(periods=72)
        clipped = _clip_scenario(scenario, simulate_days=1)
        assert all(len(ts) == 24 for ts in clipped.timeseries.values())

    def test_original_scenario_unchanged(self):
        scenario = build_toy_network(periods=72)
        _clip_scenario(scenario, simulate_days=1)
        assert all(len(ts) == 72 for ts in scenario.timeseries.values())

    def test_shorter_series_kept_as_is(self):
        scenario = build_toy_network(periods=24)
        clipped = _clip_scenario(scenario, simulate_days=3)
        assert all(len(ts) == 24 for ts in clipped.timeseries.values())


class TestComputeOverallCost:
    def test_sums_cost_weighted_power(self):
        t = [0.0, 3600.0]
        Y = np.array([[10.0, 5.0], [20.0, 0.0]])
        total, per_step = compute_overall_cost({"a": 2.0, "b": 4.0}, t, Y, ["a", "b"])
        assert per_step.tolist() == [40.0, 40.0]
        assert total == pytest.approx(80.0)

    def test_deduplicates_to_last_sample_per_hour(self):
        # Two samples in hour 0 → only the last (2.0) counts.
        t = [0.0, 100.0, 3600.0]
        Y = np.array([[1.0], [2.0], [3.0]])
        total, per_step = compute_overall_cost({"a": 1.0}, t, Y, ["a"])
        assert per_step.tolist() == [2.0, 3.0]
        assert total == pytest.approx(5.0)

    def test_negative_power_not_credited(self):
        """Storage charging (negative recorded power) must not reduce the total."""
        t = [0.0]
        Y = np.array([[10.0, -5.0]])
        total, _ = compute_overall_cost({"g": 3.0, "s": 3.0}, t, Y, ["g", "s"])
        assert total == pytest.approx(30.0)

    def test_unknown_aid_costs_zero(self):
        t = [0.0]
        Y = np.array([[10.0]])
        total, _ = compute_overall_cost({}, t, Y, ["mystery"])
        assert total == pytest.approx(0.0)


class TestBuildGroupMap:
    def test_carrier_from_cache_load_skipped(self):
        r_gas = ComponentRef("thermal", "virtual/1: CCGT")
        r_wind = ComponentRef("renewable", "w1")
        r_load = ComponentRef("load", "l1")
        group_map = build_group_map(
            [r_gas, r_wind, r_load],
            statics_by_ref={
                r_gas: {"carrier": "CCGT"},
                r_wind: {"carrier": "offwind-ac"},
                r_load: {"carrier": "AC"},
            },
        )
        assert group_map == {"virtual/1: CCGT": "CCGT", "w1": "offwind-ac"}

    def test_missing_carrier_falls_back_to_element_type(self):
        r_store = ComponentRef("storage", "batt0")
        group_map = build_group_map(
            [r_store], statics_by_ref={r_store: {"carrier": ""}}
        )
        assert group_map == {"batt0": "storage"}

    def test_nan_carrier_falls_back(self):
        r = ComponentRef("thermal", "g0")
        group_map = build_group_map([r], statics_by_ref={r: {"carrier": np.nan}})
        assert group_map == {"g0": "thermal"}

    def test_behavior_fallback_when_no_cache(self):
        class FakeBehavior:
            def get_statics(self, ref):
                return {"carrier": "solar"}

        r = ComponentRef("renewable", "s0")
        assert build_group_map([r], behavior=FakeBehavior()) == {"s0": "solar"}


class TestCarrierStyle:
    def test_populated_carriers_table(self):
        import pypsa

        net = pypsa.Network()
        net.add("Carrier", "CCGT", color="#a85522", nice_name="Combined-Cycle Gas")
        net.add("Carrier", "solar", color="#f9d002", nice_name="Solar")
        colors, names = carrier_style(net)
        assert colors["CCGT"] == "#a85522"
        assert names["solar"] == "Solar"

    def test_empty_carriers_table(self):
        assert carrier_style(build_toy_network(periods=24).net) == ({}, {})


class TestRequireLosslessTransport:
    def test_zero_loss_passes(self):
        require_lossless_transport(0.0, "TestAlgo")

    def test_nonzero_loss_raises_with_algorithm_name(self):
        with pytest.raises(ValueError, match="TestAlgo.*lossless"):
            require_lossless_transport(0.5, "TestAlgo")
