"""Unit tests for the plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from energy_scheduling_benchmark.plotting import (
    cost_over_time,
    generation_vs_demand,
    order_carriers,
    per_unit_small_multiples,
    resolve_carrier_colors,
    stacked_area,
)


class TestOrderCarriers:
    def test_merit_order_unknowns_then_other_last(self):
        assert order_carriers({"solar", "CCGT", "unobtanium", "other", "onwind"}) == [
            "CCGT",
            "onwind",
            "solar",
            "unobtanium",
            "other",
        ]


class TestResolveCarrierColors:
    def test_network_style_wins_then_fallback_then_cycle(self):
        colors = resolve_carrier_colors(
            ["CCGT", "solar", "unobtanium"], {"CCGT": "#123456"}
        )
        assert colors["CCGT"] == "#123456"  # network-supplied
        assert colors["solar"] == "#f9d002"  # FALLBACK_CARRIER_COLORS
        assert colors["unobtanium"].startswith("#")  # deterministic cycle slot


class TestStackedArea:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "stacked.pdf"
        stacked_area(
            [0.0, 1.0, 2.0],
            np.ones((3, 2)),
            ["a", "b"],
            [2.0, 2.0, 2.0],
            write_to=str(out),
        )
        assert out.exists()

    def test_grouped_aggregates_and_writes(self, tmp_path):
        out = tmp_path / "stacked_grouped.pdf"
        stacked_area(
            [0.0, 1.0, 2.0],
            np.ones((3, 3)),
            ["g0", "g1", "w0"],
            [3.0, 3.0, 3.0],
            groups={"g0": "CCGT", "g1": "CCGT"},  # "w0" unmapped -> "other"
            colors={"CCGT": "#a85522", "other": "#cccccc"},
            write_to=str(out),
        )
        assert out.exists()

    def test_rejects_non_2d_matrix(self):
        with pytest.raises(ValueError, match="2-D"):
            stacked_area([0.0], np.ones(3), ["a"], [1.0])

    def test_validates_shape_before_grouping(self):
        with pytest.raises(ValueError, match="2-D"):
            stacked_area([0.0], np.ones(3), ["a"], [1.0], groups={"a": "CCGT"})

    def test_rejects_label_count_mismatch(self):
        with pytest.raises(ValueError, match="labels"):
            stacked_area([0.0, 1.0], np.ones((2, 2)), ["a"], [1.0, 1.0])

    def test_rejects_time_length_mismatch(self):
        with pytest.raises(ValueError, match="time"):
            stacked_area([0.0], np.ones((2, 2)), ["a", "b"], [1.0])


class TestPerUnitSmallMultiples:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "observation.pdf"
        per_unit_small_multiples(
            [0.0, 1.0, 2.0],
            np.ones((3, 4)),
            ["a", "b", "c", "d"],
            {"a": "CCGT", "b": "CCGT", "c": "solar"},  # "d" -> "other"
            write_to=str(out),
        )
        assert out.exists()

    def test_handles_empty_recording(self, tmp_path):
        out = tmp_path / "observation_empty.pdf"
        per_unit_small_multiples([], np.zeros((0, 0)), [], {}, write_to=str(out))
        assert out.exists()


class TestGenerationVsDemand:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "balance.pdf"
        generation_vs_demand(
            [0.0, 1.0, 2.0],
            np.array([[1.0, -0.5], [2.0, 0.0], [1.5, 0.0]]),
            [1.0, 2.0, 1.5],
            write_to=str(out),
        )
        assert out.exists()

    def test_handles_empty_recording(self, tmp_path):
        out = tmp_path / "balance_empty.pdf"
        generation_vs_demand([], np.zeros((0, 0)), [], write_to=str(out))
        assert out.exists()


class TestCostOverTime:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "cost.pdf"
        cost_over_time([0.0, 1.0], [10.0, 20.0], write_to=str(out))
        assert out.exists()
