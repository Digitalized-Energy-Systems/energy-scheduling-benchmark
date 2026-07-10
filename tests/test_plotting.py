"""Unit tests for the plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from energy_scheduling_benchmark.plotting import cost_over_time, stacked_area


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

    def test_rejects_non_2d_matrix(self):
        with pytest.raises(ValueError, match="2-D"):
            stacked_area([0.0], np.ones(3), ["a"], [1.0])

    def test_rejects_label_count_mismatch(self):
        with pytest.raises(ValueError, match="labels"):
            stacked_area([0.0, 1.0], np.ones((2, 2)), ["a"], [1.0, 1.0])

    def test_rejects_time_length_mismatch(self):
        with pytest.raises(ValueError, match="time"):
            stacked_area([0.0], np.ones((2, 2)), ["a", "b"], [1.0])


class TestCostOverTime:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "cost.pdf"
        cost_over_time([0.0, 1.0], [10.0, 20.0], write_to=str(out))
        assert out.exists()
