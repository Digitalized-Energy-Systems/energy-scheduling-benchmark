"""Unit tests for the shared scenario helpers in ``scenarios/_common.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_scheduling_benchmark import THERMAL, ComponentRef
from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios._common import (
    PowerLoadAggregator,
    _clip_scenario,
    build_group_map,
    build_p_max_vec,
    build_scenario_argparser,
    build_world,
    carrier_style,
    compute_overall_cost,
    compute_slack,
    loss_rate_from_args,
    loss_tolerance,
    require_lossless_transport,
    resolve_loss_rate,
    run_scenario_main,
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


class TestBuildPMaxVec:
    def test_thermal_availability_is_scaled_by_p_nom(self):
        # A thermal unit's timeseries is its per-unit p_max_pu: 0.5 of a
        # 400 MW unit is 200 MW, not 0.5 MW (scigrid-de's wind units).
        scenario = build_toy_network(periods=3)
        index = next(iter(scenario.timeseries.values())).index
        ref = ComponentRef(THERMAL, "gas")
        scenario.timeseries[ref] = pd.Series([1.0, 0.5, 0.0], index=index)
        p_max = build_p_max_vec(scenario, ref, {"p_nom": 400.0}, index, 3)
        np.testing.assert_allclose(p_max, [400.0, 200.0, 0.0])


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


class TestComputeSlack:
    def test_splits_residual_into_import_and_export(self):
        imp, exp = compute_slack(np.array([5.0, -3.0, 0.0]))
        assert imp.tolist() == [5.0, 0.0, 0.0]
        assert exp.tolist() == [0.0, 3.0, 0.0]


class TestAggregatorSlack:
    def _aggregator(self, schedules: dict[str, list[float]]) -> PowerLoadAggregator:
        return PowerLoadAggregator(
            behavior=None,
            number_loads=1,
            generator_aids=list(schedules),
            trigger=None,
            time_to_index={0.0: 0, 3600.0: 1, 7200.0: 2},
            schedule_by_aid={k: np.array(v) for k, v in schedules.items()},
            finished_message_type=object,
            build_start_message=lambda: None,
            demand_target=np.array([10.0, 10.0, 10.0]),
        )

    def test_residual_is_demand_minus_schedules(self):
        agg = self._aggregator({"a": [4.0, 8.0, 10.0], "b": [2.0, 4.0, 0.0]})
        np.testing.assert_allclose(agg.schedule_residual(), [4.0, -2.0, 0.0])

    def test_slack_by_bucket_keys_hours(self):
        agg = self._aggregator({"a": [4.0, 12.0, 10.0]})
        assert agg.slack_by_bucket() == {0: (6.0, 0.0), 1: (0.0, 2.0), 2: (0.0, 0.0)}

    def test_missing_schedule_counts_as_zero(self):
        agg = self._aggregator({"a": [4.0, 4.0, 4.0]})
        agg._generator_aids.append("never-finished")
        np.testing.assert_allclose(agg.schedule_residual(), [6.0, 6.0, 6.0])


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


class TestOutDir:
    def _run(self, argv):
        seen = {}

        async def fake_execute_test_case(**kwargs):
            seen.update(kwargs)

        run_scenario_main(
            fake_execute_test_case, doc=None, default_name_base="x", argv=argv
        )
        return seen

    def test_default_writes_to_cwd(self):
        seen = self._run(["--name-base", "run"])
        assert seen["name_base"] == "run"

    def test_out_dir_prefixes_name_base_and_is_created(self, tmp_path):
        out = tmp_path / "nested" / "results"
        seen = self._run(["--name-base", "run", "--out-dir", str(out)])
        assert seen["name_base"] == str(out / "run")
        assert out.is_dir()

    def test_loss_timeouts_default(self):
        seen = self._run(["--name-base", "run"])
        assert seen["load_timeout"] is None
        assert seen["max_missing_rounds"] == 20

    def test_loss_timeouts_forwarded(self):
        seen = self._run(
            ["--name-base", "run", "--load-timeout", "0.5", "--max-missing-rounds", "7"]
        )
        assert seen["load_timeout"] == 0.5
        assert seen["max_missing_rounds"] == 7

    def test_zero_max_missing_rounds_disables(self):
        seen = self._run(["--name-base", "run", "--max-missing-rounds", "0"])
        assert seen["max_missing_rounds"] is None

    def test_negative_max_missing_rounds_rejected(self):
        with pytest.raises(SystemExit):
            self._run(["--max-missing-rounds", "-1"])

    def test_negative_load_timeout_rejected(self):
        with pytest.raises(SystemExit):
            self._run(["--load-timeout", "-1"])


class TestLossTolerance:
    def test_defaults_scale_with_delay(self):
        lt = loss_tolerance(0.02)
        assert lt.load_timeout == pytest.approx(0.06)
        assert lt.round_timeout is None
        assert lt.max_missing_rounds == 20

    def test_no_result_polls_without_loss(self):
        # A lossless run that needs more than result_retries x result_poll_s
        # (e.g. consensus on a ring of 430 units) must not be zero-filled.
        assert loss_tolerance(0.02).result_poll_s is None
        assert loss_tolerance(0.02, loss_rate=0.05).result_poll_s == 300.0

    def test_overrides(self):
        lt = loss_tolerance(0.02, load_timeout=1.5, max_missing_rounds=None)
        assert lt.load_timeout == 1.5
        assert lt.max_missing_rounds is None


class TestResolveLossRate:
    def test_fraction_passes_through(self):
        assert resolve_loss_rate(0.05) == 0.05

    @pytest.mark.parametrize("rate", [-0.1, 1.5, 5.0])
    def test_out_of_range_rejected(self, rate):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            resolve_loss_rate(rate)


class TestLossCli:
    @staticmethod
    def _parse(*argv):
        parser = build_scenario_argparser(None, default_name_base="x")
        return parser.parse_args(list(argv))

    def test_default_is_lossless_with_seed_zero(self):
        args = self._parse()
        assert loss_rate_from_args(args) == 0.0
        assert args.seed == 0

    def test_loss_rate_is_a_fraction(self):
        assert loss_rate_from_args(self._parse("--loss-rate", "0.05")) == 0.05

    @pytest.mark.parametrize(
        "argv",
        [
            ("--loss-rate", "5"),
            ("--loss-rate", "-0.1"),
            # The old --loss-percent flag is gone (Q7: --loss-rate replaces it).
            ("--loss-percent", "5"),
        ],
    )
    def test_invalid_values_rejected(self, argv):
        with pytest.raises(SystemExit):
            self._parse(*argv)

    def test_seed_parsed(self):
        assert self._parse("--seed", "7").seed == 7


class TestBuildWorldCommunication:
    def test_loss_rate_and_seeded_rng_reach_mango(self):
        scenario = build_toy_network(periods=24)
        draws = []
        for _ in range(2):
            world, _ = build_world(scenario, delay_s=0.0, loss_rate=0.25, seed=3)
            com_sim = world.container.communication_sim
            assert com_sim.loss_rate == 0.25
            draws.append([com_sim._rng.random() for _ in range(5)])
        assert draws[0] == draws[1]

    def test_out_of_range_loss_rate_rejected(self):
        with pytest.raises(ValueError):
            build_world(build_toy_network(periods=24), delay_s=0.0, loss_rate=5.0)


class TestSlackDefaults:
    def test_import_price_is_the_sdac_cap(self):
        from energy_scheduling_benchmark.scenarios._common import (
            build_scenario_argparser,
        )

        args = build_scenario_argparser(None, default_name_base="x").parse_args([])
        assert args.slack_import_price == 4000.0
        assert args.slack_export_price == 0.0


class TestCostModel:
    def test_quadratic_coefficient_is_k_c1_over_p_nom(self):
        from energy_scheduling_benchmark.scenarios._common import (
            QUADRATIC_K,
            quadratic_cost_coefficient,
        )

        statics = {"marginal_cost": 50.0, "p_nom": 200.0}
        assert quadratic_cost_coefficient(statics, "linear") == 0.0
        c2 = quadratic_cost_coefficient(statics, "quadratic")
        assert c2 == pytest.approx(QUADRATIC_K * 50.0 / 200.0)
        # marginal cost at full output: c1 + 2·c2·p_nom = (1 + 2k)·c1
        assert 50.0 + 2 * c2 * 200.0 == pytest.approx((1 + 2 * QUADRATIC_K) * 50.0)
        assert (
            quadratic_cost_coefficient(
                {"marginal_cost": 0.0, "p_nom": 10.0}, "quadratic"
            )
            == 0.0
        )
        with pytest.raises(ValueError, match="cost_model"):
            quadratic_cost_coefficient(statics, "cubic")

    def test_only_thermal_units_get_a_quadratic_term(self):
        from energy_scheduling_benchmark import RENEWABLE, STORAGE, THERMAL
        from energy_scheduling_benchmark.scenarios._common import thermal_cost_quad

        refs = [
            ComponentRef(THERMAL, "g"),
            ComponentRef(RENEWABLE, "w"),
            ComponentRef(STORAGE, "s"),
        ]
        statics = {r: {"marginal_cost": 10.0, "p_nom": 100.0} for r in refs}
        assert set(thermal_cost_quad(refs, statics, "quadratic")) == {"g"}
        assert thermal_cost_quad(refs, statics, "linear") == {}

    def test_overall_cost_adds_c2_p_squared(self):
        t = np.array([0.0, 3600.0])
        Y = np.array([[10.0], [20.0]])
        total, _ = compute_overall_cost(
            {"g": 2.0}, t, Y, ["g"], cost_quad_by_aid={"g": 0.1}
        )
        assert total == pytest.approx(2.0 * 30.0 + 0.1 * (100.0 + 400.0))

    def test_cli_default_linear_and_choices(self):
        parser = build_scenario_argparser(None, default_name_base="x")
        assert parser.parse_args([]).cost_model == "linear"
        assert (
            parser.parse_args(["--cost-model", "quadratic"]).cost_model == "quadratic"
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["--cost-model", "cubic"])


class TestMetaSidecar:
    def test_cli_run_writes_meta_with_dimensions_metrics_and_versions(self, tmp_path):
        import json

        from energy_scheduling_benchmark.scenarios.central_dispatch import main

        main(
            [
                "--network",
                "toy",
                "--simulate-days",
                "1",
                "--name-base",
                "cd",
                "--out-dir",
                str(tmp_path),
            ]
        )
        meta = json.loads((tmp_path / "cd-meta.json").read_text())
        dims = meta["dimensions"]
        assert (
            dims["network"] == "toy"
            and dims["loss_rate"] == 0.0
            and dims["topology"] == "complete"
        )
        assert meta["total_cost"] > 0
        assert meta["messages_delivered"] > 0
        assert meta["wall_time_s"] > 0
        assert meta["algorithm_stats"]["solver_status"] == "optimal"
        assert set(meta["git"]) == {
            "energy_scheduling_benchmark",
            "distributed_resource_optimization",
            "mango_energy_environments",
            "mango",
        }
