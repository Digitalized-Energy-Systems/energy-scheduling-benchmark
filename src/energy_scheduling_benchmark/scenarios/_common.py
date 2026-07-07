"""Shared boilerplate for the distributed-optimization scenarios.

Consensus, ADMM, diffusion, FDGDM, and DEED-ADMM all follow the same shape:
load agents push per-timestep demand to a leader, the leader kicks off a
distributed optimization run exactly once, and once every generator reports
back it applies the resulting per-timestep schedule as the simulation clock
advances. This module holds that shared shape so each scenario only needs to
supply the algorithm-specific pieces (actor construction, start-message
payload, and the attribute path to read the finished schedule from).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import Any

import numpy as np
import pandas as pd
from mango import Role

from energy_scheduling_benchmark import PowerUpdateInfo
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.plotting import _to_scalar as _scalar

logger = logging.getLogger(__name__)

__all__ = [
    "build_toy_network",
    "load_scenario",
    "_clip_scenario",
    "_lookup_ts",
    "_scalar",
    "_keep_hourly",
    "_write_agent_recordings_csv",
    "compute_overall_cost",
    "build_scenario_argparser",
    "PowerLoadInfo",
    "OptimizationFinishedInfo",
    "PowerLoadMonitoring",
    "PowerLoadAggregator",
    "make_finish_callback",
]


def _clip_scenario(scenario: ScenarioData, simulate_days: int) -> ScenarioData:
    """Return a copy of *scenario* with timeseries trimmed to *simulate_days* × 24 hours.

    Limits both the number of tasks scheduled by PyPSABehavior.initialize() and
    the optimization horizon used by all scenario algorithms, keeping runtimes
    proportional to the simulated window regardless of how long the underlying
    network's snapshot range is.
    """
    n = simulate_days * 24
    clipped = {k: v.iloc[:n] for k, v in scenario.timeseries.items()}
    return _dataclass_replace(scenario, timeseries=clipped)


def _lookup_ts(scenario: ScenarioData, ref: Any) -> Any | None:
    """Return the timeseries for *ref*, falling back to attribute-based lookup.

    ComponentRef equality may fail across subclasses, so we also try matching
    on ``element_type`` and ``component_id`` directly.
    """
    ts = scenario.timeseries.get(ref)
    if ts is not None:
        return ts
    for k, v in scenario.timeseries.items():
        if getattr(k, "element_type", None) == getattr(
            ref, "element_type", None
        ) and getattr(k, "component_id", None) == getattr(ref, "component_id", None):
            return v
    return None


def _keep_hourly(
    t_list: list | np.ndarray,
    Y_arr: np.ndarray,
    step_s: float = 3600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the last recorded state per hourly snapshot, dropping sub-second noise.

    A "P"/"target" recording ticks on every discrete-event step (report/solve/
    dispatch round-trips, retries, per-iteration peer rounds), not just once
    per snapshot. Applying this before plotting keeps stacked-area/cost-over-
    time charts showing one point per hour instead of intra-hour convergence
    noise — the same bucket-and-keep-last logic :func:`compute_overall_cost`
    already applies internally to its own return value.
    """
    if len(t_list) == 0:
        return np.asarray(t_list), Y_arr
    t = np.asarray(t_list, dtype=float)
    buckets = (t // step_s).astype(int)
    # last index of each unique bucket (reverse → find-first → un-reverse)
    _, first_in_rev = np.unique(buckets[::-1], return_index=True)
    last_idx = np.sort(len(t) - 1 - first_in_rev)
    return t[last_idx], Y_arr[last_idx]


def _write_agent_recordings_csv(
    world: Any,
    path: str,
    snapshot_step_s: float = 0.0,
    extra: pd.Series | None = None,
) -> None:
    """Write all per-agent recordings to a single wide CSV.

    When *snapshot_step_s* > 0, only the last recorded state per time bucket is
    kept (e.g. one row per hour), dropping sub-second convergence-phase noise
    from algorithms that iterate on the simulation clock (see FDGDM).

    *extra*, if given, is an additional column (e.g. the per-timestep total
    cost from :func:`compute_overall_cost`) merged in on the time index.
    """
    frames: list[pd.DataFrame] = []
    for key, rec in world.data_agent_collections.items():
        if not rec.timeseries:
            continue
        length = min([len(rec.time)] + [len(v) for v in rec.timeseries.values()])
        data = {
            f"{key}:{aid}": [_scalar(v) for v in values[:length]]
            for aid, values in rec.timeseries.items()
        }
        data["time"] = rec.time[:length]
        frames.append(pd.DataFrame(data).set_index("time"))

    if not frames:
        pd.DataFrame().to_csv(path)
        return

    df = pd.concat(frames, axis=1)

    if snapshot_step_s > 0 and not df.empty:
        bucket = (df.index.to_series() // snapshot_step_s).astype(int)
        df = df.groupby(bucket.values).last()
        df.index.name = "time"

    if extra is not None:
        if snapshot_step_s > 0:
            # *extra*'s index is raw seconds too (e.g. from a "P" recording
            # already de-duplicated to one sample per bucket) — rebucket it
            # the same way so it lines up with df's now-bucketed integer index
            # instead of joining on raw timestamps that no longer appear there.
            bucket = (extra.index.to_series() // snapshot_step_s).astype(int)
            extra = pd.Series(extra.values, index=bucket.values, name=extra.name)
        df = df.join(extra, how="outer")

    df.to_csv(path)


def compute_overall_cost(
    cost_by_aid: dict[str, float],
    t_P: Sequence[float],
    Y_P: np.ndarray,
    labels_P: Sequence[str],
    step_s: float = 3600.0,
) -> tuple[float, pd.Series]:
    """Total linear cost of a "P" recording: sum_t sum_i cost_by_aid[aid_i] * P_i(t).

    A "P" recording ticks on every discrete-event step, not just once per
    snapshot — e.g. central_dispatch's report/solve/dispatch round-trip
    records the same settled dispatch 3 times per hour. Summing those raw
    ticks would multiply-count the same dispatch decision, so *t_P*/*Y_P* are
    first collapsed to one (the last) sample per *step_s* bucket, same as
    FDGDM's own de-duplication for its plots.

    Returns ``(total, per_step)`` where *per_step* is a ``"cost:total"``
    series indexed by the de-duplicated *t_P*, suitable for merging into the
    recordings CSV via :func:`_write_agent_recordings_csv`'s ``extra``
    parameter.
    """
    t_arr = np.asarray(t_P, dtype=float)
    if step_s > 0 and t_arr.size:
        buckets = (t_arr // step_s).astype(int)
        _, first_in_rev = np.unique(buckets[::-1], return_index=True)
        last_idx = np.sort(len(t_arr) - 1 - first_in_rev)
        t_arr = t_arr[last_idx]
        Y_P = Y_P[last_idx]

    cost_vec = np.array([cost_by_aid.get(aid, 0.0) for aid in labels_P])
    per_step = pd.Series((Y_P * cost_vec).sum(axis=1), index=t_arr, name="cost:total")
    return float(per_step.sum()), per_step


def build_scenario_argparser(
    description: str | None,
    *,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser shared by all scenario entry points.

    *extra_args*, if given, is called with the parser to add algorithm-specific
    flags (e.g. DEED-ADMM's ``--gamma``/``--max-iter``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--network",
        type=str,
        default="toy",
        help=(
            "Network source. Either 'toy' (the built-in 5-bus fixture), "
            "a PyPSA example name "
            f"({', '.join(available_examples())}), or a path to a "
            ".nc/.h5/.xlsx file or CSV folder."
        ),
    )
    parser.add_argument("--delay-s", type=float, default=0.02)
    parser.add_argument("--loss-percent", type=float, default=0.0)
    parser.add_argument("--name-base", type=str, default=default_name_base)
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    if extra_args is not None:
        extra_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class PowerLoadInfo:
    """Current load power forwarded from a load agent to the aggregator."""

    power_load: float
    time: float


@dataclass
class OptimizationFinishedInfo:
    """Sent by a generator agent once its distributed-optimization run is done."""

    aid: str


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class PowerLoadMonitoring(Role):
    """Observes ``max_active_power`` and forwards it to *target* each timestep."""

    def __init__(self, behavior: Any, target: Any) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        power = self._behavior.observe(self.context.aid, "max_active_power")
        asyncio.create_task(
            self.context.send_message(
                PowerLoadInfo(
                    power_load=float(power), time=self.context.current_timestamp
                ),
                self._target,
            )
        )


class PowerLoadAggregator(Role):
    """Leader role: fires a distributed optimization run once, then dispatches
    the resulting schedule.

    Load agents stream ``PowerLoadInfo`` messages each hour. Once all loads
    report for a given timestep, the aggregator applies the pre-computed
    set-points for that hour. Timesteps that arrive before the optimization
    run is done are held in ``_pending`` and flushed immediately afterwards.
    """

    def __init__(
        self,
        *,
        behavior: Any,
        number_loads: int,
        generator_aids: list[str],
        trigger: Any,
        time_to_index: dict[float, int],
        schedule_by_aid: dict[str, np.ndarray],
        finished_message_type: type,
        build_start_message: Callable[[], Any],
        n_finished_required: int | None = None,
    ) -> None:
        super().__init__()
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = trigger
        self._time_to_index = time_to_index
        self._schedule_by_aid = schedule_by_aid
        self._finished_message_type = finished_message_type
        self._build_start_message = build_start_message
        self._n_finished_required = (
            n_finished_required
            if n_finished_required is not None
            else len(generator_aids)
        )

        self._ready: bool = False
        self._finished_aids: set[str] = set()
        self._pending: dict[int, float] = {}

        self.demand_map: dict[Any, list[float]] = {}
        self.target: float = 0.0  # recorded by the world for plotting

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._handle_load_info, lambda c, m: isinstance(c, PowerLoadInfo)
        )
        self.context.subscribe_message(
            self,
            self._handle_finished,
            lambda c, m: isinstance(c, self._finished_message_type),
        )

    def on_ready(self) -> None:
        if self._trigger is None or self._n_finished_required == 0:
            # All schedules are pre-filled; nothing to run.
            self._ready = True
            return
        asyncio.create_task(
            self.context.send_message(self._build_start_message(), self._trigger)
        )

    def _apply_schedule_index(self, idx: int, target_total: float) -> None:
        self.target = float(target_total)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            p_schedule = np.asarray(schedule).ravel()
            if idx < p_schedule.size:
                self._behavior.act(aid, "regulate", float(p_schedule[idx]))

    def _handle_finished(self, message: Any, meta: dict) -> None:
        self._finished_aids.add(message.aid)
        if len(self._finished_aids) == self._n_finished_required:
            self._ready = True
            for idx in sorted(self._pending):
                self._apply_schedule_index(idx, self._pending[idx])
            self._pending.clear()

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        bucket = self.demand_map.setdefault(message.time, [])
        bucket.append(float(message.power_load))
        if len(bucket) != self.number_loads:
            return  # wait until all loads report for this timestep

        total = float(sum(bucket))
        idx = self._time_to_index.get(round(float(message.time), 6))
        if idx is None:
            return

        if self._ready:
            self._apply_schedule_index(idx, total)
        else:
            # Optimization run is still in progress; remember what we need to
            # apply once the schedule is available. Do not update self.target
            # yet — that keeps P and target aligned in recordings.
            self._pending[idx] = total


def make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    schedule_by_aid: dict[str, np.ndarray],
    finished_message_type: type,
    algorithm_label: str,
    schedule_attr: str = "actor.P",
) -> Any:
    """Return the ``(algorithm, carrier) -> None`` hook shared by the
    carrier-based algorithms (consensus, diffusion, FDGDM, DEED-ADMM).

    *schedule_attr* is a dotted attribute path read off *algorithm* to obtain
    the finished per-timestep schedule (e.g. ``"actor.P"`` or ``"P"``).
    """

    def handle_finished(algorithm: Any, carrier: Any) -> None:
        role = carrier._parent
        aid = role.context.aid

        obj = algorithm
        for part in schedule_attr.split("."):
            obj = getattr(obj, part)
        schedule_by_aid[aid] = np.asarray(obj, dtype=float).copy()

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(finished_message_type(aid=aid), leader_addr)
            )
        logger.info(
            "%s finished for %s (schedule len=%s)",
            algorithm_label,
            aid,
            schedule_by_aid[aid].size,
        )

    return handle_finished
