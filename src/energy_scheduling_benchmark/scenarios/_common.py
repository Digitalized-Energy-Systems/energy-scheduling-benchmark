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
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import Any

import numpy as np
import pandas as pd
from mango import Role

from energy_scheduling_benchmark import (
    LOAD,
    ComponentRef,
    PowerUpdateInfo,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.plotting import _to_scalar as _scalar
from energy_scheduling_benchmark.plotting import (
    agent_recording_as_plottable,
    cost_over_time,
    generation_vs_demand,
    order_carriers,
    per_unit_small_multiples,
    resolve_carrier_colors,
    stacked_area,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_toy_network",
    "load_scenario",
    "resolve_scenario",
    "build_behavior",
    "_clip_scenario",
    "_lookup_ts",
    "_scalar",
    "_keep_hourly",
    "_write_agent_recordings_csv",
    "compute_overall_cost",
    "build_group_map",
    "carrier_style",
    "write_scenario_outputs",
    "build_scenario_argparser",
    "require_lossless_transport",
    "filter_and_cache_statics",
    "PowerLoadInfo",
    "OptimizationFinishedInfo",
    "PowerLoadMonitoring",
    "PowerLoadAggregator",
    "make_finish_callback",
]


def resolve_scenario(network: str, *, simulate_days: int) -> ScenarioData:
    """Resolve the CLI ``--network`` argument into a :class:`ScenarioData` bundle.

    ``"toy"`` builds the built-in 5-bus fixture sized to the simulated window;
    anything else (PyPSA example name, ``.nc``/``.h5``/``.xlsx``/CSV path, or
    PYPOWER case) goes through :func:`load_scenario`.

    :param network: Value of the ``--network`` CLI argument.
    :param simulate_days: Number of simulated days, used to size the toy network.
    """
    if network == "toy":
        return build_toy_network(periods=simulate_days * 24)
    return load_scenario(network)


def build_behavior(scenario: ScenarioData) -> PyPSABehavior:
    """Return the environment behavior driving *scenario*'s PyPSA network.

    :param scenario: Scenario bundle from :func:`resolve_scenario`/:func:`load_scenario`.
    """
    return PyPSABehavior.from_scenario(scenario)


def filter_and_cache_statics(
    behavior: PyPSABehavior, refs: Sequence[ComponentRef]
) -> tuple[list[ComponentRef], dict[ComponentRef, dict]]:
    """Drop components with zero or non-finite nominal power.

    Returns the surviving refs plus a ``{ref: statics}`` cache. ``get_statics``
    rebuilds the component's full observer registry on every call, so scenarios
    that need a component's static row more than once should fetch it here
    once and reuse the returned dict instead of calling ``get_statics`` again.

    :param behavior: Environment behavior to query, from :func:`build_behavior`.
    :param refs: Candidate component refs (e.g. from ``get_components_by_type``).
    """
    statics_by_ref: dict[ComponentRef, dict] = {}
    kept: list[ComponentRef] = []
    for ref in refs:
        statics = behavior.get_statics(ref)
        p_nom = statics.get("p_nom", 0.0)
        if not math.isfinite(p_nom) or p_nom == 0.0:
            continue
        statics_by_ref[ref] = statics
        kept.append(ref)
    return kept, statics_by_ref


def require_lossless_transport(loss_percent: float, algorithm_name: str) -> None:
    """Fail fast instead of deadlocking under simulated packet loss.

    FDGDM, Diffusion, Exact Diffusion, DEED-ADMM, and ADMM all advance a
    round only once *every* neighbour (or, for ADMM, every participant the
    coordinator awaits via ``asyncio.gather``) has replied for that round --
    there is no retry and no partial-quorum fallback. A single dropped
    message therefore hangs that round (and the whole simulation) forever
    rather than degrading gracefully, unlike the averaging-consensus
    algorithm, which has a catch-up path that lets a node skip ahead once
    any neighbour's message for a later iteration arrives.

    :param loss_percent: The scenario's configured comms packet-loss percentage.
    :param algorithm_name: Name to include in the error message.
    :raises ValueError: If *loss_percent* is not exactly zero.
    """
    if loss_percent != 0.0:
        raise ValueError(
            f"{algorithm_name} requires lossless message delivery (loss_percent=0.0); "
            f"got {loss_percent!r}. This algorithm waits for every neighbour's reply "
            "each round with no retry, so any packet loss can deadlock the run instead "
            "of degrading gracefully. Use --loss-percent 0 (or omit the flag)."
        )


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
        # Deliberately not routed through _keep_hourly: this df is a
        # multi-column, outer-joined frame where different agents' columns
        # tick at different raw timestamps, so many cells in any given row are
        # already NaN. groupby(...).last() keeps each column's own last
        # non-null value per bucket; picking one row index per bucket (as
        # _keep_hourly does for a single (t, Y) series) would instead drop
        # columns that simply didn't tick on that bucket's literal last row.
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

    Negative recorded power (storage charging) is clipped to zero before
    weighting: charging is not credited as negative cost at the unit's own
    marginal cost, so scenarios that record raw charging power (consensus,
    ADMM, diffusion, DEED-ADMM) and FDGDM (which clamps its "P" recording to
    ``max(0, ·)``) produce comparable totals.

    Returns ``(total, per_step)`` where *per_step* is a ``"cost:total"``
    series indexed by the de-duplicated *t_P*, suitable for merging into the
    recordings CSV via :func:`_write_agent_recordings_csv`'s ``extra``
    parameter.
    """
    t_arr = np.asarray(t_P, dtype=float)
    if step_s > 0 and t_arr.size:
        t_arr, Y_P = _keep_hourly(t_arr, Y_P, step_s)

    cost_vec = np.array([cost_by_aid.get(aid, 0.0) for aid in labels_P])
    per_step = pd.Series(
        (np.maximum(Y_P, 0.0) * cost_vec).sum(axis=1), index=t_arr, name="cost:total"
    )
    return float(per_step.sum()), per_step


# ---------------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------------


def build_group_map(
    refs: Sequence[ComponentRef],
    *,
    statics_by_ref: dict[ComponentRef, dict] | None = None,
    behavior: Any | None = None,
) -> dict[str, str]:
    """Map ``component_id -> carrier`` for the stacked / per-unit plots.

    The carrier comes from the cached statics row when *statics_by_ref* is
    given, else from ``behavior.get_statics(ref)``.  Components without a
    carrier fall back to their ``element_type`` (``"storage"`` etc.), then to
    ``"other"``.  ``LOAD`` refs are skipped (loads never enter the ``"P"``
    recording).  Keys are ``str(component_id)`` so they match the agent AIDs
    used as plot labels.
    """
    out: dict[str, str] = {}
    for ref in refs:
        if getattr(ref, "element_type", None) == LOAD:
            continue
        statics: dict | None = None
        if statics_by_ref is not None:
            statics = statics_by_ref.get(ref)
        if statics is None and behavior is not None:
            statics = behavior.get_statics(ref)
        carrier = (statics or {}).get("carrier")
        if carrier is not None and not pd.isna(carrier) and str(carrier).strip():
            out[str(ref.component_id)] = str(carrier)
        else:
            out[str(ref.component_id)] = str(
                getattr(ref, "element_type", None) or "other"
            )
    return out


def carrier_style(net: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``({carrier: colour}, {carrier: nice_name})`` from ``net.carriers``.

    Both dicts are empty when the network carries no carrier metadata (e.g. the
    toy network).
    """
    colors: dict[str, str] = {}
    names: dict[str, str] = {}
    carriers = getattr(net, "carriers", None)
    if carriers is None or len(carriers) == 0:
        return colors, names
    for carrier, row in carriers.iterrows():
        col = row.get("color") if hasattr(row, "get") else None
        if isinstance(col, str) and col.strip():
            colors[str(carrier)] = col
        nice = row.get("nice_name") if hasattr(row, "get") else None
        if isinstance(nice, str) and nice.strip():
            names[str(carrier)] = nice
    return colors, names


def write_scenario_outputs(
    world: Any,
    *,
    name_base: str,
    cost_by_aid: dict[str, float],
    group_map: dict[str, str],
    net: Any,
    label: str = "",
    snapshot_step_s: float = 3600.0,
) -> float:
    """Write the standard scenario result files and return the total cost.

    Emits ``<name_base>-{df.csv,stacked.pdf,observation.pdf,balance.pdf,cost.pdf}``:

    * ``df.csv`` — every per-agent recording, one row per hourly snapshot
      (unchanged format; consumed by ``compare_costs.py``).
    * ``stacked.pdf`` — generation stacked by carrier, with the demand target.
    * ``observation.pdf`` — per-unit power as a grid of per-carrier subplots.
    * ``balance.pdf`` — total generation vs demand + residual.
    * ``cost.pdf`` — dispatch cost per timestep.
    """
    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")

    t_P, Y_P = _keep_hourly(t_P, Y_P)
    t_t, Y_t = _keep_hourly(t_t, Y_t)
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    m = min(len(t_P), len(target_series))
    t_P, Y_P, target_series = t_P[:m], Y_P[:m], target_series[:m]

    total_cost, cost_series = compute_overall_cost(cost_by_aid, t_P, Y_P, labels_P)
    logger.info("%s: overall cost = %.2f", name_base, total_cost)
    annotation = f"Total cost: {total_cost:,.2f}"
    suffix = f" – {label}" if label else ""

    _write_agent_recordings_csv(
        world, f"{name_base}-df.csv", snapshot_step_s=snapshot_step_s, extra=cost_series
    )

    style_colors, nice_names = carrier_style(net)
    carriers = sorted(set(group_map.values()))
    band_colors = resolve_carrier_colors(carriers, style_colors)
    band_order = order_carriers(carriers)
    hours = np.asarray(t_P, dtype=float) / 3600.0

    stacked_area(
        hours,
        Y_P,
        labels_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title=f"Stacked power{suffix}",
        annotation=annotation,
        groups=group_map,
        colors=band_colors,
        order=band_order,
        write_to=f"{name_base}-stacked.pdf",
    )

    per_unit_small_multiples(
        hours,
        Y_P,
        labels_P,
        group_map,
        xlabel="Hour",
        ylabel="P in MW",
        title=f"Per-unit power{suffix}",
        annotation=annotation,
        carrier_names=nice_names,
        write_to=f"{name_base}-observation.pdf",
    )

    generation_vs_demand(
        hours,
        Y_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title=f"Generation vs demand{suffix}",
        annotation=annotation,
        write_to=f"{name_base}-balance.pdf",
    )

    cost_over_time(
        np.asarray(cost_series.index, dtype=float) / 3600.0,
        cost_series.to_numpy(),
        title=f"Cost per timestep{suffix}",
        annotation=annotation,
        write_to=f"{name_base}-cost.pdf",
    )

    return total_cost


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
        demand_target: np.ndarray | None = None,
        balance_label: str = "",
        balance_tol: float = 0.01,
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
        self._demand_target = (
            None if demand_target is None else np.asarray(demand_target, dtype=float)
        )
        self._balance_label = balance_label
        self._balance_tol = balance_tol

        self._ready: bool = False
        self._finished_aids: set[str] = set()
        self._pending: dict[int, float] = {}

        self.demand_map: dict[Any, list[float]] = {}
        self.target: float = 0.0  # recorded by the world for plotting
        #: Largest per-timestep |Σ schedules − demand| / demand after the
        #: optimization finished; ``None`` until verified (or if no
        #: ``demand_target`` was supplied).
        self.balance_max_rel_gap: float | None = None

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
            self._verify_balance()
            for idx in sorted(self._pending):
                self._apply_schedule_index(idx, self._pending[idx])
            self._pending.clear()

    def _verify_balance(self) -> None:
        """Check the finished schedules against the known demand target.

        Ces et al. 2025 verify power balance before accepting a distributed
        solution; without this an unconverged run silently dispatches an
        infeasible schedule whose "cost" is meaningless (unserved demand looks
        cheap, over-generation looks expensive).
        """
        if self._demand_target is None:
            return
        m = len(self._demand_target)
        total = np.zeros(m, dtype=float)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            arr = np.asarray(schedule, dtype=float).ravel()
            n = min(m, arr.size)
            total[:n] += arr[:n]
        gap = self._demand_target - total
        rel = np.abs(gap) / np.maximum(np.abs(self._demand_target), 1e-9)
        self.balance_max_rel_gap = float(rel.max()) if rel.size else 0.0
        label = self._balance_label or type(self).__name__
        if self.balance_max_rel_gap > self._balance_tol:
            logger.error(
                "%s: schedules violate power balance — max per-timestep "
                "|generation − demand| is %.1f%% of demand (tolerance %.1f%%). "
                "The optimization did not converge; treat this run's cost as "
                "invalid.",
                label,
                100.0 * self.balance_max_rel_gap,
                100.0 * self._balance_tol,
            )
        else:
            logger.info(
                "%s: power balance verified — max per-timestep gap %.3f%% of demand.",
                label,
                100.0 * self.balance_max_rel_gap,
            )

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
