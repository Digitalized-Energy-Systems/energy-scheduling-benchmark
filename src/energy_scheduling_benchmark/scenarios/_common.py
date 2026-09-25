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
from distributed_resource_optimization.carrier.mango import DistributedOptimizationRole
from mango import Role
from mango.simulation.world import record_agent_having

from energy_scheduling_benchmark import (
    RENEWABLE,
    STORAGE,
    THERMAL,
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
    stacked_area,
    visualize_results,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_toy_network",
    "load_scenario",
    "resolve_scenario",
    "build_behavior",
    "build_world",
    "_clip_scenario",
    "_lookup_ts",
    "_scalar",
    "_keep_hourly",
    "_write_agent_recordings_csv",
    "compute_overall_cost",
    "compute_linear_energy_cost",
    "build_scenario_argparser",
    "require_lossless_transport",
    "filter_and_cache_statics",
    "collect_generator_refs",
    "build_demand_horizon",
    "DemandHorizon",
    "build_p_max_vec",
    "capacity_scaled_epsilon",
    "EpsilonScaling",
    "StorageParams",
    "install_standard_recordings",
    "write_scenario_outputs",
    "run_scenario_main",
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


def build_world(
    scenario: ScenarioData, *, delay_s: float, loss_percent: float
) -> tuple[Any, PyPSABehavior]:
    """Build the mango simulation ``world`` + environment behavior for *scenario*.

    Every scenario wires the same environment/communication-sim/world triple;
    this is that wiring in one place.

    :param scenario: Scenario bundle to drive, already clipped to the
        simulated window (see :func:`_clip_scenario`).
    :param delay_s: Comms delay (seconds) for :class:`SimpleCommunicationSimulation`.
    :param loss_percent: Comms packet-loss percentage for the same.
    """
    from mango.simulation.communication import SimpleCommunicationSimulation
    from mango.simulation.environment import DefaultEnvironment
    from mango.simulation.world import create_world

    behavior = build_behavior(scenario)
    com_sim = SimpleCommunicationSimulation(
        default_delay_s=delay_s, loss_percent=loss_percent
    )
    world = create_world(
        start_time=0.0,
        communication_sim=com_sim,
        environment=DefaultEnvironment(behavior=behavior),
    )
    return world, behavior


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


def collect_generator_refs(
    behavior: PyPSABehavior, *, exclude_hydro: bool = True
) -> tuple[list[ComponentRef], dict[ComponentRef, dict]]:
    """Collect thermal/renewable/storage refs, filter, and cache their statics.

    *exclude_hydro* drops any component whose id contains ``"hydro"`` — hydro
    dispatch is driven by natural inflow, which none of the distributed
    scenarios' storage/generator actors model (they assume freely schedulable
    charge/discharge or a pure cost curve), so hydro units cannot participate
    meaningfully. This is a substring match on ``component_id``, not a
    carrier/type check, so a plant merely named e.g. ``"...hydrogen..."``
    would also be excluded — a known sharp edge, not a deliberate rule.

    Also drops zero/non-finite ``p_nom`` components via
    :func:`filter_and_cache_statics`.

    :param behavior: Environment behavior to query.
    :param exclude_hydro: Whether to apply the hydro name filter.
    """
    gen_refs = behavior.get_components_by_type([THERMAL, RENEWABLE, STORAGE])
    if exclude_hydro:
        kept = [ref for ref in gen_refs if "hydro" not in ref.component_id]
        n_dropped = len(gen_refs) - len(kept)
        if n_dropped:
            logger.info(
                "collect_generator_refs: excluded %d component(s) by hydro name match",
                n_dropped,
            )
        gen_refs = kept
    return filter_and_cache_statics(behavior, gen_refs)


@dataclass
class DemandHorizon:
    """Aggregated demand vector + time bookkeeping shared by every scenario."""

    time_index: pd.Index
    #: Number of timesteps in the horizon (``len(time_index)``).
    horizon: int
    #: Aggregated (summed across loads) demand per timestep, MW.
    target_series: np.ndarray
    #: Maps rounded mango simulation seconds → index into ``target_series``.
    time_to_index: dict[float, int]
    start_dt: Any


def build_demand_horizon(
    behavior: PyPSABehavior,
    scenario: ScenarioData,
    load_refs: Sequence[ComponentRef],
    *,
    simulate_days: int | None,
) -> DemandHorizon:
    """Build the aggregated demand vector and simulation-time → index map.

    Sums each load's timeseries (reindexed onto the first load's index) into
    a single per-timestep demand vector, and maps mango's simulation-clock
    seconds to an index into that vector (both keyed by
    ``round(seconds, 6)`` — the two must agree on that rounding or a
    timestep's dispatch is silently dropped, see
    :meth:`PowerLoadAggregator._handle_load_info`).

    :param load_refs: Load component refs, e.g. from
        ``behavior.get_components_by_type([LOAD])``.
    :param simulate_days: If given, ``time_index`` is truncated to
        ``simulate_days * 24`` hours (matches the window ``_clip_scenario``
        already trimmed the timeseries to). Pass ``None`` to use the full
        length of the first load's timeseries as-is.
    :raises RuntimeError: If the first load's timeseries, or any load's
        timeseries, cannot be found in *scenario*.
    """
    load_series_0 = _lookup_ts(scenario, load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    if simulate_days is not None:
        time_index = time_index[: simulate_days * 24]
    horizon = len(time_index)

    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = _lookup_ts(scenario, ref)
        if s is None:
            raise RuntimeError(f"Load timeseries missing for {ref}.")
        target_series += np.asarray(s.reindex(time_index), dtype=float)

    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

    return DemandHorizon(time_index, horizon, target_series, time_to_index, start_dt)


def build_p_max_vec(
    scenario: ScenarioData,
    ref: ComponentRef,
    statics: dict,
    time_index: pd.Index,
    horizon: int,
) -> np.ndarray:
    """Build a per-timestep max-power vector for a generator, aligned to *time_index*.

    Falls back to a constant ``p_nom`` vector when *ref* has no timeseries.
    For :data:`RENEWABLE` refs, the timeseries is treated as a per-unit
    availability (PyPSA convention) and scaled by ``p_nom``; other refs use
    the raw timeseries values directly.
    """
    p_nom = float(statics.get("p_nom", 0.0))
    ts = _lookup_ts(scenario, ref)
    if ts is None:
        return np.full(horizon, p_nom, dtype=float)
    values = np.asarray(ts.reindex(time_index), dtype=float)
    if ref.element_type == RENEWABLE:
        return values * p_nom
    return values


@dataclass
class EpsilonScaling:
    """Result of :func:`capacity_scaled_epsilon`."""

    #: Per-generator price-response epsilon, keyed by ``component_id``
    #: (non-storage refs only).
    eps_by_aid: dict[str, float]
    #: The shared price-response band width (epsilon * p_nom) all epsilons
    #: were derived from.
    target_band: float
    #: ``max(costs) - min(costs)`` across non-storage refs (``0.0`` if fewer
    #: than 2).
    cost_range: float
    #: Marginal costs of the non-storage refs the band was computed from
    #: (empty if fewer than 2).
    costs_all: list[float]


def capacity_scaled_epsilon(
    gen_refs: Sequence[ComponentRef],
    statics_by_ref: dict[ComponentRef, dict],
    *,
    default_epsilon: float = 0.1,
    cost_fraction: float = 0.1,
) -> EpsilonScaling:
    """Per-generator epsilon (capacity-scaled), shared by consensus/diffusion.

    A single shared epsilon gives every generator the same price-response
    band width (epsilon * p_nom) above its own marginal cost. For merit
    order to hold, that band must be small relative to the spread of
    marginal costs across generators — otherwise many generators are still
    in their partial "ramp" region at the clearing price simultaneously, so
    power gets spread roughly by capacity rather than sorted by cost. On
    real PyPSA-Eur networks generator capacities span orders of magnitude
    (tens to tens-of-thousands of MW); scaling the band off typical/mean
    capacity makes it far wider than the cost spread, breaking merit order.
    Scale epsilon inversely with p_nom so every generator's band is a small,
    fixed fraction of the cost spread, independent of its capacity.

    :param gen_refs: Generator refs (storage refs are excluded from the
        band computation and from the returned ``eps_by_aid``).
    :param statics_by_ref: Cached statics, e.g. from
        :func:`filter_and_cache_statics`/:func:`collect_generator_refs`.
    :param default_epsilon: Floor on the price-response band width.
    :param cost_fraction: Fraction of the cost spread the band should target.
    """
    nonstorage_refs = [ref for ref in gen_refs if ref.element_type != STORAGE]
    eps_by_aid: dict[str, float] = {}
    costs_all: list[float] = []
    cost_range = 0.0
    if len(nonstorage_refs) >= 2:
        costs_all = [
            float(statics_by_ref[r].get("marginal_cost", 0.0)) for r in nonstorage_refs
        ]
        cost_range = max(costs_all) - min(costs_all)
        target_band = max(default_epsilon, cost_fraction * cost_range)
    else:
        target_band = default_epsilon
    for ref in nonstorage_refs:
        p_nom_ref = float(statics_by_ref[ref].get("p_nom", 0.0))
        eps_by_aid[ref.component_id] = target_band / max(p_nom_ref, 1.0)
    return EpsilonScaling(eps_by_aid, target_band, cost_range, costs_all)


@dataclass
class StorageParams:
    """Storage actor parameters derived from PyPSA statics.

    Shared by the scenarios whose storage-parameter extraction is otherwise
    byte-for-byte identical (consensus, ADMM, diffusion, exact diffusion).
    DEED-ADMM's inline extraction happens to use the same defaults as this
    helper but is not yet wired to it. FDGDM uses materially *different*
    defaults for some of these fields (``max_hours=6.0`` vs. this helper's
    ``100.0``, SOC-initial fallback ``0.0`` vs. ``0.5 * e_max``, efficiency
    fallback ``1.0`` vs. ``0.95`` — see ``fdgdm._schedule_storage_soc``);
    that is tracked as a known cross-scenario inconsistency, not silently
    reconciled here, since it would change FDGDM's dispatch numbers on any
    network with under-specified storage statics.
    """

    e_max: float
    p_charge_max: float
    p_discharge_max: float
    #: Floored to at least ``1e-6`` (never exactly 0, to keep it usable as a
    #: divisor in the actor's dynamics).
    eta_charge: float
    eta_discharge: float
    #: Initial state of charge, as a fraction of ``e_max`` in ``[0, 1]``.
    e_initial: float

    @classmethod
    def from_statics(
        cls,
        statics: dict,
        p_nom: float,
        *,
        default_max_hours: float = 100.0,
        default_efficiency: float = 0.95,
    ) -> StorageParams:
        p_min_pu = float(statics.get("p_min_pu", -1.0))
        p_max_pu = float(statics.get("p_max_pu", 1.0))
        p_charge_max = max(0.0, (-p_min_pu * p_nom) if p_min_pu < 0.0 else p_nom)
        p_discharge_max = max(0.0, p_max_pu * p_nom)

        max_hours = float(statics.get("max_hours", default_max_hours))
        e_max = max(1e-6, p_nom * max_hours)

        eta_charge = float(
            statics.get(
                "efficiency_store", statics.get("efficiency_charge", default_efficiency)
            )
        )
        eta_discharge = float(
            statics.get(
                "efficiency_dispatch",
                statics.get("efficiency_discharge", default_efficiency),
            )
        )

        soc_initial_raw = statics.get("state_of_charge_initial", np.nan)
        if np.isfinite(soc_initial_raw) and float(soc_initial_raw) > 1e-9:
            soc_initial_abs = float(soc_initial_raw)
        else:
            # PyPSA default is often 0.0; treat that as "unspecified" here so
            # storage can participate in the benchmark without extra config.
            soc_initial_abs = 0.5 * e_max
        e_initial = float(np.clip(soc_initial_abs / e_max, 0.0, 1.0))

        return cls(
            e_max=e_max,
            p_charge_max=p_charge_max,
            p_discharge_max=p_discharge_max,
            eta_charge=max(1e-6, eta_charge),
            eta_discharge=max(1e-6, eta_discharge),
            e_initial=e_initial,
        )


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
    # NB: this is a linear energy cost (Σ marginal_cost * dispatched MW) only —
    # no startup/no-load/commitment cost. "compute_overall_cost" oversells
    # that; compute_linear_energy_cost (below) is the clearer name to use in
    # new code, kept as an alias for now rather than a breaking rename.
    return float(per_step.sum()), per_step


#: Clearer alias for :func:`compute_overall_cost` — same function, see its
#: docstring for the "linear energy cost only" caveat. Prefer this name in
#: new code; ``compute_overall_cost`` is kept for existing callers.
compute_linear_energy_cost = compute_overall_cost


def build_scenario_argparser(
    description: str | None,
    *,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    lossless_only: bool = False,
    with_balance_tol: bool = False,
) -> argparse.ArgumentParser:
    """Build the CLI parser shared by all scenario entry points.

    *extra_args*, if given, is called with the parser to add algorithm-specific
    flags (e.g. DEED-ADMM's ``--gamma``/``--max-iter``).

    :param lossless_only: If true, this scenario's algorithm advances a round
        only once every neighbour has replied (no retry/partial-quorum
        fallback — see :func:`require_lossless_transport`), so
        ``--loss-percent`` only documents that constraint rather than
        offering a working knob.
    :param with_balance_tol: If true, add ``--balance-tol`` (the power-balance
        tolerance passed to :class:`PowerLoadAggregator`) and ``--strict``
        (fail the run instead of writing outputs when that check fails).
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
    parser.add_argument("--delay-s", type=float, default=0.02, help="Comms delay (s).")
    parser.add_argument(
        "--loss-percent",
        type=float,
        default=0.0,
        help=(
            "Comms packet-loss percentage. This scenario waits for every "
            "neighbour's reply each round with no retry, so any non-zero "
            "value raises ValueError instead of running (use 0, or omit "
            "the flag)."
            if lossless_only
            else "Comms packet-loss percentage."
        ),
    )
    parser.add_argument("--name-base", type=str, default=default_name_base)
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    if with_balance_tol:
        # Dests "balance_tol"/"strict" — keep in sync with the matching
        # names hardcoded into run_scenario_main's `with_balance_tol` branch
        # below, which forwards them to execute_test_case.
        parser.add_argument(
            "--balance-tol",
            type=float,
            default=0.01,
            help=(
                "Max allowed per-timestep |generation - demand| / demand "
                "before a finished run is flagged as not converged (default: "
                "0.01, i.e. 1%%)."
            ),
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help=(
                "Exit with an error instead of writing CSV/PDF outputs if "
                "the power-balance check fails (default: log a warning and "
                "write the outputs anyway)."
            ),
        )
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
            # All schedules are pre-filled; nothing to run. Still verify the
            # balance here — schedule_by_aid was already populated by the
            # scenario's own pre-scheduling before the world started, so the
            # check is meaningful even though no distributed round ever ran.
            # Skipping this left balance_max_rel_gap permanently None on this
            # path (e.g. FDGDM with 0-1 thermal generators), which silently
            # defeated the --strict CLI flag.
            self._ready = True
            self._verify_balance()
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
        # `carrier._parent` is a distributed_resource_optimization-internal
        # attribute (the mango Role wrapping this carrier); there is no
        # public accessor for it today.
        role = carrier._parent
        aid = role.context.aid

        obj = algorithm
        try:
            for part in schedule_attr.split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(
                f"{algorithm_label}: finished algorithm for {aid!r} has no "
                f"attribute path {schedule_attr!r} ({exc}); the run may have "
                "aborted before producing a schedule."
            ) from exc
        schedule_by_aid[aid] = np.asarray(obj, dtype=float).copy()

        converged = getattr(algorithm, "converged", None)
        iterations = getattr(algorithm, "iterations", None)
        if converged is False:
            logger.warning(
                "%s did not converge for %s after %s iteration(s); schedule may "
                "be suboptimal.",
                algorithm_label,
                aid,
                iterations if iterations is not None else "an unknown number of",
            )

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            asyncio.create_task(
                role.context.send_message(finished_message_type(aid=aid), leader_addr)
            )
        logger.info(
            "%s finished for %s (schedule len=%s%s)",
            algorithm_label,
            aid,
            schedule_by_aid[aid].size,
            f", iterations={iterations}" if iterations is not None else "",
        )

    return handle_finished


# ---------------------------------------------------------------------------
# Recordings + output
# ---------------------------------------------------------------------------


def install_standard_recordings(
    world: Any,
    behavior: PyPSABehavior,
    *,
    role_cls: type = DistributedOptimizationRole,
    clip_negative: bool = False,
) -> None:
    """Record the leader's ``"target"`` and each generator's ``"P"`` per timestep.

    ``"target"`` is summed by :func:`write_scenario_outputs`; ``"P"`` is
    summed by :func:`compute_overall_cost`, which itself clips negative
    values (storage charging) to 0 before costing. *clip_negative* additionally
    clips at recording time (FDGDM's convention), so a negative set-point
    never shows as a downward bar in the stacked generation plot either.

    :param role_cls: Role class each generator's "P" is recorded from —
        must be (a subclass of) :class:`DistributedOptimizationRole`.
    """
    record_agent_having(
        world,
        "target",
        PowerLoadAggregator,
        lambda a: next(
            (r.target for r in a.roles if isinstance(r, PowerLoadAggregator)), 0.0
        ),
    )
    if clip_negative:

        def _extract(a: Any) -> float:
            return max(0.0, float(behavior.observe(a.aid, "active_power") or 0.0))
    else:

        def _extract(a: Any) -> float:
            return float(behavior.observe(a.aid, "active_power") or 0.0)

    record_agent_having(world, "P", role_cls, _extract)


def write_scenario_outputs(
    world: Any,
    *,
    name_base: str,
    cost_by_aid: dict[str, float],
    stacked_title: str = "Stacked power",
    cost_title: str = "Cost per timestep",
    clip_target_to_p: bool = True,
    aggregator: PowerLoadAggregator | None = None,
    balance_tol: float | None = None,
    strict: bool = False,
) -> float:
    """Write the ``-df.csv``/``-observation.pdf``/``-stacked.pdf``/``-cost.pdf`` outputs.

    Shared by every scenario: read back the "P"/"target" recordings
    installed by :func:`install_standard_recordings`, collapse them to one
    sample per hour, compute the overall cost, then write the CSV and the
    three plots.

    :param clip_target_to_p: Truncate ``P``/``target`` to the shorter of the
        two lengths before costing/plotting (every scenario except FDGDM
        does this; FDGDM's simulation window already ends exactly on its
        last snapshot, so the two recordings are never mismatched there).
    :param aggregator: When given together with *strict*, raise instead of
        writing outputs if ``aggregator.balance_max_rel_gap`` exceeds
        *balance_tol* -- or was never computed at all (the schedule never
        finished), which is just as much a failure as an out-of-tolerance
        gap. Callers with a ``PowerLoadAggregator``-based balance check
        should pass this instead of hand-rolling the same raise.
    :param balance_tol: Tolerance to compare *aggregator*'s gap against;
        required (and otherwise ignored) when *aggregator* and *strict* are
        both given.
    :raises RuntimeError: If *strict* is set and the balance check failed or
        never ran.
    :returns: The total cost (same value logged and used in the annotation).
    """
    if aggregator is not None and strict:
        gap = aggregator.balance_max_rel_gap
        if gap is None:
            raise RuntimeError(
                f"{name_base}: power balance was never verified (the "
                "distributed schedule never finished); refusing to write "
                "outputs because --strict was given."
            )
        if gap > balance_tol:
            raise RuntimeError(
                f"{name_base}: power balance check failed (max relative gap "
                f"{gap:.4f} > tolerance {balance_tol}); refusing to write "
                "outputs because --strict was given."
            )

    t_P, Y_P, labels_P = agent_recording_as_plottable(world, "P")
    t_t, Y_t, _ = agent_recording_as_plottable(world, "target")

    # Drop sub-second convergence-phase noise (iterative algorithms tick
    # multiple times per hour): keep the last recorded state per hourly
    # snapshot so the CSV and plots show one row per PyPSA timestep.
    t_P, Y_P = _keep_hourly(t_P, Y_P)
    t_t, Y_t = _keep_hourly(t_t, Y_t)
    target_series = Y_t[:, 0] if Y_t.size else np.zeros(len(t_P))
    if clip_target_to_p:
        m = min(len(t_P), len(target_series))
        t_P, Y_P, target_series = t_P[:m], Y_P[:m], target_series[:m]

    total_cost, cost_series = compute_overall_cost(cost_by_aid, t_P, Y_P, labels_P)
    logger.info("%s: overall cost = %.2f", name_base, total_cost)
    annotation = f"Total cost: {total_cost:,.2f}"

    _write_agent_recordings_csv(
        world, f"{name_base}-df.csv", snapshot_step_s=3600.0, extra=cost_series
    )

    visualize_results(
        world, write_to=f"{name_base}-observation.pdf", annotation=annotation
    )

    stacked_area(
        np.asarray(t_P) / 3600.0,
        Y_P,
        labels_P,
        target_series,
        xlabel="Hour",
        ylabel="P in MW",
        title=stacked_title,
        annotation=annotation,
        write_to=f"{name_base}-stacked.pdf",
    )

    cost_over_time(
        np.asarray(cost_series.index, dtype=float) / 3600.0,
        cost_series.to_numpy(),
        title=cost_title,
        annotation=annotation,
        write_to=f"{name_base}-cost.pdf",
    )

    return total_cost


def run_scenario_main(
    execute_test_case: Callable[..., Any],
    *,
    doc: str | None,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_kwargs: Sequence[str] = (),
    lossless_only: bool = False,
    with_balance_tol: bool = False,
    argv: list[str] | None = None,
) -> None:
    """Shared ``main()`` body: parse CLI args, resolve the scenario, run it.

    *extra_kwargs* names CLI args (added via *extra_args*, or ``"balance_tol"``/
    ``"strict"`` when *with_balance_tol* is set) to forward to
    *execute_test_case* verbatim, e.g. ``("gamma", "max_iter")`` for
    DEED-ADMM's ``--gamma``/``--max-iter``.
    """
    parser = build_scenario_argparser(
        doc,
        default_name_base=default_name_base,
        extra_args=extra_args,
        lossless_only=lossless_only,
        with_balance_tol=with_balance_tol,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenario = resolve_scenario(args.network, simulate_days=args.simulate_days)

    # "balance_tol"/"strict" here must match the --balance-tol/--strict
    # `dest`s added in build_scenario_argparser's with_balance_tol branch.
    names = (
        (*extra_kwargs, "balance_tol", "strict") if with_balance_tol else extra_kwargs
    )
    kwargs = {name: getattr(args, name) for name in names}
    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=args.delay_s,
            loss_percent=args.loss_percent,
            name_base=args.name_base,
            simulate_days=args.simulate_days,
            **kwargs,
        )
    )
