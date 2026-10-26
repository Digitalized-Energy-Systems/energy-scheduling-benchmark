"""Unit, demand and cost-model helpers shared by the scenarios."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace

import numpy as np
import pandas as pd

from energy_scheduling_benchmark import (
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PyPSABehavior,
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
)

logger = logging.getLogger(__name__)


#: Cost models of ``--cost-model`` (decision Q2).
COST_MODELS = ("linear", "quadratic")


#: Quadratic cost factor k of ``c₂ = k · c₁ / p_nom`` (decision Q17): a unit's
#: marginal cost rises from c₁ at 0 MW to (1 + 2k)·c₁ at full output. 0.11 is
#: the mean of c·P_max/b over the three units of Wood & Wollenberg, *Power
#: Generation, Operation and Control*, Example 4E (0.091, 0.129, 0.123).
QUADRATIC_K = 0.11


def quadratic_cost_coefficient(statics: dict, cost_model: str) -> float:
    """``c₂`` (€/MW²h) of one generator for *cost_model* (0 for ``linear``).

    The network's ``marginal_cost_quadratic`` if set (``prepare_network.py``
    writes it), else ``c₂ = QUADRATIC_K · max(c₁, 0) / p_nom``; zero-cost units
    (renewables) and units without ``p_nom`` stay linear. Storage is not passed
    here (its cost stays linear, decision Q15).
    """
    if cost_model not in COST_MODELS:
        raise ValueError(f"cost_model must be one of {COST_MODELS}, got {cost_model!r}")
    if cost_model == "linear":
        return 0.0
    c2 = float(statics.get("marginal_cost_quadratic", 0.0))
    if c2 > 0:
        return c2
    c1 = max(float(statics.get("marginal_cost", 0.0)), 0.0)
    p_nom = float(statics.get("p_nom", 0.0))
    return QUADRATIC_K * c1 / p_nom if p_nom > 0 else 0.0


def thermal_cost_quad(
    gen_refs: Sequence[ComponentRef], statics_by_ref: dict, cost_model: str
) -> dict[str, float]:
    """``{component_id: c₂}`` of the thermal units with a quadratic term.

    Empty for ``cost_model="linear"``. Only thermal units get c₂ (heat-rate
    curves, decision Q17); renewables and storage stay linear everywhere.
    """
    out: dict[str, float] = {}
    for ref in gen_refs:
        if ref.element_type != THERMAL:
            continue
        c2 = quadratic_cost_coefficient(statics_by_ref[ref], cost_model)
        if c2 > 0:
            out[ref.component_id] = c2
    return out


def apply_cost_model(
    eps: EpsilonScaling,
    gen_refs: Sequence[ComponentRef],
    statics_by_ref: dict,
    cost_model: str,
) -> tuple[EpsilonScaling, dict[str, float]]:
    """Price-response ε for *cost_model* plus the c₂ table for the cost metric.

    With quadratic costs a thermal unit's ε is its true curvature 2c₂ (Jian et
    al. eq. 23: P = (λ − b)/(2a)) instead of the capacity-scaled proxy; all
    other units keep the proxy. Linear: returns *eps* unchanged and ``{}``.
    """
    cost_quad = thermal_cost_quad(gen_refs, statics_by_ref, cost_model)
    if not cost_quad:
        return eps, {}
    eps_by_aid = dict(eps.eps_by_aid)
    for aid, c2 in cost_quad.items():
        eps_by_aid[aid] = 2.0 * c2
    return _dataclass_replace(eps, eps_by_aid=eps_by_aid), cost_quad


def price_sensitivity(
    eps: EpsilonScaling, gen_refs: Sequence[ComponentRef], statics_by_ref: dict
) -> float:
    """Σ 1/εᵢ (MW per €/MWh) over *gen_refs*; storage uses band / p_nom."""
    total = 0.0
    for ref in gen_refs:
        e = eps.eps_by_aid.get(ref.component_id)
        if e is None:
            e = eps.target_band / max(float(statics_by_ref[ref].get("p_nom", 0.0)), 1.0)
        total += 1.0 / e
    return total


def filter_and_cache_statics(
    behavior: PyPSABehavior, refs: Sequence[ComponentRef]
) -> tuple[list[ComponentRef], dict[ComponentRef, dict]]:
    """Drop components with zero or non-finite nominal power.

    Returns the surviving refs plus a ``{ref: statics}`` cache. ``get_statics``
    rebuilds the component's full observer registry on every call, so scenarios
    that need a component's static row more than once should fetch it here
    once and reuse the returned dict instead of calling ``get_statics`` again.

    :param behavior: Environment behavior to query, from :func:`build_world`.
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
    load_series_0 = scenario.timeseries.get(load_refs[0])
    if load_series_0 is None:
        raise RuntimeError("Load timeseries not found in scenario.timeseries.")

    time_index = load_series_0.index
    if simulate_days is not None:
        time_index = time_index[: simulate_days * 24]
    horizon = len(time_index)

    target_series = np.zeros(horizon, dtype=float)
    for ref in load_refs:
        s = scenario.timeseries.get(ref)
        if s is None:
            raise RuntimeError(f"Load timeseries missing for {ref}.")
        target_series += np.asarray(s.reindex(time_index), dtype=float)

    start_dt = behavior.start_datetime
    time_to_index: dict[float, int] = {}
    for i, ts in enumerate(time_index):
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        sim_seconds = (dt - start_dt).total_seconds()
        time_to_index[round(float(sim_seconds), 6)] = i

    return DemandHorizon(time_index, horizon, target_series, time_to_index)


def build_p_max_vec(
    scenario: ScenarioData,
    ref: ComponentRef,
    statics: dict,
    time_index: pd.Index,
    horizon: int,
) -> np.ndarray:
    """Build a per-timestep max-power vector for a generator, aligned to *time_index*.

    Falls back to the static ``p_max_pu * p_nom`` (e.g. nuclear at 0.871) when
    *ref* has no timeseries; a timeseries overrides it, as in PyPSA.
    The timeseries is a per-unit availability (``generators_t.p_max_pu``,
    PyPSA convention) and is scaled by ``p_nom`` — for thermal units too
    (e.g. scigrid-de's wind units, classified thermal by carrier) — except
    where ``extract_timeseries`` fell back to ``generators_t.p_set``, in MW.
    """
    p_nom = float(statics.get("p_nom", 0.0))
    ts = scenario.timeseries.get(ref)
    if ts is None:
        return np.full(horizon, p_nom * float(statics.get("p_max_pu", 1.0)))
    values = np.asarray(ts.reindex(time_index), dtype=float)
    gens_t = getattr(scenario.net, "generators_t", None)
    cid = ref.component_id
    in_mw = (
        gens_t is not None
        and cid in getattr(gens_t, "p_set", pd.DataFrame()).columns
        and cid not in getattr(gens_t, "p_max_pu", pd.DataFrame()).columns
    )
    return values if in_mw else values * p_nom


def build_unit_bounds(
    scenario: ScenarioData,
    ref: ComponentRef,
    statics: dict,
    time_index: pd.Index,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep ``(p_min, p_max)`` of a generator (MW), the same in every scenario.

    ``p_max`` is :func:`build_p_max_vec`. Only thermal units get a minimum,
    ``max(0, p_min_pu * p_nom)``, capped at each hour's ``p_max`` so a unit whose
    availability drops below its minimum stays feasible.
    """
    p_max = build_p_max_vec(scenario, ref, statics, time_index, horizon)
    p_min = np.zeros(horizon, dtype=float)
    if ref.element_type == THERMAL:
        p_min_mw = float(statics.get("p_min_pu", 0.0)) * float(
            statics.get("p_nom", 0.0)
        )
        p_min = np.minimum(max(0.0, p_min_mw), p_max)
    return p_min, p_max


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

    Shared by every scenario, so all of them model the same storage (Q15).
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
        p_charge_max = max(0.0, -p_min_pu * p_nom)  # p_min_pu >= 0: cannot charge
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
