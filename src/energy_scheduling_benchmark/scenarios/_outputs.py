"""Result output: cost and slack columns, run metadata and the CSV / PDF / meta files."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from energy_scheduling_benchmark import (
    LOAD,
    ComponentRef,
)
from energy_scheduling_benchmark.dispatch import SlackSpec
from energy_scheduling_benchmark.plotting import (
    cost_over_time,
    generation_vs_demand,
    order_carriers,
    per_unit_small_multiples,
    resolve_carrier_colors,
    stacked_area,
)

if TYPE_CHECKING:
    from energy_scheduling_benchmark.scenarios._roles import PowerLoadAggregator

logger = logging.getLogger(__name__)


def compute_overall_cost(
    cost_by_aid: dict[str, float],
    t_P: Sequence[float],
    Y_P: np.ndarray,
    labels_P: Sequence[str],
    cost_quad_by_aid: dict[str, float] | None = None,
) -> tuple[float, pd.Series]:
    """Total cost of an hourly dispatch: Σ_t Σ_i cost_by_aid[aid_i] · P_i(t).

    *t_P* / *Y_P* hold one row per hour (see
    :meth:`PowerLoadAggregator.hourly_dispatch`). Negative power (storage
    charging) is clipped to zero before weighting: charging is not credited
    as negative cost, and storage pays its cost on discharge only (Q15).

    Returns ``(total, per_step)`` where *per_step* is a ``"cost:total"``
    series indexed by *t_P*, merged into the CSV by
    :func:`write_scenario_outputs`.
    """
    t_arr = np.asarray(t_P, dtype=float)
    cost_vec = np.array([cost_by_aid.get(aid, 0.0) for aid in labels_P])
    dispatched = np.maximum(Y_P, 0.0)
    step_cost = (dispatched * cost_vec).sum(axis=1)
    if cost_quad_by_aid:
        # --cost-model quadratic: + Σ c₂ · P² per step (Q17).
        quad_vec = np.array([cost_quad_by_aid.get(aid, 0.0) for aid in labels_P])
        step_cost = step_cost + (dispatched**2 * quad_vec).sum(axis=1)
    per_step = pd.Series(step_cost, index=t_arr, name="cost:total")
    # NB: energy cost only (Σ marginal cost · MW, + c₂·P²) — no startup /
    # no-load / commitment cost.
    return float(per_step.sum()), per_step


def compute_slack(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a balance residual (demand − Σ dispatch, MW) into slack flows.

    Returns ``(import, export)``, both non-negative: a positive residual is a
    shortfall the slack supplies, a negative one a surplus it absorbs.
    """
    r = np.asarray(residual, dtype=float)
    return np.maximum(r, 0.0), np.maximum(-r, 0.0)


def _slack_columns(
    aggregator: PowerLoadAggregator,
    t_index: pd.Index,
    slack: SlackSpec,
    step_s: float = 3600.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slack import/export and their cost on the hourly rows *t_index*.

    From the aggregator's applied schedules (storage charging included).
    """
    by_bucket = aggregator.slack_by_bucket(step_s)
    rows = [by_bucket.get(int(t // step_s), (0.0, 0.0)) for t in t_index]
    imp = np.array([r[0] for r in rows], dtype=float)
    exp = np.array([r[1] for r in rows], dtype=float)
    frame = pd.DataFrame({"slack:import": imp, "slack:export": exp}, index=t_index)
    cost = pd.Series(
        imp * slack.import_price + exp * slack.export_price,
        index=t_index,
        name="cost:total",
    )
    return frame, cost


def build_group_map(
    refs: Sequence[ComponentRef],
    *,
    statics_by_ref: dict[ComponentRef, dict],
) -> dict[str, str]:
    """Map ``component_id -> carrier`` for the stacked / per-unit plots.

    The carrier comes from the cached statics row.  Components without a
    carrier fall back to their ``element_type`` (``"storage"`` etc.), then to
    ``"other"``.  ``LOAD`` refs are skipped (loads never enter the ``"P"``
    recording).  Keys are ``str(component_id)`` so they match the agent AIDs
    used as plot labels.
    """
    out: dict[str, str] = {}
    for ref in refs:
        if getattr(ref, "element_type", None) == LOAD:
            continue
        carrier = statics_by_ref.get(ref, {}).get("carrier")
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


def _run_meta(world: Any, aggregator: Any, total_cost: float) -> dict[str, Any]:
    """Outcome part of ``<name>-meta.json`` (the CLI adds dimensions, git, wall time)."""
    meta: dict[str, Any] = {
        "total_cost": total_cost,
        "messages_delivered": getattr(world, "messages_delivered_count", None),
        "simulated_seconds": float(
            getattr(getattr(world, "clock", None), "time", 0.0) or 0.0
        ),
    }
    iterations = [i for i in aggregator.iterations_by_aid.values() if i is not None]
    converged = [c for c in aggregator.converged_by_aid.values() if c is not None]
    stats = dict(getattr(aggregator, "algorithm_stats", {}) or {})
    if not iterations and stats.get("iterations") is not None:
        iterations = [stats["iterations"]]
    if not converged and stats.get("converged") is not None:
        converged = [stats["converged"]]
    if aggregator.degraded_generators:
        stop_reason = "degraded"  # zero-filled units report no converged flag
    elif converged:
        # Not "max_iter": ADMM also stops early after --max-missing-rounds.
        stop_reason = "converged" if all(converged) else "not_converged"
    else:
        stop_reason = "fixed_budget" if iterations else "n/a"
    for key in ("solver_status", "objective"):
        value = getattr(aggregator, key, None)
        if value is not None:
            stats[key] = value
    meta.update(
        {
            "balance_max_rel_gap": aggregator.balance_max_rel_gap,
            "iterations_max": max(iterations) if iterations else None,
            "iterations_min": min(iterations) if iterations else None,
            "iterations_mean": float(np.mean(iterations)) if iterations else None,
            "stop_reason": stop_reason,
            "outer_runs": getattr(aggregator, "runs", None),
            "degraded_generators": list(aggregator.degraded_generators),
            "hours_with_missing_loads": aggregator.hours_with_missing_loads,
            "algorithm_stats": stats,
        }
    )
    return meta


def file_sha256(path: Any) -> str | None:
    """SHA-256 of the file *path* (``None`` for built-in networks such as ``toy``)."""
    import hashlib

    p = Path(str(path))
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _git_state(module: Any) -> dict[str, Any]:
    """HEAD SHA and dirty flag of the git checkout containing *module* (None if none)."""
    try:
        here = str(Path(module.__file__).resolve().parent)
        sha = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError, AttributeError, TypeError):
        return {"sha": None, "dirty": None}


#: Files written by :func:`write_scenario_outputs`, as ``<name_base>-<suffix>``.
OUTPUT_SUFFIXES = (
    "df.csv",
    "meta.json",
    "stacked.pdf",
    "observation.pdf",
    "balance.pdf",
    "cost.pdf",
)


def write_scenario_outputs(
    world: Any,
    *,
    name_base: str,
    cost_by_aid: dict[str, float],
    stacked_title: str = "Stacked power",
    cost_title: str = "Cost per timestep",
    aggregator: PowerLoadAggregator,
    balance_tol: float | None = None,
    strict: bool = False,
    group_map: dict[str, str] | None = None,
    net: Any | None = None,
    label: str = "",
    slack: SlackSpec | None = None,
    cost_quad_by_aid: dict[str, float] | None = None,
) -> float:
    """Write the standard scenario result files and return the total cost.

    Shared by every scenario: take one row per hour from the schedules
    *aggregator* applied (:meth:`PowerLoadAggregator.hourly_dispatch`),
    compute the overall cost, then emit
    ``<name_base>-{df.csv,stacked.pdf,observation.pdf,balance.pdf,cost.pdf}``:

    * ``df.csv`` — ``target:``/``P:`` columns plus cost (and slack), one row
      per hourly snapshot (consumed by ``compare_costs.py``).
    * ``stacked.pdf`` — generation stacked by carrier, with the demand target.
    * ``observation.pdf`` — per-unit power as a grid of per-carrier subplots.
    * ``balance.pdf`` — total generation vs demand + residual.
    * ``cost.pdf`` — dispatch cost per timestep.

    :param aggregator: The scenario's leader role; its schedules are the
        output. With *strict*, raise instead of writing outputs if
        ``aggregator.balance_max_rel_gap`` exceeds *balance_tol* -- or was
        never computed at all (the schedule never finished), which is just
        as much a failure as an out-of-tolerance gap.
    :param balance_tol: Tolerance to compare *aggregator*'s gap against;
        required (and otherwise ignored) with *strict*.
    :param group_map: ``component_id -> carrier`` (see
        :func:`build_group_map`) used to aggregate the plots into carrier
        bands. Without it every component is its own group.
    :param net: PyPSA network whose ``carriers`` table supplies band colours
        and display names (see :func:`carrier_style`); optional.
    :param label: Appended to the per-unit and balance plot titles.
    :param slack: If given, a slack node closes the balance: per hour it imports
        ``max(0, demand − Σ schedules)`` and exports ``max(0, Σ schedules −
        demand)`` (from *aggregator*'s applied schedules), adds both at their
        prices to ``cost:total`` and writes ``slack:import``/``slack:export``
        (MW) to the CSV. Cannot be combined with *strict*.
    :raises RuntimeError: If *strict* is set and the balance check failed or
        never ran.
    :raises ValueError: If *slack* is given with *strict*.
    :returns: The total cost (same value logged and used in the annotation).
    """
    if slack is not None and strict:
        raise ValueError("slack cannot be combined with strict.")
    if not aggregator.ready:
        # Without this a run whose optimization never finished (e.g. a
        # participant's last messages were lost) wrote outputs silently: no
        # schedule was ever applied and the target column stayed at 0.
        missing = aggregator.missing_generators()
        logger.error(
            "%s: the distributed schedule never finished, so no set-points were "
            "applied; %d generator(s) never reported: %s. Treat this run's "
            "outputs as invalid.",
            name_base,
            len(missing),
            ", ".join(missing) or "(unknown)",
        )
    if strict:
        gap = aggregator.balance_max_rel_gap
        if gap is None:
            raise RuntimeError(
                f"{name_base}: power balance was never verified (the "
                "distributed schedule never finished; missing: "
                f"{', '.join(aggregator.missing_generators()) or 'none'}); "
                "refusing to write outputs because --strict was given."
            )
        if gap > balance_tol:
            raise RuntimeError(
                f"{name_base}: power balance check failed (max relative gap "
                f"{gap:.4f} > tolerance {balance_tol}); refusing to write "
                "outputs because --strict was given."
            )

    t_P, Y_P, labels_P, target_series = aggregator.hourly_dispatch()

    total_cost, cost_series = compute_overall_cost(
        cost_by_aid, t_P, Y_P, labels_P, cost_quad_by_aid=cost_quad_by_aid
    )
    extra: pd.Series | pd.DataFrame = cost_series
    slack_note = ""
    if slack is not None:
        slack_frame, slack_cost = _slack_columns(aggregator, cost_series.index, slack)
        cost_series = cost_series + slack_cost
        total_cost = float(cost_series.sum())
        extra = pd.concat([cost_series, slack_frame], axis=1)
        imp_mwh = float(slack_frame["slack:import"].sum())
        exp_mwh = float(slack_frame["slack:export"].sum())
        logger.info(
            "%s: slack import %.1f MWh (%.2f €), export %.1f MWh (%.2f €)",
            name_base,
            imp_mwh,
            imp_mwh * slack.import_price,
            exp_mwh,
            exp_mwh * slack.export_price,
        )
        slack_note = f"\nSlack: import {imp_mwh:,.0f} MWh, export {exp_mwh:,.0f} MWh"
    logger.info("%s: overall cost = %.2f", name_base, total_cost)
    annotation = f"Total cost: {total_cost:,.2f}{slack_note}"
    suffix = f" – {label}" if label else ""

    df = pd.DataFrame(Y_P, index=t_P, columns=[f"P:{aid}" for aid in labels_P])
    df.insert(0, f"target:{aggregator.context.aid}", target_series)
    df = df.join(extra)
    df.index = (df.index // 3600.0).astype(int)
    df.index.name = "time"
    df.to_csv(f"{name_base}-df.csv")
    with open(f"{name_base}-meta.json", "w", encoding="utf-8") as f:
        json.dump(_run_meta(world, aggregator, total_cost), f, indent=2, default=str)

    if group_map is None:
        group_map = {str(lbl): str(lbl) for lbl in labels_P}
    style_colors, nice_names = carrier_style(net) if net is not None else ({}, {})
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
        title=stacked_title,
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
        title=cost_title,
        annotation=annotation,
        write_to=f"{name_base}-cost.pdf",
    )

    return total_cost
