"""Plotting utilities.

Port of the Julia ``plotting.jl`` module from ``EnergySchedulingBenchmark.jl``.
CairoMakie is replaced by matplotlib.

For scenario result output see :func:`stacked_area` (carrier-grouped stacked
power), :func:`per_unit_small_multiples` (per-component detail as a grid of
per-carrier subplots) and :func:`generation_vs_demand` (supply/demand balance
diagnostic). :func:`visualize_results` is retained for ad-hoc use but is no
longer wired into the scenarios.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mango.simulation.world import SimulationWorld

logger = logging.getLogger(__name__)

__all__ = [
    "visualize_results",
    "stacked_area",
    "per_unit_small_multiples",
    "generation_vs_demand",
    "cost_over_time",
    "agent_recording_as_plottable",
    "order_carriers",
    "resolve_carrier_colors",
    "CARRIER_ORDER",
    "FALLBACK_CARRIER_COLORS",
]

#: Carrier stacking order, bottom of the stack (index 0) to top.  Carriers not
#: listed here are appended in alphabetical order just before ``"other"``, which
#: always sorts last.  Roughly merit order: must-run / baseload thermal at the
#: bottom, variable renewables and storage near the top.
CARRIER_ORDER: tuple[str, ...] = (
    "nuclear",
    "lignite",
    "coal",
    "CCGT",
    "OCGT",
    "gas",
    "oil",
    "waste",
    "biomass",
    "geothermal",
    "hydro",
    "ror",
    "onwind",
    "offwind",
    "offwind-ac",
    "offwind-dc",
    "wind",
    "solar",
    "PHS",
    "hydro-storage",
    "battery",
    "storage",
    "other",
)

#: Fallback carrier colours, used when the PyPSA network carries no
#: ``n.carriers.color`` entry for a carrier (e.g. the toy network).  Keyed by
#: lower-cased carrier name.
FALLBACK_CARRIER_COLORS: dict[str, str] = {
    "nuclear": "#ff8c00",
    "lignite": "#826837",
    "coal": "#545454",
    "hard coal": "#545454",
    "ccgt": "#a85522",
    "ocgt": "#e0986c",
    "gas": "#e0986c",
    "oil": "#c9c9c9",
    "waste": "#e3d37d",
    "biomass": "#baa741",
    "geothermal": "#ba91b1",
    "hydro": "#298c81",
    "ror": "#3dbfb0",
    "run of river": "#3dbfb0",
    "wind": "#235ebc",
    "onwind": "#235ebc",
    "offwind": "#6895dd",
    "offwind-ac": "#6895dd",
    "offwind-dc": "#74c6f2",
    "solar": "#f9d002",
    "pv": "#f9d002",
    "phs": "#51dbcc",
    "hydro-storage": "#298c81",
    "battery": "#ace37f",
    "storage": "#999999",
    "load": "#111111",
    "other": "#cccccc",
}


def order_carriers(carriers: Iterable[str]) -> list[str]:
    """Return *carriers* sorted by :data:`CARRIER_ORDER`.

    Carriers absent from :data:`CARRIER_ORDER` are appended in alphabetical
    order, just ahead of ``"other"`` which always sorts last.
    """
    index = {name: i for i, name in enumerate(CARRIER_ORDER)}

    def key(carrier: str) -> tuple[int, str]:
        if carrier == "other":
            return (len(CARRIER_ORDER) + 1, "")
        if carrier in index:
            return (index[carrier], "")
        return (len(CARRIER_ORDER), carrier)  # unknown: alphabetical, before "other"

    return sorted(dict.fromkeys(carriers), key=key)


def resolve_carrier_colors(
    carriers: Sequence[str],
    carrier_style: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map each carrier to a hex colour.

    Resolution order per carrier: the network-supplied *carrier_style*
    (``{carrier: color}`` from ``n.carriers.color``), then
    :data:`FALLBACK_CARRIER_COLORS`, then a deterministic ``tab20`` slot keyed
    by the carrier's position in ``sorted(carriers)``.
    """
    import matplotlib.pyplot as plt

    style = carrier_style or {}
    cycle = plt.get_cmap("tab20")
    ordered = sorted(dict.fromkeys(carriers))
    out: dict[str, str] = {}
    for i, carrier in enumerate(ordered):
        if style.get(carrier):
            out[carrier] = style[carrier]
            continue
        fallback = FALLBACK_CARRIER_COLORS.get(str(carrier).strip().lower())
        if fallback:
            out[carrier] = fallback
            continue
        out[carrier] = _to_hex(cycle(i % 20))
    return out


def _to_hex(rgba: Any) -> str:
    import matplotlib.colors as mcolors

    return mcolors.to_hex(rgba)


def _aggregate_columns(
    Y: np.ndarray,
    labels: Sequence[str],
    groups: Mapping[str, str],
    order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Sum the columns of *Y* by ``groups[label]`` (missing label → ``"other"``).

    Negative values are clipped to zero first (storage charging is not
    generation — same convention as
    :func:`~energy_scheduling_benchmark.scenarios._common.compute_overall_cost`).
    All-zero groups are dropped.  Groups are returned ordered by *order* (a
    carrier list, default :func:`order_carriers` of the groups present).
    """
    Y_arr = np.maximum(np.asarray(Y, dtype=float), 0.0)
    n = Y_arr.shape[0]
    keys = [groups.get(str(label), "other") for label in labels]
    present = order_carriers(keys) if order is None else [g for g in order if g in keys]
    # any group that slipped through (e.g. order given but incomplete)
    for g in keys:
        if g not in present:
            present.append(g)

    cols: list[np.ndarray] = []
    kept: list[str] = []
    for g in present:
        idx = [i for i, k in enumerate(keys) if k == g]
        if not idx:
            continue
        col = Y_arr[:, idx].sum(axis=1)
        if not np.any(col):
            continue
        cols.append(col)
        kept.append(g)

    if not cols:
        return np.zeros((n, 0)), []
    return np.column_stack(cols), kept


def visualize_results(
    world: SimulationWorld,
    *,
    write_to: str = "observation.pdf",
    colormap: str = "Paired",
    annotation: str | None = None,
) -> Any:
    """Render a grid of all recordings in *world* to *write_to*.

    Thin wrapper around :func:`mango.simulation.visualization.plot_recordings`.
    When *annotation* is given (e.g. a total-cost summary), it's added as a
    figure-level suptitle and the figure is re-saved with it included.

    The figure is closed after saving (matplotlib retains open pyplot figures
    indefinitely, which leaks memory across batch runs); it is still returned
    for callers that want to inspect it.
    """
    import matplotlib.pyplot as plt
    from mango.simulation.visualization import plot_recordings

    if annotation is None:
        fig = plot_recordings(world, colormap=colormap, write_to=write_to)
    else:
        fig = plot_recordings(world, colormap=colormap, write_to=None)
        fig.suptitle(annotation, fontsize=9, y=0.995)
        fig.savefig(write_to)
    plt.close(fig)
    return fig


def _dynamic_width(n_points: int) -> float:
    """Figure width in inches, scaled to the number of x samples (clamped)."""
    return float(min(24.0, max(8.0, 0.05 * n_points)))


def stacked_area(
    time: Sequence[float],
    Y: np.ndarray,
    labels: Sequence[str],
    target: Sequence[float],
    *,
    xlabel: str = "Time",
    ylabel: str = "Value",
    title: str = "Stacked power",
    annotation: str | None = None,
    groups: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    write_to: str = "stacked.pdf",
) -> Any:
    """Draw a stacked-area plot of ``Y`` with a target overlay.

    Parameters
    ----------
    time:
        Length-*N* x-axis values.
    Y:
        ``(N, K)`` array.  Each column is stacked on top of the previous one.
    labels:
        Length-*K* per-column labels.
    target:
        Length-*N* reference curve drawn on top (black line labelled ``"Target"``).
    groups:
        Optional ``{label: group}`` map (e.g. component id → PyPSA carrier).
        When given, columns sharing a group are summed into a single band, so a
        network with hundreds of components collapses to a handful of bands.
        Negative values are clipped to zero before summing.
    colors:
        Optional ``{group: colour}`` map (used only with *groups*).
    order:
        Optional explicit bottom-to-top band order (used only with *groups*;
        defaults to :func:`order_carriers`).
    """
    import matplotlib.pyplot as plt

    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y_arr.shape}")
    n, k = Y_arr.shape
    if len(labels) != k:
        raise ValueError(f"len(labels)={len(labels)} doesn't match columns={k}")
    if len(time) != n:
        raise ValueError(f"len(time)={len(time)} doesn't match rows={n}")

    if groups is not None:
        Y_arr, plot_labels = _aggregate_columns(Y_arr, labels, groups, order)
        palette: list[Any] = [
            (colors or {}).get(group, "#cccccc") for group in plot_labels
        ]
    else:
        plot_labels = list(labels)
        cmap = plt.get_cmap("Paired", max(k, 1))
        palette = [cmap(j) for j in range(k)]

    k_plot = Y_arr.shape[1]
    C = np.cumsum(Y_arr, axis=1)

    fig, ax = plt.subplots(figsize=(_dynamic_width(n), 6))
    for j in range(k_plot):
        lower = np.zeros(n) if j == 0 else C[:, j - 1]
        upper = C[:, j]
        ax.fill_between(
            time,
            lower,
            upper,
            label=plot_labels[j],
            alpha=0.85,
            color=palette[j],
            edgecolor="none",
            step="post",
        )

    ax.plot(
        time,
        target,
        color="black",
        linewidth=2.0,
        label="Target",
        drawstyle="steps-post",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{annotation}" if annotation else title)
    ax.margins(x=0)

    # Legend outside the axes (right), top-of-stack first so it reads in the
    # same vertical order as the bands.
    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        target_h = handles[-1], leg_labels[-1]
        band_h = list(zip(handles[:-1], leg_labels[:-1]))[::-1]
        ordered = [target_h, *band_h]
        ncol = 2 if len(ordered) > 18 else 1
        ax.legend(
            [h for h, _ in ordered],
            [name for _, name in ordered],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize="small",
            ncol=ncol,
        )
    fig.savefig(write_to, bbox_inches="tight")
    plt.close(fig)
    return fig


def per_unit_small_multiples(
    time: Sequence[float],
    Y: np.ndarray,
    labels: Sequence[str],
    groups: Mapping[str, str],
    *,
    xlabel: str = "Hour",
    ylabel: str = "P in MW",
    title: str = "Per-unit power",
    annotation: str | None = None,
    carrier_names: Mapping[str, str] | None = None,
    write_to: str = "observation.pdf",
) -> Any:
    """Per-component detail as a grid of per-carrier subplots.

    One subplot per group in *groups* (e.g. PyPSA carrier); within each, a thin
    translucent line per component plus the group total as one opaque line.
    Scales to hundreds of components without an unreadable legend — the carrier
    is the subplot title and the component count is annotated in the corner.
    """
    import matplotlib.pyplot as plt

    Y_arr = np.asarray(Y, dtype=float)
    time_arr = np.asarray(time, dtype=float)
    names = carrier_names or {}

    if Y_arr.ndim != 2 or Y_arr.shape[0] == 0 or Y_arr.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no per-unit data recorded", ha="center", va="center")
        ax.set_title(f"{title}\n{annotation}" if annotation else title)
        fig.savefig(write_to, bbox_inches="tight")
        plt.close(fig)
        return fig

    keys = [groups.get(str(label), "other") for label in labels]
    present = order_carriers(keys)

    ncols = min(4, max(1, math.ceil(math.sqrt(len(present)))))
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.6 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = [ax for row in axes for ax in row]

    for ax, carrier in zip(axes_flat, present):
        idx = [i for i, key in enumerate(keys) if key == carrier]
        block = Y_arr[:, idx]
        for col in range(block.shape[1]):
            ax.plot(time_arr, block[:, col], lw=0.6, alpha=0.5, color="tab:blue")
        ax.plot(time_arr, block.sum(axis=1), lw=1.5, color="black")
        ax.set_title(names.get(carrier, carrier), fontsize="small")
        ax.text(
            0.98,
            0.94,
            f"n={len(idx)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="x-small",
            color="0.4",
        )
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(present) :]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)

    fig.suptitle(f"{title} — {annotation}" if annotation else title)
    fig.savefig(write_to, bbox_inches="tight")
    plt.close(fig)
    return fig


def generation_vs_demand(
    time: Sequence[float],
    Y: np.ndarray,
    target: Sequence[float],
    *,
    xlabel: str = "Hour",
    ylabel: str = "P in MW",
    title: str = "Generation vs demand",
    annotation: str | None = None,
    tol: float = 0.01,
    write_to: str = "balance.pdf",
) -> Any:
    """Supply/demand balance diagnostic.

    Top panel: total generation ``Σ max(P_i, 0)`` as a filled area against the
    *target* demand curve.  Bottom panel: the raw residual ``Σ P_i − target``
    (storage charging shows as a genuine dip) around a zero line, with a shaded
    ``±tol`` band scaled to peak demand.
    """
    import matplotlib.pyplot as plt

    Y_arr = np.asarray(Y, dtype=float)
    time_arr = np.asarray(time, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    if Y_arr.ndim != 2 or Y_arr.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data recorded", ha="center", va="center")
        ax.set_title(f"{title}\n{annotation}" if annotation else title)
        fig.savefig(write_to, bbox_inches="tight")
        plt.close(fig)
        return fig

    total_gen = np.maximum(Y_arr, 0.0).sum(axis=1)
    net_gen = Y_arr.sum(axis=1)
    m = min(len(time_arr), len(target_arr), len(total_gen))
    time_arr, target_arr = time_arr[:m], target_arr[:m]
    total_gen, net_gen = total_gen[:m], net_gen[:m]
    residual = net_gen - target_arr
    band = tol * float(np.max(np.abs(target_arr))) if m else 0.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(_dynamic_width(m), 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.fill_between(
        time_arr,
        0.0,
        total_gen,
        step="post",
        alpha=0.6,
        color="tab:blue",
        label="Generation",
    )
    ax_top.plot(
        time_arr,
        target_arr,
        color="black",
        lw=2.0,
        drawstyle="steps-post",
        label="Demand",
    )
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(f"{title}\n{annotation}" if annotation else title)
    ax_top.margins(x=0)
    ax_top.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize="small"
    )

    ax_bot.axhline(0.0, color="0.3", lw=1.0)
    if band > 0:
        ax_bot.axhspan(-band, band, color="0.85", zorder=0)
    ax_bot.plot(time_arr, residual, color="tab:red", lw=1.2, drawstyle="steps-post")
    ax_bot.set_ylabel("Gen − Demand")
    ax_bot.set_xlabel(xlabel)
    ax_bot.margins(x=0)

    fig.savefig(write_to, bbox_inches="tight")
    plt.close(fig)
    return fig


def cost_over_time(
    time: Sequence[float],
    cost: Sequence[float],
    *,
    xlabel: str = "Hour",
    ylabel: str = "Cost",
    title: str = "Cost per timestep",
    annotation: str | None = None,
    write_to: str = "cost.pdf",
) -> Any:
    """Draw a line plot of total cost per timestep.

    Intended for the ``per_step`` series returned by
    :func:`~energy_scheduling_benchmark.scenarios._common.compute_overall_cost`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, cost, color="tab:red", linewidth=1.5, drawstyle="steps-post")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{annotation}" if annotation else title)
    fig.tight_layout()
    fig.savefig(write_to)
    plt.close(fig)
    return fig


def agent_recording_as_plottable(
    world: SimulationWorld, key: str
) -> tuple[list[float], np.ndarray, list[str]]:
    """Convert an :class:`~mango.simulation.world.AgentsRecording` into arrays.

    Returns ``(time, Y, labels)`` where ``Y`` has shape ``(len(time), n_agents)``.
    When individual agent series have different lengths they are left-aligned
    and truncated to the shortest length.
    """
    rec = world.data_agent_collections.get(key)
    if rec is None:
        raise KeyError(f"No agent recording for key '{key}'")

    if not rec.timeseries:
        return list(rec.time), np.zeros((0, 0)), []

    min_len = min(len(v) for v in rec.timeseries.values())
    min_len = min(min_len, len(rec.time))

    labels = list(rec.timeseries.keys())
    Y = np.zeros((min_len, len(labels)))
    for j, aid in enumerate(labels):
        values = rec.timeseries[aid][:min_len]
        Y[:, j] = [_to_scalar(v) for v in values]

    return list(rec.time[:min_len]), Y, labels


def _to_scalar(value: Any) -> float:
    """Coerce a recording cell to a scalar."""
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else 0.0
    arr = np.asarray(value).ravel()
    return float(arr[0]) if arr.size else 0.0
