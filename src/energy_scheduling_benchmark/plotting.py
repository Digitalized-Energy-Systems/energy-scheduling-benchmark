"""Plotting utilities.

Port of the Julia ``plotting.jl`` module from ``EnergySchedulingBenchmark.jl``.
CairoMakie is replaced by matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mango.simulation.world import SimulationWorld

__all__ = [
    "visualize_results",
    "stacked_area",
    "cost_over_time",
    "agent_recording_as_plottable",
]


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
        Length-*K* layer labels shown in the legend.
    target:
        Length-*N* reference curve drawn on top (black line labelled ``"Target"``).
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

    C = np.cumsum(Y_arr, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("Paired", max(k, 1))
    for j in range(k):
        lower = np.zeros(n) if j == 0 else C[:, j - 1]
        upper = C[:, j]
        ax.fill_between(
            time,
            lower,
            upper,
            label=labels[j],
            alpha=0.85,
            color=cmap(j),
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(write_to)
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
