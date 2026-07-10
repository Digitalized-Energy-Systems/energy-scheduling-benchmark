"""PyPSA network loading utilities.

Users of this benchmark are *not* expected to hand-craft PyPSA networks.
This module offers a small set of loaders that cover the common cases:

* **Built-in PyPSA examples** — e.g. ``ac-dc-meshed``, ``storage-hvdc``,
  ``scigrid-de``.  Dispatched to ``pypsa.examples``.
* **File loading** — ``.nc`` (NetCDF), ``.h5`` (HDF5), ``.xlsx`` (Excel),
  and CSV folders.  Dispatched by path suffix.
* **PYPOWER IEEE cases** — ``case14``, ``case30``, ``case57``, ``case118``
  via ``pypsa.Network.import_from_pypower_ppc``.
* **A toy 5-bus network** — for tests and quick demos.

Every loader returns a :class:`ScenarioData` bundle (network + timeseries
+ start datetime).  Timeseries are either extracted from the network's
own ``_t`` DataFrames (``generators_t.p_max_pu`` / ``loads_t.p_set`` /
``storage_units_t.p_set``) or synthesised when the network has no time
dependency.

Typical use::

    from energy_scheduling_benchmark import PyPSABehavior, load_scenario

    scenario = load_scenario("ac-dc-meshed")
    behavior = PyPSABehavior.from_scenario(scenario)
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    ComponentRef,
    extract_timeseries,
)

if TYPE_CHECKING:  # pragma: no cover
    import pypsa

__all__ = [
    "ScenarioData",
    "available_examples",
    "extract_timeseries",
    "load_example",
    "load_network",
    "load_pypower_case",
    "load_scenario",
    "build_toy_network",
]


# ---------------------------------------------------------------------------
# PyPSA built-in examples
# ---------------------------------------------------------------------------


#: Short-name → attribute-name mapping for ``pypsa.examples``.
_EXAMPLE_MAP: dict[str, str] = {
    "ac-dc-meshed": "ac_dc_meshed",
    "storage-hvdc": "storage_hvdc",
    "scigrid-de": "scigrid_de",
    "model-energy": "model_energy",
}


def available_examples() -> list[str]:
    """Return the list of supported PyPSA example short names."""
    return sorted(_EXAMPLE_MAP)


def load_example(name: str, **kwargs: Any):
    """Load a built-in ``pypsa.examples.*`` network by short name.

    Parameters
    ----------
    name:
        One of :func:`available_examples`.  Both dash and underscore forms
        are accepted (``ac-dc-meshed`` or ``ac_dc_meshed``).
    **kwargs:
        Forwarded to the underlying ``pypsa.examples.*`` function
        (typically ``update`` / ``from_master``).
    """
    import pypsa

    normalised = name.replace("_", "-").lower()
    attr = _EXAMPLE_MAP.get(normalised, name.replace("-", "_"))
    examples_mod = getattr(pypsa, "examples", None)
    if examples_mod is None:
        raise RuntimeError(
            "The installed PyPSA does not expose `pypsa.examples`; "
            "upgrade PyPSA or load the network from a file instead."
        )
    fn = getattr(examples_mod, attr, None)
    if fn is None:
        raise ValueError(
            f"Unknown PyPSA example: {name!r}. "
            f"Available: {', '.join(available_examples())}"
        )
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Generic network loader (dispatches on source type)
# ---------------------------------------------------------------------------


def load_network(source: str | Path | pypsa.Network | Callable[[], Any]):
    """Return a :class:`pypsa.Network` for any supported *source*.

    * ``pypsa.Network`` instance → returned as-is.
    * ``callable`` → called with no args, expected to return a network.
    * ``str`` that matches an :func:`available_examples` name → loaded
      via :func:`load_example`.
    * ``str`` of the form ``case<N>`` (e.g. ``case14``, ``case118``) →
      loaded via :func:`load_pypower_case` (requires ``pypower``).
    * ``str | Path`` pointing at a file or directory → loaded via the
      appropriate PyPSA importer based on the suffix:

      * ``.nc``, ``.netcdf``     → :py:meth:`pypsa.Network.import_from_netcdf`
      * ``.h5``, ``.hdf5``       → :py:meth:`pypsa.Network.import_from_hdf5`
      * ``.xlsx``, ``.xls``      → :py:meth:`pypsa.Network.import_from_excel`
      * directory                → :py:meth:`pypsa.Network.import_from_csv_folder`
    """
    import pypsa

    if isinstance(source, pypsa.Network):
        return source
    if callable(source):
        return source()

    # String: example / PYPOWER-case names take priority over file paths.
    if isinstance(source, str):
        normalised = source.replace("_", "-").lower()
        if normalised in _EXAMPLE_MAP:
            return load_example(source)
        if re.fullmatch(r"case\d+", source.lower()):
            return load_pypower_case(source.lower())

    path = Path(source)
    if path.is_dir():
        net = pypsa.Network()
        net.import_from_csv_folder(str(path))
        return net

    suffix = path.suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        return pypsa.Network(str(path))
    if suffix in {".h5", ".hdf5"}:
        net = pypsa.Network()
        net.import_from_hdf5(str(path))
        return net
    if suffix in {".xlsx", ".xls"}:
        net = pypsa.Network()
        net.import_from_excel(str(path))
        return net

    raise ValueError(
        f"Unsupported network source: {source!r}. "
        f"Expected a pypsa.Network, an example name "
        f"({', '.join(available_examples())}), a PYPOWER case name "
        f"(case14, case30, …), or a .nc/.h5/.xlsx/CSV-folder path."
    )


# ---------------------------------------------------------------------------
# PYPOWER case loader (IEEE 14/30/57/118 bus standard test cases)
# ---------------------------------------------------------------------------


def load_pypower_case(case: str | int):
    """Load a PYPOWER IEEE standard test case into a PyPSA network.

    Parameters
    ----------
    case:
        Either a bus count (14, 30, 57, 118, 300) or a case name
        (``"case14"``, ``"case118"`` …).

    Requires the ``pypower`` package to be installed.
    """
    import pypsa

    try:
        import pypower.api as _ppa  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading PYPOWER cases requires the `pypower` package. "
            "Install it with `pip install pypower`."
        ) from exc

    name = f"case{case}" if isinstance(case, int) else case
    if not hasattr(_ppa, name):
        raise ValueError(f"Unknown PYPOWER case: {case!r}")

    ppc = getattr(_ppa, name)()
    net = pypsa.Network()
    net.import_from_pypower_ppc(ppc)
    net.name = name
    return net


# ---------------------------------------------------------------------------
# Scenario bundle (network + timeseries + start)
# ---------------------------------------------------------------------------


@dataclass
class ScenarioData:
    """Everything a scenario needs to spin up a simulation.

    Attributes
    ----------
    net:
        The PyPSA network.
    timeseries:
        Per-component timeseries keyed by :class:`ComponentRef`.
    start:
        Datetime corresponding to simulation clock ``0.0``.
    label:
        Human-readable scenario name (used for output filenames).
    """

    net: Any
    timeseries: dict[ComponentRef, pd.Series] = field(default_factory=dict)
    start: datetime | None = None
    label: str = ""


def load_scenario(
    source: str | Path | pypsa.Network | Callable[[], Any],
    *,
    timeseries: dict[ComponentRef, pd.Series] | None = None,
    renewable_carriers: frozenset[str] | set[str] | None = None,
    label: str | None = None,
) -> ScenarioData:
    """One-stop loader: returns :class:`ScenarioData` for any supported source.

    Parameters
    ----------
    source:
        Anything accepted by :func:`load_network` — example name, file path,
        PYPOWER case name (``case14``, ``case118`` …), callable,
        or existing :class:`pypsa.Network`.
    timeseries:
        Optional explicit timeseries that overrides automatic extraction.
    renewable_carriers:
        Override the default carrier vocabulary for renewable classification.
    label:
        Pretty name for output files (defaults to ``str(source)``).

    Example
    -------
    ::

        scenario = load_scenario("ac-dc-meshed")
        behavior = PyPSABehavior.from_scenario(scenario)
    """
    net = load_network(source)

    if timeseries is None:
        timeseries = extract_timeseries(net, renewable_carriers=renewable_carriers)

    start = _infer_start(net, timeseries)
    final_label = label if label is not None else _default_label(source)

    if getattr(net, "name", "") in ("", "Unnamed Network"):
        net.name = final_label

    return ScenarioData(net=net, timeseries=timeseries, start=start, label=final_label)


def _infer_start(net, timeseries: dict[ComponentRef, pd.Series]) -> datetime:
    """Use ``net.snapshots[0]`` when available, otherwise the earliest ts index."""
    snaps = getattr(net, "snapshots", None)
    if snaps is not None and len(snaps) > 0:
        first = snaps[0]
        return first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
    for series in timeseries.values():
        if not series.empty:
            first = series.index[0]
            return first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
    return datetime(2024, 1, 1)


def _default_label(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(str(source)).stem or str(source)
    return getattr(source, "__name__", type(source).__name__)


# ---------------------------------------------------------------------------
# Toy network (kept for tests & quick smoke-runs)
# ---------------------------------------------------------------------------


def build_toy_network(
    *, periods: int = 72, freq: str = "h", label: str = "toy-5bus"
) -> ScenarioData:
    """Hand-built 5-bus PyPSA network for tests & quick demos.

    2 thermal generators (``gas``, ``coal``), 1 wind generator, 2 loads,
    1 battery.  Renewable availability and load profiles are embedded as
    hourly timeseries.

    Prefer :func:`load_scenario` with a real PyPSA example for anything
    beyond smoke tests.
    """
    import pypsa

    net = pypsa.Network()
    net.name = label
    start = datetime(2024, 1, 1)
    snapshots = pd.date_range(start, periods=periods, freq=freq)
    net.set_snapshots(snapshots)

    for i in range(5):
        net.add("Bus", f"bus{i}", v_nom=20.0)

    net.add(
        "Generator",
        "thermal0",
        bus="bus0",
        carrier="gas",
        p_nom=100.0,
        p_min_pu=0.1,
        p_max_pu=1.0,
        marginal_cost=30.0,
        p_set=50.0,
    )
    net.add(
        "Generator",
        "thermal1",
        bus="bus1",
        carrier="coal",
        p_nom=60.0,
        p_min_pu=0.0833,
        p_max_pu=1.0,
        marginal_cost=50.0,
        p_set=30.0,
    )
    net.add(
        "Generator",
        "wind0",
        bus="bus2",
        carrier="wind",
        p_nom=40.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
        p_set=20.0,
    )

    net.add("Load", "load0", bus="bus3", p_set=45.0)
    net.add("Load", "load1", bus="bus4", p_set=30.0)

    net.add(
        "StorageUnit",
        "batt0",
        bus="bus0",
        p_nom=20.0,
        max_hours=4.0,
        marginal_cost=0.0,
        p_set=0.0,
    )

    wind_values = [0.3 + 0.4 * abs((i % 24 - 12) / 12) for i in range(periods)]

    def _load_profile(base: float, amp: float, phase: float) -> list[float]:
        return [
            base + amp * math.cos(2 * math.pi * ((i % 24) / 24.0) + phase)
            for i in range(periods)
        ]

    timeseries = {
        ComponentRef(RENEWABLE, "wind0"): pd.Series(wind_values, index=snapshots),
        ComponentRef(LOAD, "load0"): pd.Series(
            _load_profile(45.0, 10.0, 0.0), index=snapshots
        ),
        ComponentRef(LOAD, "load1"): pd.Series(
            _load_profile(30.0, 6.0, math.pi / 3), index=snapshots
        ),
    }

    return ScenarioData(net=net, timeseries=timeseries, start=start, label=label)
