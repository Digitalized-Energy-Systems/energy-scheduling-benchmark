"""energy-scheduling-benchmark
===============================

Multi-agent energy scheduling benchmark scenarios built on top of
`mango-agents <https://github.com/OFFIS-DAI/mango>`_ and PyPSA.

This package is the Python re-implementation of
``EnergySchedulingBenchmark.jl``.  It exposes:

* :class:`PyPSABehavior` — a mango simulation environment behavior backed
  by a PyPSA network (replaces the Julia ``PowerSystemsBehavior``).
* :func:`load_scenario` and friends — load PyPSA built-in examples,
  NetCDF/HDF5/Excel files, CSV folders, or IEEE PYPOWER cases into a
  :class:`ScenarioData` bundle ready for the benchmark.
* :func:`solve_central_dispatch` — Pyomo LP economic dispatch used by
  the central-dispatch scenario.
* :func:`stacked_area`, :func:`visualize_results` — matplotlib plotting
  utilities (replace CairoMakie).
* Runnable scenarios in :mod:`energy_scheduling_benchmark.scenarios`.
"""

from mango_energy_environments import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PowerUpdateInfo,
    PyPSABehavior,
)

from .dispatch import solve_central_dispatch
from .networks import (
    ScenarioData,
    available_examples,
    build_toy_network,
    extract_timeseries,
    load_example,
    load_network,
    load_pypower_case,
    load_scenario,
)
from .plotting import stacked_area, visualize_results

__all__ = [
    "PyPSABehavior",
    "ComponentRef",
    "PowerUpdateInfo",
    "THERMAL",
    "RENEWABLE",
    "LOAD",
    "STORAGE",
    "solve_central_dispatch",
    "stacked_area",
    "visualize_results",
    "ScenarioData",
    "available_examples",
    "build_toy_network",
    "extract_timeseries",
    "load_example",
    "load_network",
    "load_pypower_case",
    "load_scenario",
]
