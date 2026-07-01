"""Backwards-compatible re-exports — loaders now live in :mod:`~energy_scheduling_benchmark.networks`."""

from dataclasses import replace as _dataclass_replace

from energy_scheduling_benchmark.networks import (
    ScenarioData as TestNetwork,  # alias retained for older call-sites
)
from energy_scheduling_benchmark.networks import (
    ScenarioData,
    build_toy_network,
    load_scenario,
)

__all__ = ["TestNetwork", "build_toy_network", "load_scenario", "_clip_scenario"]


# Deprecated name — kept so existing imports keep working.
build_test_network = build_toy_network


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
