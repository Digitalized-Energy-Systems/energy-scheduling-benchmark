"""Backwards-compatible re-exports — loaders now live in :mod:`~energy_scheduling_benchmark.networks`."""

from energy_scheduling_benchmark.networks import (
    ScenarioData as TestNetwork,  # alias retained for older call-sites
    build_toy_network,
    load_scenario,
)

__all__ = ["TestNetwork", "build_toy_network", "load_scenario"]


# Deprecated name — kept so existing imports keep working.
build_test_network = build_toy_network
