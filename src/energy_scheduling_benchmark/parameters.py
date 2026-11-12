"""Algorithm parameters from ``resources/tuned_parameters.toml`` (one section per algorithm).

``[<algorithm>]`` = defaults (= the paper values), ``[<algorithm>.<network>.<topology>.<route>]``
= a tuned set per route: ``manual`` (hand-tuned, with a note) or ``sweep`` (written by esb-tune).
Generic for every algorithm since audit 2026-11-12 Q-A4 (was consensus-only).
"""

from __future__ import annotations

import fcntl
import tomllib

from .datasets import RESOURCES

PARAMETERS = RESOURCES / "tuned_parameters.toml"
TUNINGS = ("paper", "manual", "sweep")


def has_parameters(algorithm: str) -> bool:
    """Whether *algorithm* has a ``[<algorithm>]`` section (central and dummy have none)."""
    return algorithm in _read()


def parameters(
    algorithm: str,
    network: str,
    topology: str,
    tuning: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """``[<algorithm>]`` defaults, then the *tuning* route's set, then *overrides* (``--set``).

    ``tuning=None``: "manual" if that set exists, else "paper".
    """
    table = _read()[algorithm]
    defaults = {k: v for k, v in table.items() if not isinstance(v, dict)}
    routes = table.get(network, {}).get(topology, {})
    if tuning is None:
        tuning = "manual" if "manual" in routes else "paper"
    if tuning not in TUNINGS:
        raise ValueError(f"tuning must be one of {TUNINGS}, got {tuning!r}")
    if tuning != "paper" and tuning not in routes:
        raise ValueError(
            f"no {tuning!r} {algorithm} parameters for {network}/{topology} in {PARAMETERS}"
        )
    unknown = set(overrides or {}) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {algorithm} parameters {sorted(unknown)}")
    own = routes.get(tuning, {}) if tuning != "paper" else {}
    return defaults | own | (overrides or {}) | {"tuning": tuning}


def _read() -> dict:
    with open(PARAMETERS) as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # esb-tune may be rewriting it in parallel
        return tomllib.loads(f.read())
