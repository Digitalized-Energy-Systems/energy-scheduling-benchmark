"""Consensus + innovations for economic dispatch: the bus actor and its step parameters.

Plan: ``docs/consensus-plan.md`` (§7 I1–I10). The generic algorithm is DRO's
``ConsensusInnovationsAlgorithm``; here each bus agent answers a price vector λ (€/MWh per hour)
with the cheapest schedule of every unit at its bus (HKW15 eq. 19/29) and reports its innovation
= Σ unit injections − its own load (KH12 eq. 24, HKW15 eq. 28).
"""

from __future__ import annotations

import fcntl
import tomllib

import numpy as np
from distributed_resource_optimization import ConsensusActor, power_step
from mango_energy_environments import LOAD, STORAGE, Component

from . import storage
from .datasets import RESOURCES

PARAMETERS = RESOURCES / "tuned_parameters.toml"


TUNINGS = ("paper", "manual", "sweep")


def parameters(
    network: str,
    topology: str,
    tuning: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """``[consensus]`` defaults (= the paper values), then the *tuning* route's set
    ``[consensus.<network>.<topology>.<tuning>]``, then *overrides*.

    ``tuning=None``: "manual" if that set exists, else "paper".
    """
    with open(PARAMETERS) as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # esb-tune may be rewriting it in parallel
        table = tomllib.loads(f.read())["consensus"]
    defaults = {k: v for k, v in table.items() if not isinstance(v, dict)}
    routes = table.get(network, {}).get(topology, {})
    if tuning is None:
        tuning = "manual" if "manual" in routes else "paper"
    if tuning not in TUNINGS:
        raise ValueError(f"tuning must be one of {TUNINGS}, got {tuning!r}")
    if tuning != "paper" and tuning not in routes:
        raise ValueError(
            f"no {tuning!r} parameters for {network}/{topology} in {PARAMETERS}"
        )
    unknown = set(overrides or {}) - set(defaults)
    if unknown:
        raise ValueError(f"unknown consensus parameters {sorted(unknown)}")
    own = routes.get(tuning, {}) if tuning != "paper" else {}
    return defaults | own | (overrides or {}) | {"tuning": tuning}


def steps(p: dict):
    """``(alpha, beta)`` schedules: decaying (HKW15 eq. 32 form) or constant (KH12 eq. 43)."""
    if p["steps"] == "decaying":
        return power_step(p["alpha0"], p["alpha_exp"]), power_step(
            p["beta0"], p["beta_exp"]
        )
    if p["steps"] == "constant":
        return power_step(p["kappa"], 0), power_step(p["kappa"] * p["gamma"], 0)
    raise ValueError(f"steps must be 'decaying' or 'constant', got {p['steps']!r}")


class BusActor(ConsensusActor):
    """All components at one bus, seen through MEE's copper-plate behaviour."""

    def __init__(self, behavior, aid: str, eps_p: float) -> None:
        self.behavior, self.aid = behavior, aid
        observe = behavior.observe
        self.load = 0.0
        self.respond, self.tol = {}, {}
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                self.load = self.load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            self.tol[cid] = eps_p * s["p_nom"]
            if s["kind"] == STORAGE:
                params = {
                    k: v for k, v in s.items() if k not in ("kind", "bus", "cost")
                }
                c = Component(s["kind"], s["bus"], p_min, p_max, s["cost"], params)
                self.respond[cid] = storage.response(c)
            else:
                self.respond[cid] = _generator(cid, s["cost"], p_min, p_max)
        self.p: dict | None = None
        self.prev: dict | None = None

    def innovation(self, lam: np.ndarray) -> np.ndarray:
        self.prev, self.p = self.p, {cid: f(lam) for cid, f in self.respond.items()}
        return sum(self.p.values(), np.zeros_like(lam)) - self.load

    def converged(self) -> bool:
        return self.prev is not None and all(
            np.max(np.abs(self.p[cid] - self.prev[cid])) < tol
            for cid, tol in self.tol.items()
        )

    def finish(self) -> None:
        for cid, p in (self.p or {}).items():
            self.behavior.act(self.aid, "set_schedule", p, cid)


def _generator(cid, cost, p_min, p_max):
    """argmin c₁p + c₂p² − λp on [p_min, p_max] = clip((λ − c₁)/(2c₂)) (HKW15 eq. 19)."""
    c1, c2 = cost
    if c2 <= 0:
        raise ValueError(
            f"{cid}: consensus needs c2 > 0 (prepare with a curvature band)"
        )
    return lambda lam: np.clip((lam - c1) / (2 * c2), p_min, p_max)
