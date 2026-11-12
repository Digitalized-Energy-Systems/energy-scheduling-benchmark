"""Consensus + innovations for economic dispatch: the bus actor and its step parameters.

Plan: ``docs/consensus-plan.md`` (§7 I1–I10). The generic algorithm is DRO's
``ConsensusInnovationsAlgorithm``; here each bus agent answers a price vector λ (€/MWh per hour)
with the cheapest schedule of every unit at its bus (HKW15 eq. 19/29) and reports its innovation
= Σ unit injections − its own net load (KH12 eq. 24, HKW15 eq. 28). Net load = load − the bus's
pre-dispatched units at p_max (KH12 eq. 27 fixed infeed; S1 plan D3).
"""

from __future__ import annotations

import fcntl
import tomllib

import networkx as nx
import numpy as np
from distributed_resource_optimization import ConsensusActor, power_step
from mango_energy_environments import LOAD, STORAGE, Component, Network

from . import storage
from .datasets import RESOURCES
from .world import LEADER, SLACK, predispatched, topology_graph

PARAMETERS = RESOURCES / "tuned_parameters.toml"
#: Floor on λ (€/MWh) = the marginal cost of the pre-dispatched renewables: in a surplus hour
#: nobody can absorb, λ would fall without bound; at 0 renewables are the marginal unit, so 0 is
#: the clearing price (S1 plan D4, deviation C9; KH12/HKW15 have no bound).
LAM_MIN = 0.0


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


AGENT_SCALES = ("none", "capacity")


def agent_factors(network: Network, agent_scale: str) -> dict[str, float]:
    """Per-bus step factor s_j: each bus agent runs with s_j·α_i and s_j·β_i (S3,
    `docs/per-agent-steps-plan.md`, deviation C2b — KH12/HKW15 use one α, β for all agents).

    "capacity": s_j = median / Σ p_nom of bus j's dispatchable units (not pre-dispatched), so the
    median bus keeps the tuned values; buses without such units get the median factor. α and β
    are scaled together: scaling α alone moves the fixed point off the power balance (plan §2).
    "none": every bus 1 (the papers' update; the default and the paper route)."""
    if agent_scale not in AGENT_SCALES:
        raise ValueError(
            f"agent_scale must be one of {AGENT_SCALES}, got {agent_scale!r}"
        )
    buses = {c.bus for c in network.components.values()}
    if agent_scale == "none":
        return dict.fromkeys(buses, 1.0)
    cap: dict[str, float] = {}
    for c in network.components.values():
        if c.kind != LOAD and not predispatched(c.kind, c.cost[0]):
            cap[c.bus] = cap.get(c.bus, 0.0) + c.params["p_nom"]
    median = float(np.median(list(cap.values())))
    factors = {b: median / v for b, v in cap.items()}
    rest = float(np.median(list(factors.values())))
    return {b: factors.get(b, rest) for b in buses}


def max_sensitivity(network, agent_scale: str = "none", full=None) -> float:
    """Largest bus price sensitivity Σ 1/(2c₂) (MW per €/MWh) over the units that answer λ (not
    the pre-dispatched ones), times the bus's step factor s_j (:func:`agent_factors`): constant
    steps need s_j·α < 2 / it at that bus, so the α₀ / κ grids are factors of its inverse (S3
    Q-A4). The factors come from *full* (the network with storage, as in the runs; default
    *network*)."""
    factors = agent_factors(full or network, agent_scale)
    per_bus: dict[str, float] = {}
    for c in network.components.values():
        if c.kind != LOAD and not predispatched(c.kind, c.cost[0]):
            per_bus[c.bus] = per_bus.get(c.bus, 0.0) + 1 / (2 * c.cost[1])
    return max(factors[b] * v for b, v in per_bus.items())


def laplacian_max(network: Network, topology: str) -> float:
    """Largest Laplacian eigenvalue of the bus agents' topology (leader/slack leaves excluded: they
    do not iterate); the consensus term is stable only for β < 2 / it. 1 for a one-bus network."""
    graph = topology_graph(network, topology, slack=True)
    graph.remove_nodes_from([LEADER, SLACK])
    return float(nx.laplacian_spectrum(graph).max()) or 1.0


#: esb-tune grid (audit 2026-11-12, Q-A1 T5/T6; 32 candidates per algorithm, Q-A5): α₀ and κ are
#: factors of 1 / max_sensitivity, β₀ and β = κ·γ factors of 1 / laplacian_max.
SWEEP_ALPHA = (0.1, 0.3, 1.0, 1.5)
SWEEP_BETA_CONSTANT = (0.1, 0.3, 1.0, 1.5)
SWEEP_BETA_DECAYING = (0.3, 1.0)
#: decaying α exponents: the paper's 0.98 decays too fast for a 3000-iteration budget
#: (synthetic_10: −100 %, T5); both > 0.5, so Σα² < ∞ (KH12 Theorem 5).
SWEEP_ALPHA_EXP = (0.6, 0.98)
#: settings of every sweep run (not stored in the tuned set): stop rule off (I19), divergence
#: abort at the slack import price (T4)
SWEEP_RUN = {"eps_lambda": -1, "lam_abort": 4000.0}
#: esb-tune compares candidates per variant and notes the best of each
VARIANT = "steps"


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates (per-agent steps, S3 Q-A3) and a note on their scaling; *full* = the
    network with storage (per-agent factors)."""
    sens = max_sensitivity(network, "capacity", full)
    lmax = laplacian_max(network, topology)

    def r(v: float) -> float:
        return float(f"{v:.4g}")

    base = {"agent_scale": "capacity"}
    decaying = [
        base
        | {"steps": "decaying", "alpha0": r(a / sens), "beta0": r(g / lmax)}
        | {"alpha_exp": e}
        for a in SWEEP_ALPHA
        for g in SWEEP_BETA_DECAYING
        for e in SWEEP_ALPHA_EXP
    ]
    # β = κ·γ (KH12 eq. 43): γ = β / κ keeps the tuned-set format
    constant = [
        base
        | {
            "steps": "constant",
            "kappa": r(a / sens),
            "gamma": r(g / lmax / r(a / sens)),
        }
        for a in SWEEP_ALPHA
        for g in SWEEP_BETA_CONSTANT
    ]
    note = f"α grid × 1/{sens:.4g} MW per €/MWh, β grid × 1/λ_max(L) = 1/{lmax:.4g}"
    return decaying + constant, note


class BusActor(ConsensusActor):
    """All components at one bus, seen through MEE's copper-plate behaviour.

    *lam_abort*: sweep-only divergence abort (audit 2026-11-12, T4): once |λ| exceeds it the actor
    marks itself ``diverged`` and raises, which ends a lossless run at once."""

    def __init__(
        self, behavior, aid: str, eps_p: float, lam_abort: float = np.inf
    ) -> None:
        self.behavior, self.aid, self.lam_abort = behavior, aid, lam_abort
        self.diverged = False
        observe = behavior.observe
        self.load = 0.0  # net load: own load − own pre-dispatched units
        self.respond, self.tol, self.fixed = {}, {}, {}
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                self.load = self.load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            if predispatched(s["kind"], s["cost"][0]):
                self.fixed[cid] = p_max
                self.load = self.load - p_max
                continue
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
        if np.max(np.abs(lam)) > self.lam_abort:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: |λ| > {self.lam_abort} €/MWh (diverged)")
        self.prev, self.p = self.p, {cid: f(lam) for cid, f in self.respond.items()}
        return sum(self.p.values(), np.zeros_like(lam)) - self.load

    def converged(self) -> bool:
        return self.prev is not None and all(
            np.max(np.abs(self.p[cid] - self.prev[cid])) < tol
            for cid, tol in self.tol.items()
        )

    def finish(self) -> None:
        for cid, p in (self.fixed | (self.p or {})).items():
            self.behavior.act(self.aid, "set_schedule", p, cid)


def _generator(cid, cost, p_min, p_max):
    """argmin c₁p + c₂p² − λp on [p_min, p_max] = clip((λ − c₁)/(2c₂)) (HKW15 eq. 19)."""
    c1, c2 = cost
    if c2 <= 0:
        raise ValueError(
            f"{cid}: consensus needs c2 > 0 (prepare with a curvature band)"
        )
    return lambda lam: np.clip((lam - c1) / (2 * c2), p_min, p_max)
