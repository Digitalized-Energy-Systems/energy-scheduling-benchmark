"""Consensus + innovations for economic dispatch: the bus actor, its step parameters and the
benchmark wiring (``setup``; algorithm interface: ``run.ALGORITHMS``, audit 2026-11-12 Q-A4).

Plan: ``docs/consensus-plan.md`` (§7 I1–I10). The generic algorithm is DRO's
``ConsensusInnovationsAlgorithm``; here each bus agent answers a price vector λ (€/MWh per hour)
with the cheapest schedule of every unit at its bus (HKW15 eq. 19/29) and reports its innovation
= Σ unit injections − its own net load (KH12 eq. 24, HKW15 eq. 28). Net load = load − the bus's
pre-dispatched units at p_max (KH12 eq. 27 fixed infeed; S1 plan D3).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from distributed_resource_optimization import (
    ConsensusActor,
    ConsensusCoordinator,
    ConsensusInnovationsAlgorithm,
    power_step,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
)
from mango_energy_environments import LOAD, STORAGE, Component, Network

from . import storage
from .outputs import Outcome
from .problem import predispatched
from .world import LEADER, longest_hops, settle_s, topology_graph

#: Floor on λ (€/MWh) = the marginal cost of the pre-dispatched renewables: in a surplus hour
#: nobody can absorb, λ would fall without bound; at 0 renewables are the marginal unit, so 0 is
#: the clearing price (S1 plan D4, deviation C9; KH12/HKW15 have no bound).
LAM_MIN = 0.0


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
    """Largest Laplacian eigenvalue of the bus agents' topology (the leader excluded: it does not
    iterate); the consensus term is stable only for β < 2 / it. 1 for a one-bus network."""
    graph = topology_graph(network, topology)
    graph.remove_node(LEADER)
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
#: abort at the grid import price (T4)
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


class IterationTrace:
    """Per-iteration convergence trace of one run (``trace.csv``; audit 2026-11-12 Q-A3, decided
    by Jonas: common columns + the algorithm's, recorded in the ESB actors, always on).

    Iteration k = the state at λ after k updates (k = 0: the start λ⁰). Every bus actor adds its
    mismatch, λ and current cost; a row is kept only once all *n* actors reached it.
    ``sim_time_s`` = simulated time at which the last actor got there (communication only: mango
    has no computation-time model)."""

    def __init__(self, n: int, now) -> None:
        self.n, self.now = n, now
        # k -> [count, Σ mismatch, Σ λ, min λ, max λ, Σ cost, latest time]
        self.rows: dict[int, list] = {}

    def add(self, k: int, lam: np.ndarray, mismatch: np.ndarray, cost: float) -> None:
        row = self.rows.get(k)
        if row is None:
            self.rows[k] = [
                1,
                mismatch.copy(),
                lam.copy(),
                lam.copy(),
                lam.copy(),
                cost,
                self.now(),
            ]
            return
        row[0] += 1
        row[1] += mismatch
        row[2] += lam
        np.minimum(row[3], lam, out=row[3])
        np.maximum(row[4], lam, out=row[4])
        row[5] += cost
        row[6] = max(row[6], self.now())

    def frame(self) -> pd.DataFrame:
        """iteration, sim_time_s, imbalance_mwh (Σ_h |Σ_j mismatch|), cost (€, units without the
        grid exchange), lambda_mean, lambda_spread (max_h of max_j λ − min_j λ)."""
        return pd.DataFrame(
            [
                {
                    "iteration": k,
                    "sim_time_s": t,
                    "imbalance_mwh": float(np.abs(mis).sum()),
                    "cost": cost,
                    "lambda_mean": float(lam_sum.mean() / self.n),
                    "lambda_spread": float((hi - lo).max()),
                }
                for k, (count, mis, lam_sum, lo, hi, cost, t) in sorted(
                    self.rows.items()
                )
                if count == self.n
            ]
        )


class BusActor(ConsensusActor):
    """All components at one bus, seen through MEE's copper-plate behaviour.

    *lam_abort*: sweep-only divergence abort (audit 2026-11-12, T4): once |λ| exceeds it the actor
    marks itself ``diverged`` and raises, which ends a lossless run at once. *trace*: an
    :class:`IterationTrace` the actor feeds once per iteration."""

    def __init__(
        self,
        behavior,
        aid: str,
        eps_p: float,
        lam_abort: float = np.inf,
        trace: IterationTrace | None = None,
    ) -> None:
        self.behavior, self.aid, self.lam_abort = behavior, aid, lam_abort
        self.diverged = False
        self.trace, self.k = trace, 0
        observe = behavior.observe
        self.load = 0.0  # net load: own load − own pre-dispatched units
        self.respond, self.tol, self.fixed, self.cost = {}, {}, {}, {}
        self.fixed_cost = 0.0  # the pre-dispatched units' cost (constant)
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                self.load = self.load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            c1, c2 = s["cost"]
            if predispatched(s["kind"], c1):
                self.fixed[cid] = p_max
                self.load = self.load - p_max
                self.fixed_cost += float(np.sum(c1 * p_max + c2 * p_max**2))
                continue
            self.tol[cid] = eps_p * s["p_nom"]
            if s["kind"] == STORAGE:
                params = {
                    k: v for k, v in s.items() if k not in ("kind", "bus", "cost")
                }
                c = Component(s["kind"], s["bus"], p_min, p_max, s["cost"], params)
                self.respond[cid] = storage.response(c)
                self.cost[cid] = lambda p, c=c: storage.cost(c, p)
            else:
                self.respond[cid] = _generator(cid, s["cost"], p_min, p_max)
                self.cost[cid] = lambda p, c1=c1, c2=c2: float(
                    np.sum(c1 * p + c2 * p**2)
                )
        self.p: dict | None = None
        self.prev: dict | None = None

    def innovation(self, lam: np.ndarray) -> np.ndarray:
        if np.max(np.abs(lam)) > self.lam_abort:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: |λ| > {self.lam_abort} €/MWh (diverged)")
        self.prev, self.p = self.p, {cid: f(lam) for cid, f in self.respond.items()}
        mismatch = sum(self.p.values(), np.zeros_like(lam)) - self.load
        if self.trace is not None:
            cost = self.fixed_cost + sum(self.cost[cid](p) for cid, p in self.p.items())
            self.trace.add(self.k, lam, mismatch, cost)
        self.k += 1
        return mismatch

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


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, ``run.ALGORITHMS``): returns
    ``(behavior, agents, routes) -> outcome`` that adds the roles — bus agents + leader; the grid
    exchange takes no part (plan Q6) — and ``outcome() -> Outcome``.

    *full*: the network with storage — the per-agent factors come from it also in a
    ``--no-storage`` run, as the one tuned set (tuned with storage) serves both (review 2026-11-11:
    without storage the factors moved up to 14× on PyPSA-Eur)."""
    steps_n, delay_s = len(network.time_index), args.delay_s

    def add_roles(behavior, agents, routes):
        alpha, beta = steps(params)
        factors = agent_factors(full, params["agent_scale"])
        buses = [a for node, a in agents.items() if node != LEADER]
        aids, leader = [a.aid for a in buses], agents[LEADER].aid
        # lossy links (S5): a round ends for everyone at t0 + (k+1)·2·delay_s (Q-L1); t0 after the
        # slowest start message. Lossless runs keep waiting for every λ (unchanged baseline).
        lossy = args.loss_rate > 0
        coordinator = ConsensusCoordinator(
            [a.addr for a in buses],
            np.zeros(steps_n),
            settle_s=settle_s(routes, aids, leader, delay_s),
            start_latency=longest_hops(routes, aids, leader) * delay_s
            if lossy
            else 0.0,
        )
        # full addresses: the agents may live in several (real) containers (audit Q-A8)
        addresses = {a.aid: a.addr for a in agents.values()}
        agents[LEADER].add_role(
            CoordinatorRole(coordinator, routes=routes[leader], addresses=addresses)
        )
        algorithms = []
        # every agent's clock is the world's simulated clock (audit Q-A3)
        trace = IterationTrace(
            len(buses), lambda: agents[LEADER].context.current_timestamp
        )
        for agent in buses:
            actor = BusActor(
                behavior, agent.aid, params["eps_p"], params["lam_abort"], trace
            )
            s = factors[agent.aid]  # agent ids are the bus names (world.build_world)
            algorithm = ConsensusInnovationsAlgorithm(
                lambda alg, carrier: alg.actor.finish(),
                actor,
                lambda i, s=s: s * alpha(i),
                lambda i, s=s: s * beta(i),
                params["eps_lambda"],
                params["max_iter"],
                LAM_MIN,
                round_timeout=2 * delay_s if lossy else None,
            )
            algorithms.append(algorithm)
            agent.add_role(
                DistributedOptimizationRole(algorithm, routes=routes[agent.aid])
            )

        def outcome() -> Outcome:
            r = coordinator.result
            # KH12 §IV: how often a round ended by deadline and how many λ were missing (Q-L5)
            counts = {
                "rounds_by_deadline": sum(a.rounds_by_deadline for a in algorithms),
                "missing_edges": sum(a.missing_edges for a in algorithms),
            }
            if r is not None:
                # sim time of the coordinator's stop = time to solution (Q-A3)
                return Outcome(
                    r.stop_reason, r.iterations, counts, r.time, trace.frame()
                )
            # sweep abort (audit T4), else a run that never stopped
            stop = (
                "diverged" if any(a.actor.diverged for a in algorithms) else "stalled"
            )
            return Outcome(stop, None, counts, None, trace.frame())

        return outcome

    return add_roles
