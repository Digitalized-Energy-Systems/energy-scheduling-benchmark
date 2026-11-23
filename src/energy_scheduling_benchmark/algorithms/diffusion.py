"""Diffusion (adapt-then-combine) for economic dispatch: the bus actor, its parameters and the
benchmark wiring (``setup``; algorithm interface: :mod:`.algorithms`).

Plan: ``docs/diffusion-plan.md`` (S8a, D-203–D-214). The generic algorithm is DRO's
``DiffusionAlgorithm`` (Ces et al. 2025 §3.3, eqs. 26/27; Yuan et al. 2019 Part I eqs. 6/7); here
each bus agent answers λ with its units' schedules — the consensus bus actor — and ∇J_i = its
local mismatch P_i(λ) − D_i (eq. 25 read locally with the descent sign, deviation DF1). Unit
limits by Ces's flag loop (§4, Fig. 1): generators answer unclipped, a pass end pins every (unit,
hour) past a limit at that limit, and a new pass starts (D-206, D-212).
"""

from __future__ import annotations

import numpy as np
from distributed_resource_optimization import (
    DiffusionActor,
    DiffusionAlgorithm,
    DiffusionCoordinator,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
)
from mango_energy_environments import STORAGE, Network

from ..outputs import IterationTrace, Outcome
from ..world import leader, longest_hops, settle_s, topology_graph
from . import consensus

#: 2026-11-23 D-209: esb-tune grid (replaces Ces's GA, §3.5): 32 ε = g / the largest
#: capacity-scaled bus sensitivity, each bus stable for s_j·ε·sens_j < 2
SWEEP_G = tuple(np.geomspace(0.01, 1.9, 32))
#: 2026-11-23 D-214: pass ends (and their flag checks) stay on in sweeps, only the final stop is
#: off; divergence abort at the grid import price (as consensus T4)
SWEEP_RUN = {"final_stop": False, "lam_abort": 4000.0}
#: esb-tune notes the best per variant: one variant (capacity-scaled ε)
VARIANT = "agent_scale"


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates ε (per-agent factors from *full*, D-204) and a note on their scale."""
    sens = consensus.max_sensitivity(network, "capacity", full)
    grid = [
        {"agent_scale": "capacity", "eps_step": float(f"{g / sens:.4g}")}
        for g in SWEEP_G
    ]
    return grid, f"ε grid × 1/{sens:.4g} MW per €/MWh"


def weights(graph, aid: str) -> dict[str, float]:
    """Ces eq. (19) row of *aid*: w_ij = 2/(n_i + n_j + 1) per neighbour j (D-203, as printed; on
    grid topologies w_ii = 1 − Σ_j w_ij can be negative, plan §1)."""
    n = graph.degree
    return {j: 2 / (n[aid] + n[j] + 1) for j in graph[aid]}


class BusActor(consensus.BusActor, DiffusionActor):
    """The consensus bus actor with unclipped generators, pinning and Ces's start (plan §4)."""

    def __init__(self, behavior, aid: str, lam_abort=np.inf, trace=None) -> None:
        super().__init__(behavior, aid, 0.0, lam_abort, trace)
        observe = behavior.observe
        c1s, self.limits, self.pinned = [], {}, {}
        for cid in self.respond:
            s = observe(aid, "statics", cid)
            c1, c2 = s["cost"]
            c1s.append(c1)
            if s["kind"] == STORAGE:
                continue  # its horizon response stays bounded (Q-D5, D-212)
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            self.limits[cid] = (p_min, p_max)
            self.pinned[cid] = None  # pinned value per hour, NaN = free
            self.respond[cid] = lambda lam, cid=cid, c1=c1, c2=c2: self._unit(
                cid, (lam - c1) / (2 * c2)
            )
        # 2026-11-23 D-205: Ces eq. (23) with P⁰ = 0 → the bus's cheapest marginal cost at P = 0
        self.lam0 = float(np.min(c1s)) if c1s else 0.0

    def _unit(self, cid: str, p: np.ndarray) -> np.ndarray:
        pin = self.pinned[cid]
        return p if pin is None else np.where(np.isnan(pin), p, pin)

    def gradient(self, lam: np.ndarray) -> np.ndarray:
        return self.innovation(lam)

    def start(self, lam0: np.ndarray) -> np.ndarray:
        return np.full_like(lam0, self.lam0)

    def _past(self, cid: str) -> tuple[np.ndarray, np.ndarray]:
        """Hours where the free unit is below p_min / above p_max (Ces §4 flags)."""
        p_min, p_max = self.limits[cid]
        p, pin = self.p[cid], self.pinned[cid]
        free = True if pin is None else np.isnan(pin)
        return free & (p < p_min), free & (p > p_max)

    def flags(self) -> bool:
        return self.p is not None and any(
            np.any(lo | hi) for lo, hi in map(self._past, self.limits)
        )

    def restart(self) -> None:
        if self.p is None:
            return
        for cid in self.limits:  # pinned stays pinned (Ces §4, D-212)
            lo, hi = self._past(cid)
            p_min, p_max = np.broadcast_to(self.limits[cid], (2, len(self.p[cid])))
            pin = self.pinned[cid]
            pin = np.full(len(lo), np.nan) if pin is None else pin
            pin[lo], pin[hi] = p_min[lo], p_max[hi]
            self.pinned[cid] = pin


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, :mod:`.algorithms`): bus agents as peers on the
    topology with Ces eq. (19) weights, the first one also hosts the coordinator (start, passes,
    stop); the grid exchange takes no part (C5); returns ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s
    graph = topology_graph(network, args.topology)

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        aids = [a.aid for a in buses]
        addresses = {a.aid: a.addr for a in agents.values()}
        factors = consensus.agent_factors(full, params["agent_scale"])  # D-204
        lossy = args.loss_rate > 0  # round deadline t0 + (k+1)·2·delay_s (as consensus C8)
        coordinator = DiffusionCoordinator(
            [a.addr for a in buses],
            np.zeros(steps_n),
            settle_s=settle_s(routes, aids, host.aid, delay_s),
            start_latency=longest_hops(routes, aids, host.aid) * delay_s
            if lossy
            else 0.0,
            final_stop=params["final_stop"],
        )
        host.add_role(
            CoordinatorRole(
                coordinator,
                include_self=True,
                routes=routes[host.aid],
                addresses=addresses,
            )
        )
        trace = IterationTrace(len(buses), lambda: host.context.current_timestamp)
        algorithms = []
        for agent in buses:
            row = weights(graph, agent.aid)  # agent ids are the bus names
            algorithm = DiffusionAlgorithm(
                lambda alg, carrier: alg.actor.finish(),
                BusActor(behavior, agent.aid, params["lam_abort"], trace),
                factors[agent.aid] * params["eps_step"],
                {addresses[j]: w for j, w in row.items()},
                params["eps_lambda"],
                params["max_iter"],
                round_timeout=2 * delay_s if lossy else None,
            )
            algorithms.append(algorithm)
            agent.add_role(
                DistributedOptimizationRole(algorithm, routes=routes[agent.aid])
            )

        def outcome() -> Outcome:
            r = coordinator.result
            counts = {
                "rounds_by_deadline": sum(a.rounds_by_deadline for a in algorithms),
                "missing_edges": sum(a.missing_edges for a in algorithms),
            }
            if r is not None:
                return Outcome(r.stop_reason, r.iterations, counts, r.time, trace.frame())
            stop = (
                "diverged" if any(a.actor.diverged for a in algorithms) else "stalled"
            )
            return Outcome(stop, None, counts, None, trace.frame())

        return outcome

    return add_roles
