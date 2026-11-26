"""Exact diffusion (Yuan et al. 2019, Alg. 1) for economic dispatch: the bus actor, its parameters
and the benchmark wiring (``setup``; algorithm interface: :mod:`.algorithms`).

Plan: ``docs/exact-diffusion-plan.md`` (S8b, D-248–D-256). The generic algorithm is DRO's
``ExactDiffusionAlgorithm`` (adapt, correct, combine with Ā = (I + A)/2, Hastings weights). Here it
runs on the economic-dispatch dual (Ces et al. 2025 §3.4 form): every bus agent holds a price
vector λ (€/MWh per hour) and J_k(λ) = −min_P [C_k(P) − λP] − λD_k is its (negated) dual over the
horizon, so by Danskin

    ∇J_k(λ) = P_k(λ) − D_k

= the consensus bus actor's innovation (its units' cheapest answer to λ, clipped to their limits,
storage = P3 over the horizon): λ rises while demand exceeds supply. Set up like S8a's diffusion for comparison: Hastings
weights = S8a's, one μ, start 0, no stop rule. The schedule is the answer to the final λ.
"""

from __future__ import annotations

import numpy as np
from distributed_resource_optimization import (
    ConsensusCoordinator,
    ExactDiffusionActor,
    ExactDiffusionAlgorithm,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
)
from mango_energy_environments import Network

from ..outputs import IterationTrace, Outcome
from ..world import leader, longest_hops
from . import consensus

#: 2026-11-26 D-254: esb-tune grid, 32 μ = g / the largest bus sensitivity Σ 1/(2c₂) (MW per
#: €/MWh), g log-spaced 0.01…1.9 (Yuan Part II Lemma 6: stable for μσ² < 2)
SWEEP_G = tuple(np.geomspace(0.01, 1.9, 32))
#: no stop rule to switch off (D-255) and no divergence abort (as diffusion, D-247)
SWEEP_RUN = {}
#: one variant (esb-tune: candidates[i].get(None))
VARIANT = None


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates (μ) and a note on their scale."""
    sens = consensus.max_sensitivity(network)

    def r(v: float) -> float:
        return float(f"{v:.4g}")

    return [{"mu": r(g / sens)} for g in SWEEP_G], f"μ grid × 1/{sens:.4g} MW per €/MWh"


class BusActor(ExactDiffusionActor):
    """All components at one bus: the consensus bus actor seen as J_k (plan §4)."""

    def __init__(self, behavior, aid: str, steps: int, trace: IterationTrace) -> None:
        self.bus = consensus.BusActor(behavior, aid, eps_p=0.0, trace=trace)
        self.steps, self.diverged = steps, False
        self.algorithm: ExactDiffusionAlgorithm | None = None

    def start(self) -> np.ndarray:
        return np.zeros(self.steps)  # 2026-11-26 D-252: λ⁰ = 0 (Yuan: w_{−1} arbitrary)

    def gradient(self, lam: np.ndarray) -> np.ndarray:
        # 2026-11-26 D-249/D-257: local ∇J_k = P_k(λ) − D_k, not Ces eq. (25)'s global mismatch
        try:
            if not np.all(np.isfinite(lam)):
                raise RuntimeError(f"{self.bus.aid}: λ not finite (diverged)")
            return self.bus.innovation(lam)
        except RuntimeError:  # incl. a storage QP that gives up at extreme λ
            self.diverged = True
            raise

    def finish(self) -> None:
        # the units' answer to the final λ (the last gradient was at the round before)
        lam = self.algorithm.w
        self.bus.p = {cid: f(lam) for cid, f in self.bus.respond.items()}
        self.bus.finish()


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, :mod:`.algorithms`): bus agents as peers on the
    topology (Hastings weights from the degrees in the packets), the first one also hosts the
    coordinator (start, ``max_iter``); the grid exchange takes no part (C5); returns
    ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        aids = [a.aid for a in buses]
        lossy = (
            args.loss_rate > 0
        )  # round deadline t0 + (k+1)·2·delay_s (as consensus C8)
        coordinator = ConsensusCoordinator(
            [a.addr for a in buses],
            (),  # no λ⁰ here: the actor's start (D-252) sets the periods
            start_latency=longest_hops(routes, aids, host.aid) * delay_s
            if lossy
            else 0.0,
        )
        host.add_role(
            CoordinatorRole(
                coordinator,
                include_self=True,
                routes=routes[host.aid],
                addresses={a.aid: a.addr for a in agents.values()},
            )
        )
        trace = IterationTrace(len(buses), lambda: host.context.current_timestamp)
        algorithms = []
        for agent in buses:
            actor = BusActor(behavior, agent.aid, steps_n, trace)
            # 2026-11-26 D-251: one μ for all (Hastings: uniform Perron vector, Yuan eq. 13)
            algorithm = ExactDiffusionAlgorithm(
                lambda alg, carrier: alg.actor.finish(),
                actor,
                step=params["mu"],
                max_iter=params["max_iter"],
                round_timeout=2 * delay_s if lossy else None,
            )
            actor.algorithm = algorithm
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
                return Outcome(
                    r.stop_reason, r.iterations, counts, r.time, trace.frame()
                )
            stop = (
                "diverged" if any(a.actor.diverged for a in algorithms) else "stalled"
            )
            return Outcome(stop, None, counts, None, trace.frame())

        return outcome

    return add_roles
