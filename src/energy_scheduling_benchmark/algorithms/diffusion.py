"""Diffusion (de Azevedo et al. 2017) for economic dispatch: the bus actor, its parameters and the
benchmark wiring (``setup``; algorithm interface: :mod:`.algorithms`).

Plan: ``docs/diffusion-plan.md`` (S8a, D-215–D-226). The generic algorithm is DRO's
``DiffusionAlgorithm`` (§III: information sharing eq. 13, CTA optimization eq. 16, nudge
eqs. 20/21). Here every bus agent is one entry of w (D-215): it measures its net load (load − its
pre-dispatched units) for the information sharing pass, and its cost is the cheapest way to
produce w from its units and storage, with the paper's penalty (eqs. 22/23) as a slack (D-219):

    F(w) = min_u Σ cost(u) + ηα‖w − Σu‖²,   ∇F(w) = 2ηα(w − Σu*) = the bus's marginal cost,

one cached QP per bus (units in their bounds, storage P3, D-220); w only enters its linear term.
Buses without dispatchable units are uninformed: they only combine (§III-B).
"""

from __future__ import annotations

import cvxpy as cp
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
from mango_energy_environments import LOAD, STORAGE, Component, Network

from ..outputs import IterationTrace, Outcome
from ..problem import storage
from ..problem.qp import QP
from ..problem.rules import predispatched
from ..world import leader, longest_hops, settle_s, topology_graph

#: 2026-11-23 D-224: esb-tune grid, 8 μ × 2 ηα × 2 μ_IS = 32 (the paper: trial and error):
#: μ = g / the largest bus curvature, ηα = factor × the largest unit c₂
SWEEP_G = tuple(np.geomspace(0.01, 1.9, 8))
SWEEP_PENALTY = (1.0, 100.0)
SWEEP_MU_IS = (0.0, 0.5)
#: no stop rule to switch off (D-221); divergence abort at the grid import price (as consensus T4)
SWEEP_RUN = {"lam_abort": 4000.0}
#: esb-tune notes the best per penalty
VARIANT = "penalty"


def _dispatchable(network: Network) -> list[Component]:
    return [
        c
        for c in network.components.values()
        if c.kind != LOAD and not predispatched(c.kind, c.cost[0])
    ]


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates (μ, ηα, μ_IS) and a note on their scale."""
    units = _dispatchable(network)
    sens: dict[str, float] = {}
    for c in units:
        sens[c.bus] = sens.get(c.bus, 0.0) + 1 / (2 * c.cost[1])
    curvature = 1 / min(sens.values())  # the stiffest bus
    c2 = max(c.cost[1] for c in units)

    def r(v: float) -> float:
        return float(f"{v:.4g}")

    grid = [
        {"mu": r(g / curvature), "penalty": r(f * c2), "mu_is": m}
        for f in SWEEP_PENALTY
        for m in SWEEP_MU_IS
        for g in SWEEP_G
    ]
    return grid, (
        f"μ grid × 1/{curvature:.4g} €/MWh per MW (stiffest bus), "
        f"ηα ∈ {SWEEP_PENALTY} × max c₂ = {c2:.4g}"
    )


def weights(graph, aid: str) -> dict[str, float]:
    """de Azevedo eq. (12) as printed (D-216): a_ij = 1/max(n_i, n_j), n = connections (no +1;
    even rings have eigenvalue −1, plan §1 — 1/(1+max) is the noted alternative)."""
    n = graph.degree
    return {j: 1 / max(n[aid], n[j]) for j in graph[aid]}


class BusActor(DiffusionActor):
    """All components at one bus: net load for the information sharing, the penalised local
    dispatch for the gradient (plan §4)."""

    def __init__(
        self,
        behavior,
        aid: str,
        steps: int,
        penalty: float,
        lam_abort=np.inf,
        trace=None,
    ):
        self.behavior, self.aid, self.penalty, self.lam_abort = (
            behavior,
            aid,
            penalty,
            lam_abort,
        )
        self.trace, self.algorithm, self.diverged = trace, None, False
        observe = behavior.observe
        self.load, self.fixed, self.fixed_cost = 0.0, {}, 0.0
        self.units: dict[str, Component] = {}
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                self.load = self.load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            if predispatched(s["kind"], s["cost"][0]):
                self.fixed[cid] = p_max
                self.load = self.load - p_max
                c1, c2 = s["cost"]
                self.fixed_cost += float(np.sum(c1 * p_max + c2 * p_max**2))
                continue
            params = {k: v for k, v in s.items() if k not in ("kind", "bus", "cost")}
            self.units[cid] = Component(
                s["kind"], s["bus"], p_min, p_max, s["cost"], params
            )
        self.load = np.zeros(steps) + self.load  # a bus without loads: 0 per hour
        self.p: dict[str, np.ndarray] = {}
        self._qp = self._build() if self.units else None

    def _build(self) -> QP:
        steps = len(self.load)
        cost, supply, self._cols = 0, 0, []  # (cid, columns, sign) per variable
        cons, var = [], {}
        for cid, c in self.units.items():
            if c.kind == STORAGE:
                p_c, p_d = cp.Variable(steps), cp.Variable(steps)
                cons += storage.constraints(c, p_c, p_d)
                cost += c.cost[0] * cp.sum(p_d) + c.cost[1] * cp.sum_squares(p_d - p_c)
                supply += p_d - p_c
                var[cid] = [(p_d, 1.0), (p_c, -1.0)]
            else:
                p = cp.Variable(steps)
                cons += [p >= c.p_min, p <= c.p_max]
                cost += c.cost[0] * cp.sum(p) + c.cost[1] * cp.sum_squares(p)
                supply += p
                var[cid] = [(p, 1.0)]
        # w = 0: the w-dependent part −2ηα wᵀΣu is added to q per solve
        qp = QP(
            cp.Problem(cp.Minimize(cost + self.penalty * cp.sum_squares(supply)), cons),
            self.aid,
        )
        self._cols = [
            (cid, [(qp.col[v.id], sign) for v, sign in vs]) for cid, vs in var.items()
        ]
        return qp

    def measure(self, periods: int) -> np.ndarray:
        return self.load

    def gradient(self, w: np.ndarray) -> np.ndarray:
        k = self.algorithm.k + 1 if self.algorithm is not None else 0
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e12:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: estimate not finite (diverged)")
        if self._qp is None:  # uninformed: no cost, no gradient (§III-B)
            if self.trace is not None:
                self.trace.add(k, -self.load, self.fixed_cost, None)
            return np.zeros_like(w)
        steps = len(w)
        q = self._qp.q0.copy()
        for _, cols in self._cols:
            for i, sign in cols:
                q[i : i + steps] -= sign * 2 * self.penalty * w
        try:
            x = self._qp.solve(q)
        except (
            RuntimeError
        ):  # the solver gives up at extreme w (eq. 21, D-226): a divergence
            self.diverged = True
            raise
        self.p = {
            cid: sum(sign * x[i : i + steps] for i, sign in cols)
            for cid, cols in self._cols
        }
        supply = sum(self.p.values())
        g = 2 * self.penalty * (w - supply)
        if self.trace is not None:
            cost = self.fixed_cost + sum(
                storage.cost(self.units[cid], p)
                if self.units[cid].kind == STORAGE
                else float(
                    np.sum(self.units[cid].cost[0] * p + self.units[cid].cost[1] * p**2)
                )
                for cid, p in self.p.items()
            )
            self.trace.add(k, supply - self.load, cost, g)
        if np.max(np.abs(g)) > self.lam_abort:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: |∇F| > {self.lam_abort} €/MWh (diverged)")
        return g

    def finish(self) -> None:
        for cid, p in (self.fixed | self.p).items():
            self.behavior.act(self.aid, "set_schedule", p, cid)


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, :mod:`.algorithms`): bus agents as peers on the
    topology with eq. (12) weights, the first one also hosts the coordinator (start, IS → optimization
    pass, max_iter); the grid exchange takes no part (C5); returns
    ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s
    graph = topology_graph(network, args.topology)
    active_buses = {c.bus for c in _dispatchable(network)}

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        aids = [
            a.aid for a in buses
        ]  # agent ids are the bus names; one entry each (D-215)
        addresses = {a.aid: a.addr for a in agents.values()}
        active = [i for i, b in enumerate(aids) if b in active_buses]
        lossy = (
            args.loss_rate > 0
        )  # round deadline t0 + (k+1)·2·delay_s (as consensus C8)
        coordinator = DiffusionCoordinator(
            [a.addr for a in buses],
            np.zeros(steps_n),
            settle_s=settle_s(routes, aids, host.aid, delay_s),
            start_latency=longest_hops(routes, aids, host.aid) * delay_s
            if lossy
            else 0.0,
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
        for i, agent in enumerate(buses):
            actor = BusActor(
                behavior,
                agent.aid,
                steps_n,
                params["penalty"],
                params["lam_abort"],
                trace,
            )
            algorithm = DiffusionAlgorithm(
                lambda alg, carrier: alg.actor.finish(),
                actor,
                {addresses[j]: a for j, a in weights(graph, agent.aid).items()},
                own=i,
                active=active,
                entries=len(buses),
                step=params["mu"],  # one μ for all (D-218)
                step_is=params["mu_is"],
                eps_is=params["eps_is"],
                max_iter=params["max_iter"],
                round_timeout=2 * delay_s if lossy else None,
            )
            actor.algorithm = algorithm  # trace rows at the run's iteration count
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
