"""Diffusion (de Azevedo et al. 2017) for economic dispatch: the bus actor, its parameters and the
benchmark wiring (``setup``; algorithm interface: :mod:`.algorithms`).

Plan: ``docs/diffusion-plan.md`` (S8a, D-215–D-231). The generic algorithm is DRO's
``DiffusionAlgorithm`` (§III: information sharing eq. 13 with neighbour flags and restarts, CTA
optimization eq. 16, nudge eqs. 20/21). Here every bus agent is one entry of w (D-215): it measures
its net load (load − its pre-dispatched units) for the information sharing, and its cost is the
cheapest way to produce w from its units and storage (P3, D-220) with the paper's penalty
(eqs. 22/23) outside what the bus can produce (D-231):

    ∇F(w) = λ(w̄) + 2ηα(w − w̄),   w̄ = the bus total closest to w it can produce,

λ(w̄) = the marginal cost of the exact dispatch Σu = w̄ — two cached QPs per bus (a projection and
the dispatch; w only enters a linear term / right-hand side). Inside the limits w̄ = w and the
penalty is 0, as eq. (23). The schedule is the dispatch at the bus's own entry of its final
estimate w_i,k[i] (Fig. 10, D-227). Buses without dispatchable units only combine (§III-B).
"""

from __future__ import annotations

import cvxpy as cp
import networkx as nx
import numpy as np
from distributed_resource_optimization import (
    ConsensusCoordinator,
    DiffusionActor,
    DiffusionAlgorithm,
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
from ..world import leader, longest_hops, topology_graph

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


class BusActor(DiffusionActor):
    """All components at one bus: net load for the information sharing, the exact local dispatch
    with the eq. (23) penalty for the gradient (plan §4)."""

    def __init__(
        self,
        behavior,
        aid: str,
        steps: int,
        penalty: float,
        lam_abort=np.inf,
        trace=None,
    ):
        self.behavior, self.aid, self.penalty = behavior, aid, penalty
        self.lam_abort, self.trace = lam_abort, trace
        self.algorithm, self.diverged = None, False
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
        if self.units:
            self._build(steps)

    def _build(self, steps: int) -> None:
        """The projection QP (min ‖w − Σu‖², w in q) and the dispatch QP (min cost, Σu = w̄ as
        the right-hand side), both over the units' bounds and the storage P3 set."""
        cost, supply, cons, var = 0, 0, [], {}
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
        # projection at w = 0: the w-dependent part −2wᵀΣu is added to q per solve
        self._proj = QP(cp.Problem(cp.Minimize(cp.sum_squares(supply)), cons), self.aid)
        self._w_bar = cp.Parameter(steps, value=np.zeros(steps))
        self._disp = QP(
            cp.Problem(cp.Minimize(cost), [supply == self._w_bar, *cons]), self.aid
        )
        self._cols = {
            qp: [
                (cid, [(qp.col[v.id], sign) for v, sign in vs])
                for cid, vs in var.items()
            ]
            for qp in (self._proj, self._disp)
        }
        # the balance rows of b (λ = −z·coef there; checked by finite differences)
        db = self._disp.b_for({self._w_bar.id: np.ones(steps)}) - self._disp.b_for(
            {self._w_bar.id: np.zeros(steps)}
        )
        self._rows = np.nonzero(db)[0]
        self._coef = db[self._rows]

    def _schedules(self, qp: QP, x: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        return {
            cid: sum(sign * x[i : i + steps] for i, sign in cols)
            for cid, cols in self._cols[qp]
        }

    def _dispatch(self, w: np.ndarray):
        """(schedules, λ, w̄): the exact dispatch of the producible total w̄ closest to w."""
        steps = len(w)
        q = self._proj.q0.copy()
        for _, cols in self._cols[self._proj]:
            for i, sign in cols:
                q[i : i + steps] -= sign * 2 * w
        try:
            w_bar = sum(
                self._schedules(self._proj, self._proj.solve(q), steps).values()
            )
            b = self._disp.b_for({self._w_bar.id: w_bar})
            x = self._disp.solve(self._disp.q0, b=b)
        except (
            RuntimeError
        ):  # the solver gives up at extreme w (eq. 21, D-226): a divergence
            self.diverged = True
            raise
        lam = -self._disp.z[self._rows] * self._coef
        return self._schedules(self._disp, x, steps), lam, w_bar

    def measure(self, periods: int) -> np.ndarray:
        return self.load

    def gradient(self, w: np.ndarray) -> np.ndarray:
        k = self.algorithm.k + 1 if self.algorithm is not None else 0
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > 1e12:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: estimate not finite (diverged)")
        if not self.units:  # uninformed: no cost, no gradient (§III-B)
            if self.trace is not None:
                self.trace.add(k, -self.load, self.fixed_cost, None)
            return np.zeros_like(w)
        p, lam, w_bar = self._dispatch(w)
        # 2026-11-23 D-231: eq. (23) — penalty only outside what the bus can produce
        g = lam + 2 * self.penalty * (w - w_bar)
        if self.trace is not None:
            self.trace.add(k, sum(p.values()) - self.load, self._cost(p), g)
        if np.max(np.abs(g)) > self.lam_abort:
            self.diverged = True
            raise RuntimeError(f"{self.aid}: |∇F| > {self.lam_abort} €/MWh (diverged)")
        return g

    def _cost(self, p: dict) -> float:
        return self.fixed_cost + sum(
            storage.cost(self.units[cid], v)
            if self.units[cid].kind == STORAGE
            else float(
                np.sum(self.units[cid].cost[0] * v + self.units[cid].cost[1] * v**2)
            )
            for cid, v in p.items()
        )

    def finish(self) -> None:
        # 2026-11-23 D-227: the setpoint is the own entry of the final estimate (Fig. 10);
        # no estimate yet (the information sharing never converged): no dispatch
        alg = self.algorithm
        if self.units and alg is not None and alg.w is not None:
            self.p = self._dispatch(alg.w[alg.own])[0]
        for cid, p in (self.fixed | self.p).items():
            self.behavior.act(self.aid, "set_schedule", p, cid)


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, :mod:`.algorithms`): bus agents as peers on the
    topology (eq. 12 weights built from the degrees in the packets), the first one also hosts the
    coordinator (start, max_iter); the grid exchange takes no part (C5); returns
    ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s
    graph = topology_graph(network, args.topology)
    # 2026-11-23 D-228: d given at setup (the paper does not say how agents learn it)
    diameter = nx.diameter(graph) if len(graph) > 1 else 0
    active_buses = {c.bus for c in _dispatchable(network)}

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        aids = [
            a.aid for a in buses
        ]  # agent ids are the bus names; one entry each (D-215)
        active = [i for i, b in enumerate(aids) if b in active_buses]
        lossy = (
            args.loss_rate > 0
        )  # round deadline t0 + (k+1)·2·delay_s (as consensus C8)
        coordinator = ConsensusCoordinator(
            [a.addr for a in buses],
            np.zeros(steps_n),
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
                own=i,
                active=active,
                entries=len(buses),
                step=params["mu"],  # one μ for all (D-218)
                step_is=params["mu_is"],
                eps_is=params["eps_is"],
                diameter=diameter,
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
