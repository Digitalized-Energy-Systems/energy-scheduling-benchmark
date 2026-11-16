"""Exchange ADMM for economic dispatch: the bus participant, its parameters and the benchmark
wiring (``setup``; algorithm interface: ``run.ALGORITHMS``).

Plan: ``docs/admm-exchange-plan.md`` (S6a). The generic algorithm is DRO's exchange ADMM
(Boyd et al. 2011 §7.3.2, ``create_exchange_admm_coordinator``) with the surplus slack
Σ x_i ≥ 0 (Q-E1, D-117) on the leader (first bus agent); every bus agent answers the coordinator's
correction v with its net injection x_i = Σ unit injections − its net load (load − its
pre-dispatched units at p_max, as consensus), minimising its units' cost + ρ/2 ||x_i + v||².
"""

from __future__ import annotations

import clarabel
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones
from distributed_resource_optimization import (
    ADMMAnswer,
    ADMMMessage,
    Coordinator,
    DistributedAlgorithm,
    create_admm_start,
    create_exchange_admm_coordinator,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
    OptimizationFinishedMessage,
)
from mango_energy_environments import LOAD, STORAGE, THERMAL, Component, Network

from . import storage
from .outputs import IterationTrace, Outcome
from .problem import predispatched
from .world import leader, longest_hops

#: esb-tune: 32 fixed ρ (= consensus' count, Q-A5; 2026-11-16 D-136 corrects D-119's 128),
#: log-spaced over six decades around the median 2c₂ of the dispatchable thermal units (€/MWh per
#: MW, the bus problem's own curvature; Q-E6 sweep route)
SWEEP_CANDIDATES = 32
SWEEP_DECADES = 3  # each side of the centre
#: settings of every sweep run: stop rule off (as consensus I19)
SWEEP_RUN = {"eps_abs": -1.0, "eps_rel": 0.0}
VARIANT = "adaptive"


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates: fixed ρ (route ``sweep``) and a note on the grid's centre."""
    curv = [
        2 * c.cost[1]
        for c in network.components.values()
        if c.kind == THERMAL and not predispatched(c.kind, c.cost[0]) and c.cost[1] > 0
    ]
    centre = float(np.median(curv)) if curv else 1.0
    rhos = np.geomspace(
        centre * 10**-SWEEP_DECADES, centre * 10**SWEEP_DECADES, SWEEP_CANDIDATES
    )
    grid = [{"adaptive": False, "rho": float(f"{r:.4g}")} for r in rhos]
    return grid, f"ρ log grid ×10^±{SWEEP_DECADES} around median 2c₂ = {centre:.4g}"


class BusParticipant(DistributedAlgorithm):
    """All components at one bus, seen through MEE's copper-plate behaviour; answers
    ``ADMMMessage(v, ρ)`` with argmin cost(p) + ρ/2 ||x + v||², x = Σ p − net load.

    The model (units within p_min … p_max at c₁p + c₂p², storage per :mod:`.storage`, cost as
    central dispatch) is canonicalised once; per message only q changes, a new ρ (adaptive route)
    rebuilds the solver — Clarabel is called directly, as in :func:`.storage.response`."""

    def __init__(self, behavior, aid: str, trace: IterationTrace | None = None) -> None:
        self.behavior, self.aid, self.trace = behavior, aid, trace
        observe = behavior.observe
        steps = len(behavior.network.time_index)
        load = np.zeros(steps)
        self.fixed: dict = {}
        self.fixed_cost = 0.0
        self.cost: dict = {}
        exprs, cost, cons = {}, 0, []
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                load = load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            c1, c2 = s["cost"]
            if predispatched(s["kind"], c1):
                self.fixed[cid] = p_max
                load = load - p_max
                self.fixed_cost += float(np.sum(c1 * p_max + c2 * p_max**2))
                continue
            if s["kind"] == STORAGE:
                params = {
                    k: v for k, v in s.items() if k not in ("kind", "bus", "cost")
                }
                c = Component(s["kind"], s["bus"], p_min, p_max, s["cost"], params)
                p_c, p_d = cp.Variable(steps), cp.Variable(steps)
                cons += storage.constraints(c, p_c, p_d)
                exprs[cid] = [(p_d, 1.0), (p_c, -1.0)]
                cost += c1 * cp.sum(p_d) + c2 * cp.sum_squares(p_d - p_c)
                self.cost[cid] = lambda p, c=c: storage.cost(c, p)
            else:
                p = cp.Variable(steps)
                cons += [p >= p_min, p <= p_max]
                exprs[cid] = [(p, 1.0)]
                cost += c1 * cp.sum(p) + c2 * cp.sum_squares(p)
                self.cost[cid] = lambda p, c1=c1, c2=c2: float(
                    np.sum(c1 * p + c2 * p**2)
                )
        x = cp.Variable(steps)
        net = sum((sign * v for parts in exprs.values() for v, sign in parts), 0)
        problem = cp.Problem(cp.Minimize(cost), [*cons, x == net - load])
        data, _, _ = problem.get_problem_data(cp.CLARABEL)
        col = data["param_prob"].var_id_to_col
        self._cols = {
            cid: [(col[v.id], sign) for v, sign in parts]
            for cid, parts in exprs.items()
        }
        self._x = col[x.id]
        n = data["c"].size
        P = data.get("P")  # absent when the bus's cost is linear
        self._P = sp.csc_matrix((n, n)) if P is None else sp.triu(P).tocsc()
        diag = np.zeros(n)
        diag[self._x : self._x + steps] = 1.0
        self._prox = sp.diags(diag).tocsc()
        self._q0, self._A, self._b = data["c"], data["A"], data["b"]
        self._cones = dims_to_solver_cones(data["dims"])
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._solver, self._rho, self._steps = None, None, steps
        self.p: dict | None = None

    def _solve(self, v: np.ndarray, rho: float) -> np.ndarray:
        q = self._q0.copy()
        q[self._x : self._x + self._steps] += rho * v  # −ρ wᵀx with w = −v
        if self._solver is None or rho != self._rho:
            self._rho = rho
            P = (self._P + rho * self._prox).tocsc()
            self._solver = clarabel.DefaultSolver(
                P, q, self._A, self._b, self._cones, self._settings
            )
        else:
            self._solver.update(q=q)
        solution = self._solver.solve()
        if solution.status != clarabel.SolverStatus.Solved:
            raise RuntimeError(f"{self.aid}: ADMM bus QP {solution.status}")
        z = np.asarray(solution.x)
        s = self._steps
        self.p = {
            cid: sum(sign * z[i : i + s] for i, sign in cols)
            for cid, cols in self._cols.items()
        }
        return z[self._x : self._x + s]

    async def on_exchange_message(self, carrier, message_data, meta) -> None:
        if isinstance(message_data, OptimizationFinishedMessage):
            for cid, p in (self.fixed | (self.p or {})).items():
                self.behavior.act(self.aid, "set_schedule", p, cid)
            return
        if not isinstance(message_data, ADMMMessage):
            return  # a reply to the coordinator on this agent (Q-B7), not for the bus
        v, rho = np.asarray(message_data.v, dtype=float), message_data.rho
        x = self._solve(v, rho)
        if self.trace is not None:
            cost = self.fixed_cost + sum(self.cost[cid](p) for cid, p in self.p.items())
            # the bus's marginal cost ∂f/∂x = ρ(w − x), w = −v: λ at the fixed point
            self.trace.add(message_data.k - 1, x, cost, -rho * (v + x))
        carrier.reply_to_other(ADMMAnswer(x=x), meta)


class _Kickoff(Coordinator):
    """run.py starts every leader with an empty start message; this one starts ADMM with
    *start* (``ADMMStart``: the window length)."""

    def __init__(self, coordinator, start) -> None:
        self.coordinator, self.start = coordinator, start

    async def start_optimization(self, carrier, message_data, meta):
        return await self.coordinator.start_optimization(carrier, self.start, meta)


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, ``run.ALGORITHMS``): the coordinator on the first
    bus agent (leader), a :class:`BusParticipant` on every bus agent; the grid exchange takes no
    part (Q-E2 A, D-118); returns ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        lossy = args.loss_rate > 0
        # ≥ 1: the leader's messages to itself take one delay too (a one-bus network has 0 hops)
        hops = max(longest_hops(routes, [a.aid for a in buses], host.aid), 1)
        coordinator = create_exchange_admm_coordinator(
            surplus=True,
            rho=params["rho"],
            max_iters=params["max_iter"],
            abs_tol=params["eps_abs"],
            rel_tol=params["eps_rel"],
            mu=params["mu"] if params["adaptive"] else None,
            tau=params["tau"],
            # 2026-11-16 Q-E11 (D-135): 4·max_hops·delay_s after the round's scatter (D-137),
            # twice the longest round trip
            round_timeout=4 * hops * delay_s if lossy else None,
        )
        role = CoordinatorRole(
            _Kickoff(coordinator, create_admm_start(None, steps_n)),
            include_self=True,
            routes=routes[host.aid],
            addresses={a.aid: a.addr for a in buses},
        )
        host.add_role(role)
        trace = IterationTrace(len(buses), lambda: host.context.current_timestamp)
        for agent in buses:
            participant = BusParticipant(behavior, agent.aid, trace)
            agent.add_role(
                DistributedOptimizationRole(participant, routes=routes[agent.aid])
            )

        def outcome() -> Outcome:
            counts = {"missing_replies": coordinator.missing_replies}
            if role._done_future is None or not role._done_future.done():
                return Outcome("stalled", None, counts, None, trace.frame())
            stop = "converged" if coordinator.converged else "max_iter"
            return Outcome(
                stop,
                coordinator.iterations,
                counts,
                coordinator.stop_time,
                trace.frame(),
            )

        return outcome

    return add_roles
