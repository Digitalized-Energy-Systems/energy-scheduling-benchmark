"""DEED-ADMM for economic dispatch: the bus participant, its parameters and the benchmark wiring
(``setup``; algorithm interface: :mod:`.algorithms`).

Plan: ``docs/deed-admm-plan.md`` (S6b). The generic algorithm is DRO's ``DEEDADMMAlgorithm``
(Zhu et al. 2025, Algorithm 1 lines 3–4, 12–13); here each bus agent is one energy hub and runs
lines 5–11 and 14 (:class:`BusParticipant`), reduced to electricity (Q-Z1 A): x = its dispatchable
units, y = its draw from the network, z = (charge, discharge) per storage + a surplus sink (Q-Z2 A),
coupling Σ_units x − y, A y + B z = d with d = load − its pre-dispatched units at p_max.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
from distributed_resource_optimization import (
    ConsensusCoordinator,
    DEEDADMMAlgorithm,
    DEEDParticipant,
)
from distributed_resource_optimization.carrier.mango_carrier import (
    CoordinatorRole,
    DistributedOptimizationRole,
)
from mango_energy_environments import LOAD, STORAGE, THERMAL, Component, Network

from ..outputs import IterationTrace, Outcome
from ..problem import storage
from ..problem.rules import predispatched
from ..world import leader, longest_hops, settle_s
from .consensus import laplacian_max

#: esb-tune grid (2026-11-17 D-177; 32 candidates as D-136): β = g/λ_max(L) of the bus topology
#: (Q-Z3 A; every g < 2 keeps Assumption 1) × 8 γ log-spaced ×10^±3 around the median 2c₂ of
#: the dispatchable units (as admm's ρ)
SWEEP_G = (0.3, 0.6, 1.0, 1.5)
SWEEP_GAMMAS = 8
SWEEP_DECADES = 3
#: 2026-11-17 D-176 (A3, as admm D-140/D-142): the stop rule stays on in the sweeps, 10× tighter
#: than the runs; a window that met it ranks by iterations
#: 2026-11-17 D-178: budget 10 000 it. (= the runs'; synthetic_50 is still ≈ +9 % at 3000)
SWEEP_RUN = {"eps_abs": 1e-5, "eps_rel": 1e-4, "max_iter": 10_000}
RANK_BY_ITERATIONS = True
#: esb-tune notes the best γ per β
VARIANT = "beta"


def sweep_grid(
    network: Network, full: Network, topology: str
) -> tuple[list[dict], str]:
    """esb-tune candidates (β, γ) and a note on the grid's scaling."""
    lmax = laplacian_max(network, topology)
    curv = [
        2 * c.cost[1]
        for c in network.components.values()
        if c.kind == THERMAL and not predispatched(c.kind, c.cost[0]) and c.cost[1] > 0
    ]
    centre = float(np.median(curv)) if curv else 1.0
    gammas = np.geomspace(
        centre * 10**-SWEEP_DECADES, centre * 10**SWEEP_DECADES, SWEEP_GAMMAS
    )
    grid = [
        {"beta": float(f"{g / lmax:.4g}"), "gamma": float(f"{gam:.4g}")}
        for g in SWEEP_G
        for gam in gammas
    ]
    note = (
        f"β grid × 1/λ_max(L) = 1/{lmax:.4g}, "
        f"γ log grid ×10^±{SWEEP_DECADES} around median 2c₂ = {centre:.4g}"
    )
    return grid, note


class BusParticipant(DEEDParticipant):
    """All components at one bus, seen through MEE's copper-plate behaviour (plan §4).

    p = (units u_j, y, per storage (charge, discharge), sink s), each one value per hour. Lines
    8–10 are one equality-constrained quadratic in p (x and (y, z) do not interact, so solving
    them together equals the paper's separate closed forms) with a constant matrix, factorised
    once; line 11 projects M p + ν/γ onto the box [lo, hi] of M's rows (unit p_min…p_max,
    charge / discharge 0…p_nom, SOC 0…E_max with E_T ≥ E_0, sink 0…pre-dispatched infeed;
    X = I, Y = 0); line 14 updates ν. Start all zero (Q-E4 A, D-120)."""

    def __init__(
        self, behavior, aid: str, gamma: float, trace: IterationTrace | None = None
    ) -> None:
        self.behavior, self.aid, self.gamma, self.trace = behavior, aid, gamma, trace
        observe = behavior.observe
        T = len(behavior.network.time_index)
        self.T = T
        load, infeed = np.zeros(T), np.zeros(T)
        self.fixed: dict = {}
        self.fixed_cost = 0.0
        units, stores = [], []
        for cid in observe(aid, "components"):
            s = observe(aid, "statics", cid)
            if s["kind"] == LOAD:
                load = load + observe(aid, "demand", cid)
                continue
            p_min, p_max = observe(aid, "p_min", cid), observe(aid, "p_max", cid)
            c1, c2 = s["cost"]
            if predispatched(s["kind"], c1):
                self.fixed[cid] = p_max
                infeed = infeed + p_max
                self.fixed_cost += float(np.sum(c1 * p_max + c2 * p_max**2))
            elif s["kind"] == STORAGE:
                params = {
                    k: v for k, v in s.items() if k not in ("kind", "bus", "cost")
                }
                stores.append(
                    (
                        cid,
                        Component(s["kind"], s["bus"], p_min, p_max, s["cost"], params),
                    )
                )
            else:
                units.append((cid, c1, c2, p_min, p_max))
        self.d = load - infeed  # line 9: A y + B z = d, A = 1
        eye = np.eye(T)
        n = (len(units) + 1 + 2 * len(stores) + 1) * T  # u…, y, (ch, dis)…, s
        self.Q, self.c = np.zeros((n, n)), np.zeros(n)  # cost ½pᵀQp + cᵀp
        rows, lo, hi = [], [], []  # M p, p̂ bounds
        E = np.zeros((T, n))  # y − Σ (ch − dis) − s = d

        def block(i: int) -> slice:
            return slice(i * T, (i + 1) * T)

        def bound(M_rows: np.ndarray, low, high) -> None:
            rows.append(M_rows)
            lo.append(np.broadcast_to(low, T))
            hi.append(np.broadcast_to(high, T))

        def select(i: int) -> np.ndarray:
            r = np.zeros((T, n))
            r[:, block(i)] = eye
            return r

        self.units = {}
        for j, (cid, c1, c2, p_min, p_max) in enumerate(units):
            self.Q[block(j), block(j)] = 2 * c2 * eye
            self.c[block(j)] = c1
            bound(select(j), p_min, p_max)  # X = I
            self.units[cid] = block(j)
        self.x_cols = slice(0, len(units) * T)
        iy = len(units)
        self.y_cols = block(iy)
        E[:, block(iy)] = eye
        self.stores = {}
        for k, (cid, comp) in enumerate(stores):
            ic, id_ = iy + 1 + 2 * k, iy + 2 + 2 * k
            c1, c2 = comp.cost
            self.Q[block(ic), block(ic)] = self.Q[block(id_), block(id_)] = 2 * c2 * eye
            self.Q[block(ic), block(id_)] = self.Q[block(id_), block(ic)] = (
                -2 * c2 * eye
            )
            self.c[block(id_)] = c1  # ageing cost on discharge (P3b)
            E[:, block(ic)], E[:, block(id_)] = -eye, eye
            bound(select(ic), 0.0, -comp.p_min)
            bound(select(id_), 0.0, comp.p_max)
            eta_c = comp.params["efficiency_store"]
            eta_d = comp.params["efficiency_dispatch"]
            e0 = comp.params["state_of_charge_initial"]
            e_max = comp.params["p_nom"] * comp.params["max_hours"]
            soc = np.zeros((T, n))  # E_t − E_0 = Σ_{τ≤t} (η_c ch − dis/η_d)
            soc[:, block(ic)] = eta_c * np.tril(np.ones((T, T)))
            soc[:, block(id_)] = -np.tril(np.ones((T, T))) / eta_d
            low = np.full(T, -e0)
            low[-1] = 0.0  # E_T ≥ E_0 (P3)
            bound(soc, low, e_max - e0)
            self.stores[cid] = (comp, block(ic), block(id_))
        isink = iy + 1 + 2 * len(stores)
        E[:, block(isink)] = -eye
        bound(select(isink), 0.0, infeed)  # Q-Z2 A
        self.M = np.vstack(rows) if rows else np.zeros((0, n))
        self.lo, self.hi = np.concatenate(lo), np.concatenate(hi)
        # lines 8–10: min ½pᵀHp + gᵀp s.t. E p = d, H = Q + γ(SᵀS + ‖y‖² + MᵀM)
        S = np.zeros((T, n))
        S[:, self.x_cols] = np.tile(eye, len(units))
        Y = np.zeros((T, n))
        Y[:, self.y_cols] = eye
        self.S, self.Y = S, Y
        H = self.Q + gamma * (S.T @ S + Y.T @ Y + self.M.T @ self.M)
        self.kkt = la.lu_factor(np.block([[H, E.T], [E, np.zeros((T, T))]]))
        self.n = n
        self.p, self.p_hat, self.nu = (
            np.zeros(n),
            np.zeros(len(self.lo)),
            np.zeros(len(self.lo)),
        )
        self.dual = np.inf
        self.k = 0

    def coupling(self) -> np.ndarray:
        return self.S @ self.p - self.Y @ self.p

    def update(self, lam_t: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
        g = self.gamma
        x, y = self.S @ self.p, self.Y @ self.p
        # lines 5–7 as the linear term of the primal step
        lin = (
            self.c
            + self.S.T @ (lam_t - g * (x - xi_t / 2))
            + self.Y.T @ (-lam_t - g * (y + xi_t / 2))
            + self.M.T @ (self.nu - g * self.p_hat)
        )
        p = la.lu_solve(self.kkt, np.concatenate([-lin, self.d]))[
            : self.n
        ]  # lines 8–10
        Mp = self.M @ p
        p_hat = np.clip(Mp + self.nu / g, self.lo, self.hi)  # line 11
        self.nu = self.nu + g * (Mp - p_hat)  # line 14
        self.dual = g * np.linalg.norm(p_hat - self.p_hat)
        old, self.p, self.p_hat = self.coupling(), p, p_hat
        new = self.coupling()
        if self.trace is not None:
            # price = −λ̃ (λ multiplies supply − demand; ∂c/∂x = −λ)
            self.trace.add(self.k, new, self.cost(), -lam_t)
        self.k += 1
        return new - old

    def x_norm(self) -> float:
        return float(np.linalg.norm(self.S @ self.p))

    def converged(self, eps_abs: float, eps_rel: float) -> bool:
        tol = np.sqrt(self.p_hat.size) * eps_abs + eps_rel * np.linalg.norm(self.nu)
        return bool(self.dual <= tol)

    def schedules(self) -> dict[str, np.ndarray]:
        """The bus's schedules from the primal p (plan §4: x, z) + the pre-dispatched units."""
        out = {cid: self.p[cols] for cid, cols in self.units.items()}
        for cid, (_, ch, dis) in self.stores.items():
            out[cid] = self.p[dis] - self.p[ch]
        return self.fixed | out

    def cost(self) -> float:
        units = sum(
            float(self.c[c] @ self.p[c] + self.Q[c, c].diagonal() @ self.p[c] ** 2 / 2)
            for c in self.units.values()
        )
        stores = sum(
            storage.cost(comp, self.p[dis] - self.p[ch])
            for comp, ch, dis in self.stores.values()
        )
        return self.fixed_cost + units + stores

    def finish(self) -> None:
        for cid, p in self.schedules().items():
            self.behavior.act(self.aid, "set_schedule", p, cid)


def setup(params: dict, network: Network, full: Network, args):
    """Benchmark wiring (algorithm interface, :mod:`.algorithms`): bus agents as peers on the
    topology, the first one also hosts the coordinator (start / stop / flags, Q-Z4 A); the grid
    exchange takes no part (Q-E2 A); returns ``(behavior, agents, routes) -> outcome``."""
    steps_n, delay_s = len(network.time_index), args.delay_s

    def add_roles(behavior, agents, routes):
        buses, host = list(agents.values()), leader(agents)
        aids = [a.aid for a in buses]
        # lossy links (S5, Q-Z6 A = consensus C8): round k ends at t0 + (k+1)·2·delay_s
        lossy = args.loss_rate > 0
        coordinator = ConsensusCoordinator(
            [a.addr for a in buses],
            np.zeros(steps_n),  # λ_0 = 0 (Q-E4 A)
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
                addresses={a.aid: a.addr for a in agents.values()},
            )
        )
        trace = IterationTrace(len(buses), lambda: host.context.current_timestamp)
        algorithms = []
        for agent in buses:
            algorithm = DEEDADMMAlgorithm(
                lambda alg, carrier: alg.participant.finish(),
                BusParticipant(behavior, agent.aid, params["gamma"], trace),
                beta=params["beta"],
                gamma=params["gamma"],
                eps_abs=params["eps_abs"],
                eps_rel=params["eps_rel"],
                max_iter=params["max_iter"],
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
            if r is None:
                return Outcome("stalled", None, counts, None, trace.frame())
            return Outcome(r.stop_reason, r.iterations, counts, r.time, trace.frame())

        return outcome

    return add_roles
