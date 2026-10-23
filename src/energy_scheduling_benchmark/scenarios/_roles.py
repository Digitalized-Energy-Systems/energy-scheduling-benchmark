"""Messages and mango roles: load reporting, the leader (``PowerLoadAggregator``) and result collection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from distributed_resource_optimization.carrier.mango import (
    DistributedOptimizationRole,
    RoleLocalMessage,
)
from mango import Role, sender_addr

from energy_scheduling_benchmark import (
    PowerUpdateInfo,
)
from energy_scheduling_benchmark.dispatch import SlackSpec
from energy_scheduling_benchmark.scenarios._outputs import (
    compute_slack,
)

logger = logging.getLogger(__name__)


@dataclass
class PowerLoadInfo:
    """Current load power forwarded from a load agent to the aggregator."""

    power_load: float
    time: float


@dataclass
class OptimizationFinishedInfo:
    """A generator's finished schedule, sent to the leader.

    Sent when the generator's run is done, and again as the reply to a
    :class:`ScheduleRequest` (so a lost one can be recovered).
    """

    aid: str
    schedule: np.ndarray | None = None
    #: Rounds the unit's algorithm ran and how it stopped (for the meta sidecar).
    iterations: int | None = None
    converged: bool | None = None


class ScheduleRequest(RoleLocalMessage):
    """Leader → generator: send your finished schedule (answered by
    :class:`GeneratorResultRole`, never seen by the algorithm)."""


@dataclass
class KickOff(RoleLocalMessage):
    """Leader → the trigger generator: start the distributed run with
    *start_message* (acknowledged, so a lost kick-off is retried). *run*
    numbers the runs, so retries of one run start it once while a later run
    (e.g. FDGDM's outer storage rounds) starts again."""

    start_message: Any
    run: int = 1


@dataclass
class KickOffAck:
    """Reply to :class:`KickOff`."""

    aid: str


class PowerLoadMonitoring(Role):
    """Observes ``max_active_power`` and forwards it to *target* each timestep."""

    def __init__(self, behavior: Any, target: Any) -> None:
        super().__init__()
        self._behavior = behavior
        self._target = target

    def on_agent_event(self, event: Any) -> None:
        if not isinstance(event, PowerUpdateInfo):
            return
        power = self._behavior.observe(self.context.aid, "max_active_power")
        self.context.schedule_instant_message(
            PowerLoadInfo(power_load=float(power), time=self.context.current_timestamp),
            self._target,
        )


class GeneratorResultRole(Role):
    """Generator-side companion of the :class:`DistributedOptimizationRole`.

    Remembers the finished schedule (set by :func:`make_finish_callback`) and
    answers the leader's :class:`ScheduleRequest` with it, so a lost
    "finished" message can be recovered.  On :class:`KickOff` it starts the
    local algorithm once and acknowledges every request, so a lost kick-off is
    retried without ever restarting a run.
    """

    def __init__(self) -> None:
        super().__init__()
        #: The finished schedule, ``None`` while the run is in progress.
        self.schedule: np.ndarray | None = None
        self.iterations: int | None = None
        self.converged: bool | None = None
        self._started_run = 0

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._on_schedule_request, lambda c, m: isinstance(c, ScheduleRequest)
        )
        self.context.subscribe_message(
            self, self._on_kick_off, lambda c, m: isinstance(c, KickOff)
        )

    def _reply(self, content: Any, meta: dict) -> None:
        self.context.schedule_instant_task(self.context.reply_to(content, meta))

    def _on_schedule_request(self, content: ScheduleRequest, meta: dict) -> None:
        if self.schedule is None:
            return  # still running: the leader asks again later
        self._reply(
            OptimizationFinishedInfo(
                aid=self.context.aid,
                schedule=self.schedule,
                iterations=self.iterations,
                converged=self.converged,
            ),
            meta,
        )

    def _on_kick_off(self, content: KickOff, meta: dict) -> None:
        if content.run > self._started_run:
            self._started_run = content.run
            self.schedule = None  # a poll must not return the previous run's result
            self.context.get_role(DistributedOptimizationRole).start(
                content.start_message
            )
        self._reply(KickOffAck(aid=self.context.aid), meta)


def generator_agent(opt_role: DistributedOptimizationRole) -> Any:
    """A generator agent: the optimization role plus its :class:`GeneratorResultRole`."""
    from mango import agent_composed_of

    return agent_composed_of(opt_role, GeneratorResultRole())


class PowerLoadAggregator(Role):
    """Leader role: fires a distributed optimization run once, then dispatches
    the resulting schedule.

    Load agents stream ``PowerLoadInfo`` messages each hour. Once all loads
    report for a given timestep, the aggregator applies the pre-computed
    set-points for that hour. Timesteps that arrive before the optimization
    run is done are held in ``_pending`` and flushed immediately afterwards.

    Message loss (all optional; without loss none of it changes a run):

    * *load_timeout*: every hour is applied at the latest *load_timeout*
      seconds after it starts, even if load reports are missing (counted in
      :attr:`hours_with_missing_loads`).
    * *trigger_addr_is_generator*: the kick-off is a :class:`KickOff` request
      to the trigger generator's :class:`GeneratorResultRole`, retried (mango
      ``gather``) until acknowledged; *start_locally* instead starts a run on
      this agent without any message (e.g. an ADMM coordinator role).
    * *generator_addrs* + *result_poll_s*: every *result_poll_s* seconds the
      leader requests the schedules still missing with ``gather``; after *result_retries*
      polls a missing generator gets an all-zero schedule and the run is
      marked degraded (:attr:`degraded_generators`), which the balance check
      then reports.
    """

    def __init__(
        self,
        *,
        behavior: Any,
        number_loads: int,
        generator_aids: list[str],
        trigger: Any,
        time_to_index: dict[float, int],
        schedule_by_aid: dict[str, np.ndarray],
        finished_message_type: type,
        build_start_message: Callable[[], Any],
        n_finished_required: int | None = None,
        demand_target: np.ndarray | None = None,
        balance_label: str = "",
        balance_tol: float = 0.01,
        slack: SlackSpec | None = None,
        start_locally: Callable[[Any], None] | None = None,
        generator_addrs: dict[str, Any] | None = None,
        load_timeout: float | None = None,
        request_timeout: float = 1.0,
        result_poll_s: float | None = None,
        result_retries: int = 12,
        kick_off_retries: int = 20,
        on_run_finished: Callable[[], Any | None] | None = None,
    ) -> None:
        super().__init__()
        self._slack = slack
        # Called when every required unit reported. Returning a start message
        # launches another distributed run (e.g. FDGDM's outer storage rounds);
        # returning None finishes. Finished messages carry no run id, so under
        # heavy loss a late result poll could deliver a previous run's schedule.
        self._on_run_finished = on_run_finished
        #: Distributed runs started (1 unless *on_run_finished* asks for more).
        self.runs: int = 0
        #: Per unit: rounds run and stop flag from the finished messages.
        self.iterations_by_aid: dict[str, int | None] = {}
        self.converged_by_aid: dict[str, bool | None] = {}
        #: Coordinator-level statistics (e.g. ADMM iterations / missed answers).
        self.algorithm_stats: dict[str, Any] = {}
        self._behavior = behavior
        self.number_loads = number_loads
        self._generator_aids = generator_aids
        self._trigger = trigger
        self._time_to_index = time_to_index
        self._schedule_by_aid = schedule_by_aid
        self._finished_message_type = finished_message_type
        self._build_start_message = build_start_message
        self._n_finished_required = (
            n_finished_required
            if n_finished_required is not None
            else len(generator_aids)
        )
        self._demand_target = (
            None if demand_target is None else np.asarray(demand_target, dtype=float)
        )
        self._balance_label = balance_label
        self._balance_tol = balance_tol
        self._start_locally = start_locally
        # Kept by reference: a scenario may fill it after registering the units.
        self._generator_addrs = generator_addrs if generator_addrs is not None else {}
        self._load_timeout = load_timeout
        self._request_timeout = request_timeout
        self._result_poll_s = result_poll_s
        self._result_retries = result_retries
        self._kick_off_retries = kick_off_retries

        self._ready: bool = False
        self._finished_aids: set[str] = set()
        self._pending: set[int] = set()

        #: Per hour (simulation seconds): the loads that reported it.
        self.demand_map: dict[Any, set[Any]] = {}
        self._closed_times: set[Any] = set()
        #: Largest per-timestep |Σ schedules − demand| / demand after the
        #: optimization finished; ``None`` until verified (or if no
        #: ``demand_target`` was supplied).
        self.balance_max_rel_gap: float | None = None
        #: Generators that never delivered a schedule and got the zero default.
        self.degraded_generators: list[str] = []
        #: Hours applied at their deadline with at least one load missing.
        self.hours_with_missing_loads: int = 0

    def setup(self) -> None:
        self.context.subscribe_message(
            self, self._handle_load_info, lambda c, m: isinstance(c, PowerLoadInfo)
        )
        self.context.subscribe_message(
            self,
            self._handle_finished,
            lambda c, m: isinstance(c, self._finished_message_type),
        )

    def on_ready(self) -> None:
        self._arm_hour_deadlines()
        if self._trigger is None or self._n_finished_required == 0:
            # All schedules are pre-filled; nothing to run. Still verify the
            # balance here — schedule_by_aid was already populated by the
            # scenario's own pre-scheduling before the world started, so the
            # check is meaningful even though no distributed round ever ran.
            # Skipping this left balance_max_rel_gap permanently None on this
            # path (e.g. FDGDM with 0-1 thermal generators), which silently
            # defeated the --strict CLI flag.
            self._ready = True
            self._verify_balance()
            return
        self._start_run(self._build_start_message())

    def _start_run(self, start_message: Any) -> None:
        self.runs += 1
        if self._start_locally is not None:
            self._start_locally(start_message)
        elif self._generator_addrs:
            self.context.schedule_instant_task(self._kick_off(start_message))
        else:
            self.context.schedule_instant_message(start_message, self._trigger)
        self._schedule_result_poll(1)

    def _arm_hour_deadlines(self) -> None:
        """Apply every hour at its deadline, whether or not any report arrived."""
        if self._load_timeout is None:
            return
        for time in self._time_to_index:
            self.context.schedule_timestamp_task(
                self._close_hour_at_deadline(time), float(time) + self._load_timeout
            )

    # -- kick-off and result collection ---------------------------------------

    async def _kick_off(self, start_message: Any) -> None:
        """Send :class:`KickOff` until acknowledged.

        The first run starts at the trigger generator (the others join on its
        round-0 messages). A later run kicks off every participant: after a
        finished run, participants ignore non-initial messages.
        """
        request = KickOff(start_message=start_message, run=self.runs)
        pending = (
            [self._trigger] if self.runs == 1 else list(self._generator_addrs.values())
        )
        for _ in range(self._kick_off_retries):
            replies = await self.context.gather(
                request,
                pending,
                timeout=self._request_timeout,
                reply_type=KickOffAck,
            )
            acked = {reply.aid for reply in replies.values()}
            pending = [addr for addr in pending if addr.aid not in acked]
            if not pending:
                return
        logger.error(
            "%s: the kick-off was never acknowledged after %d attempts; the "
            "distributed run did not start.",
            self._balance_label or type(self).__name__,
            self._kick_off_retries,
        )

    def _schedule_result_poll(self, attempt: int) -> None:
        if self._result_poll_s is None or not self._generator_addrs:
            return
        self.context.schedule_timestamp_task(
            self._poll_results(attempt, self.runs),
            self.context.current_timestamp + self._result_poll_s,
        )

    async def _poll_results(self, attempt: int, run: int) -> None:
        """Ask the generators whose schedule is still missing for it.

        A chain of polls belongs to one *run*; it stops once a later run
        (FDGDM's outer storage rounds) has started its own chain.
        """
        if self._ready or run != self.runs:
            return
        missing = [
            aid for aid in self._generator_addrs if aid not in self._finished_aids
        ]
        if missing:
            replies = await self.context.gather(
                self._result_request(),
                [self._generator_addrs[aid] for aid in missing],
                timeout=self._request_timeout,
                reply_type=self._finished_message_type,
            )
            for reply in replies.values():
                self._handle_finished(reply, {})
        if self._ready or run != self.runs:
            return
        if attempt < self._result_retries:
            self._schedule_result_poll(attempt + 1)
            return
        still_missing = [
            aid for aid in self._generator_addrs if aid not in self._finished_aids
        ]
        self.degraded_generators = still_missing
        logger.error(
            "%s: no answer from %d unit(s) after %d polls (%s); continuing "
            "without them. Treat this run as degraded.",
            self._balance_label or type(self).__name__,
            len(still_missing),
            attempt,
            ", ".join(still_missing),
        )
        self._give_up(still_missing)

    def _result_request(self) -> Any:
        """The request sent to units whose result is still missing."""
        return ScheduleRequest()

    def _give_up(self, missing: list[str]) -> None:
        """Units in *missing* never answered: give them all-zero schedules."""
        horizon = len(self._demand_target) if self._demand_target is not None else 0
        for aid in missing:
            self._handle_finished(
                OptimizationFinishedInfo(aid=aid, schedule=np.zeros(horizon)), {}
            )

    def accept_schedules(self, schedules: dict[str, np.ndarray]) -> None:
        """Take schedules that are already on this agent (e.g. from a local
        coordinator) as if every generator had reported them."""
        for aid, schedule in schedules.items():
            self._handle_finished(
                OptimizationFinishedInfo(aid=aid, schedule=schedule), {}
            )

    # -- applying the schedule -------------------------------------------------

    def _apply_schedule_index(self, idx: int) -> None:
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            p_schedule = np.asarray(schedule).ravel()
            if idx < p_schedule.size:
                self._behavior.act(aid, "regulate", float(p_schedule[idx]))

    def _handle_finished(self, message: Any, meta: dict) -> None:
        if message.aid in self._finished_aids:
            return  # duplicate (e.g. the reply to a poll after all)
        schedule = getattr(message, "schedule", None)
        if schedule is not None:
            self._schedule_by_aid[message.aid] = np.asarray(
                schedule, dtype=float
            ).copy()
        self._finished_aids.add(message.aid)
        iterations = getattr(message, "iterations", None)
        if iterations is not None:
            # Summed over runs: FDGDM reruns once per outer storage round.
            iterations += self.iterations_by_aid.get(message.aid) or 0
        self.iterations_by_aid[message.aid] = iterations
        self.converged_by_aid[message.aid] = getattr(message, "converged", None)
        if len(self._finished_aids) == self._n_finished_required:
            if self._on_run_finished is not None:
                next_start = self._on_run_finished()
                if next_start is not None:
                    self._finished_aids = set()
                    self._start_run(next_start)
                    return
            self._ready = True
            self._verify_balance()
            for idx in sorted(self._pending):
                self._apply_schedule_index(idx)
            self._pending.clear()

    @property
    def ready(self) -> bool:
        """Whether every required generator reported and the schedule is being applied."""
        return self._ready

    def missing_generators(self) -> list[str]:
        """Generators whose schedule never arrived; empty once the run is ready."""
        if self._ready:
            return []
        return [
            aid
            for aid in self._generator_aids
            if aid not in self._finished_aids and aid not in self._schedule_by_aid
        ]

    def _verify_balance(self) -> None:
        """Check the finished schedules against the known demand target.

        Ces et al. 2025 verify power balance before accepting a distributed
        solution; without this an unconverged run silently dispatches an
        infeasible schedule whose "cost" is meaningless (unserved demand looks
        cheap, over-generation looks expensive).
        """
        gap = self.schedule_residual()
        if gap is None:
            return
        rel = np.abs(gap) / np.maximum(np.abs(self._demand_target), 1e-9)
        self.balance_max_rel_gap = float(rel.max()) if rel.size else 0.0
        label = self._balance_label or type(self).__name__
        if self._slack is not None:
            imp, exp = compute_slack(gap)
            demand_total = max(float(np.sum(self._demand_target)), 1e-9)
            logger.info(
                "%s: slack closes the balance — import %.1f MWh (%.2f%% of "
                "demand), export %.1f MWh (%.2f%%).",
                label,
                float(imp.sum()),
                100.0 * float(imp.sum()) / demand_total,
                float(exp.sum()),
                100.0 * float(exp.sum()) / demand_total,
            )
        elif self.balance_max_rel_gap > self._balance_tol:
            logger.error(
                "%s: schedules violate power balance — max per-timestep "
                "|generation − demand| is %.1f%% of demand (tolerance %.1f%%). "
                "The optimization did not converge; treat this run's cost as "
                "invalid.",
                label,
                100.0 * self.balance_max_rel_gap,
                100.0 * self._balance_tol,
            )
        else:
            logger.info(
                "%s: power balance verified — max per-timestep gap %.3f%% of demand.",
                label,
                100.0 * self.balance_max_rel_gap,
            )

    def schedule_residual(self) -> np.ndarray | None:
        """Per-timestep ``demand − Σ schedules`` (MW) of the current schedules.

        ``None`` without a demand target. Missing or short schedules count as
        zero output.
        """
        if self._demand_target is None:
            return None
        m = len(self._demand_target)
        total = np.zeros(m, dtype=float)
        for aid in self._generator_aids:
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            arr = np.asarray(schedule, dtype=float).ravel()
            n = min(m, arr.size)
            total[:n] += arr[:n]
        return self._demand_target - total

    def hourly_dispatch(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """``(seconds, P, aids, demand)`` per hour from the final schedules.

        Not from the simulation's recordings: an hour that starts before the
        distributed run finishes is applied late, in the same instant as the
        others, so a recording never shows its set-point. Missing or short
        schedules count as zero output.
        """
        times = sorted(self._time_to_index, key=self._time_to_index.get)
        idx = np.array([self._time_to_index[t] for t in times], dtype=int)
        aids = list(self._generator_aids)
        P = np.zeros((idx.size, len(aids)))
        for j, aid in enumerate(aids):
            schedule = self._schedule_by_aid.get(aid)
            if schedule is None:
                continue
            arr = np.asarray(schedule, dtype=float).ravel()
            ok = idx < arr.size
            P[ok, j] = arr[idx[ok]]
        demand = (
            np.zeros(idx.size)
            if self._demand_target is None
            else self._demand_target[idx]
        )
        return np.asarray(times, dtype=float), P, aids, demand

    def slack_by_bucket(self, step_s: float = 3600.0) -> dict[int, tuple[float, float]]:
        """Slack ``(import, export)`` MW per simulation-time bucket.

        Keyed like the hourly CSV rows (``int(sim_seconds // step_s)``).
        """
        gap = self.schedule_residual()
        if gap is None:
            return {}
        imp, exp = compute_slack(gap)
        return {
            int(seconds // step_s): (float(imp[idx]), float(exp[idx]))
            for seconds, idx in self._time_to_index.items()
            if idx < gap.size
        }

    # -- load reports -----------------------------------------------------------

    def _handle_load_info(self, message: PowerLoadInfo, meta: dict) -> None:
        sender = sender_addr(meta).aid if meta else None
        key = sender if sender is not None else object()  # unknown sender: own slot
        time = round(float(message.time), 6)
        if time in self._closed_times:
            return  # hour already applied (at its deadline)
        reported = self.demand_map.setdefault(time, set())
        reported.add(key)
        if len(reported) != self.number_loads:
            return  # wait until all loads report for this timestep (or the deadline)
        self._close_hour(time)

    async def _close_hour_at_deadline(self, time: Any) -> None:
        if time in self._closed_times:
            return
        if len(self.demand_map.get(time, ())) >= self.number_loads:
            return
        self.hours_with_missing_loads += 1
        self._close_hour(time)

    def _close_hour(self, time: Any) -> None:
        self._closed_times.add(time)
        idx = self._time_to_index.get(time)
        if idx is None:
            return
        if self._ready:
            self._apply_schedule_index(idx)
        else:
            self._pending.add(idx)  # applied once the schedule is available


def make_finish_callback(
    *,
    leader_addr_ref: dict[str, Any | None],
    finished_message_type: type,
    algorithm_label: str,
    schedule_attr: str = "actor.P",
) -> Any:
    """Return the ``(algorithm, carrier) -> None`` hook shared by the
    carrier-based algorithms (consensus, diffusion, FDGDM, DEED-ADMM).

    *schedule_attr* is a dotted attribute path read off *algorithm* to obtain
    the finished per-timestep schedule (e.g. ``"actor.P"`` or ``"P"``).  The
    schedule travels to the leader in the finished message and stays on the
    agent's :class:`GeneratorResultRole` for later requests.
    """

    def handle_finished(algorithm: Any, carrier: Any) -> None:
        # `carrier._parent` is a distributed_resource_optimization-internal
        # attribute (the mango Role wrapping this carrier); there is no
        # public accessor for it today.
        role = carrier._parent
        aid = role.context.aid

        obj = algorithm
        try:
            for part in schedule_attr.split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(
                f"{algorithm_label}: finished algorithm for {aid!r} has no "
                f"attribute path {schedule_attr!r} ({exc}); the run may have "
                "aborted before producing a schedule."
            ) from exc
        schedule = np.asarray(obj, dtype=float).copy()
        converged = getattr(algorithm, "converged", None)
        iterations = getattr(algorithm, "iterations", None)
        result_role = role.context.get_role(GeneratorResultRole)
        if result_role is not None:
            result_role.schedule = schedule
            result_role.iterations = iterations
            result_role.converged = converged
        if converged is False:
            logger.warning(
                "%s did not converge for %s after %s iteration(s); schedule may "
                "be suboptimal.",
                algorithm_label,
                aid,
                iterations if iterations is not None else "an unknown number of",
            )

        leader_addr = leader_addr_ref.get("addr")
        if leader_addr is not None:
            role.context.schedule_instant_message(
                finished_message_type(
                    aid=aid,
                    schedule=schedule,
                    iterations=iterations,
                    converged=converged,
                ),
                leader_addr,
            )
        logger.info(
            "%s finished for %s (schedule len=%s%s)",
            algorithm_label,
            aid,
            schedule.size,
            f", iterations={iterations}" if iterations is not None else "",
        )

    return handle_finished
