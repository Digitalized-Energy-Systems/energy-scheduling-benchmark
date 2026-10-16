# Plan: make every algorithm tolerate lost messages (gather-based)

Goal: every scenario runs to completion under simulated message loss. A value that was
not updated in a round keeps its last received value, or a given default if it was never
received. Results under loss stay comparable across algorithms (same seed, same loss rate).

Builds on [`plan-mango-compat-and-loss-rate.md`](plan-mango-compat-and-loss-rate.md),
which must be done first: mango on `fix-log-messages-opt-out` (it has
`gather`/`open_gather`), `--loss-rate`, `--seed`, and the "aggregator never ready" error.
Background and per-algorithm analysis: [`mango-utilisation-review.md`](mango-utilisation-review.md)
§2.5–2.6.

Repos: `distributed-resource-optimization` (Phases 1–3), `energy-scheduling-benchmark`
(Phases 4–5). One branch per repo; no pushes.

## Implementation status (2026-10-12)

Implemented on branch `loss-tolerance` of `distributed-resource-optimization`
and this branch of `energy-scheduling-benchmark`.

* **Phase 1, carrier:** `Carrier.request_all/schedule_after/now/sender_of/address`
  with defaults for existing carriers. `MangoCarrier.request_all` is mango's
  `open_gather` with per-receiver content and a scheduler-clock timeout;
  replies use `reply_to`; all work runs as mango scheduler tasks; orphaned
  late replies are dropped. `SimpleCarrier` has loss injection.
  `RoleLocalMessage` marks messages for other roles of a generator agent.
* **Phase 2, round helpers** (`algorithm/_rounds.py`): `RoundBuffer`,
  `RoundDeadline`, and `EdgeFlows` (the running-sum protocol). EdgeFlows
  turned out to be needed for **exact diffusion** too, not only FDGDM and
  DEED-ADMM: replacing a missing value with a stale one broke its sum
  invariant and biased the result (generation 23.5 MW for 20 MW demand in a
  test); with running sums it matches the lossless dispatch.
* **Phase 3:** ADMM (3.1), consensus (3.2), diffusion and exact diffusion
  (3.3/3.4, sharing one `_DiffusionRounds` base), FDGDM (3.5), DEED-ADMM
  (3.6). Additions found during implementation: a participant that stops on
  diffusion's all-converged criterion notifies its neighbours; a finished
  DEED-ADMM agent ignores late messages of its run; a firing deadline must not
  cancel itself (it dropped `SimpleCarrier`'s in-flight count to zero).
  Single-step end-of-run confirmation of the running sums (3.5, step 5) is
  not implemented; flows in flight at the end are lost (tiny once converged).
* **Phase 4:** `GeneratorResultRole`, schedule carried in the finished
  message, acknowledged kick-off (`KickOff`), hourly deadlines armed for every
  hour, leader polls missing results with `gather`, `--round-timeout`,
  `--result-poll-s`, `--result-retries`; ADMM's coordinator is started locally
  and its results go to the aggregator directly; central dispatch requests
  missing unit reports. All `require_lossless_transport` guards are removed.
  Open questions answered with the proposals: a load never heard of counts
  with the hour's forecast total; silent participants keep counting with
  their stale value; the leader still applies the set-points.
* **Not done:** `--load-timeout` / `--max-missing-rounds` as CLI flags (both
  exist as parameters), per-run output metadata and the `compare_costs.py`
  loss column (4.6), the benchmark sweep on a PyPSA-Eur network (5.5).

Verification (toy network, mango `fix-log-messages-opt-out`):

* Without loss, all seven scenarios give the same costs and balance gaps as
  before; CSVs are bit-identical for central dispatch, FDGDM and ADMM, and
  within 6e-10 MW for the others (reordered float additions).
* With `--loss-rate 0.05` and `0.2` (seeds 1, 2) every scenario finishes;
  all except consensus pass the 1 % balance check (gaps ≤ 0.18 %) at costs
  within 0.1 % of lossless; central dispatch and FDGDM reach exactly the
  lossless result. Consensus failed its balance check with or without loss
  (the separate tuning bug from the review); fixed afterwards, see the
  review's §2.4 — it now balances at 0 %, 5 % and 20 % loss.
* Tests: 260 in `distributed-resource-optimization` (incl. new loss tests on
  `SimpleCarrier` and in a stepped mango world), 144 in the benchmark (each
  scenario has a strict 5 %-loss test).

## Design in one page

There are two message patterns, and they need different tools:

| Pattern | Where | Tool |
|---|---|---|
| **Request/reply**: one agent asks many and needs their answers | ADMM x-update; leader collecting schedules; (optionally) aggregator polling loads | mango `gather` / `open_gather` with a timeout; missing replies → previous value, else default |
| **Peer-to-peer broadcast rounds**: everyone sends its state to its neighbours every round | consensus, diffusion, exact diffusion, FDGDM, DEED-ADMM | `NeighbourState` (last value per sender) + a round deadline on the scheduler clock |

Why not `gather` for the peer-to-peer rounds:
- `gather` is request/reply, so each round would cost twice the messages.
- The asked neighbour may not have its round-k value yet.
- A broadcast from a faster neighbour arrives before the round's collector exists, and
  mango passes such messages to the normal handlers instead of buffering them.

Both tools use the same rule: **advance when everything arrived, or at the deadline with
whatever arrived**. Both measure time on mango's scheduler clock, so they behave the
same in real time and in the stepped simulation.

Values that must *add up* need extra care. "Keep the old value" is safe for averaging
values (consensus λ, diffusion φ, DEED-ADMM λ, ADMM x). It is **not** safe for FDGDM's
power exchange or DEED-ADMM's ξ tracker: their correctness relies on every pairwise
contribution being applied exactly once on both ends. For those, messages carry
**running sums**, so a lost message only delays a contribution and never loses it (Phase 3.5/3.6).

Verified mango behaviour this plan relies on (target branch):
- `gather` returns partial results on timeout and keeps the first reply per sender.
- `ExternalClock.sleep` registers its deadline, and the stepped world's
  `_determine_next_step_size` includes `clock.get_next_activity()`, so the world jumps
  to deadlines.
- A reply with an unknown `tracking_id` falls through to normal dispatch
  (`agent/core.py:1224`), so it must be filtered out.
- Only scheduler-registered tasks are seen by the world's settle logic.

---

## Phase 1: carrier foundation (`distributed-resource-optimization/carrier/`)

### 1.1 Extend the `Carrier` interface (`carrier/core.py`)

```python
class Carrier(ABC):
    # existing: send_to_other, reply_to_other, others, send_awaitable (deprecated)
    @abstractmethod
    def sender_of(self, meta) -> Any: ...
    @abstractmethod
    def now(self) -> float: ...
    @abstractmethod
    def schedule_after(self, delay_s: float, callback: Callable[[], None]) -> "Cancellable": ...
    @abstractmethod
    async def request_all(
        self, requests: Mapping[Any, Any], *, timeout: float,
        reply_type: type | tuple[type, ...] | None = None,
    ) -> dict[Any, Any]:
        """Send requests[addr] to each addr and collect replies until all arrived or
        *timeout* (scheduler clock) elapsed. Missing receivers are absent from the result."""
```

`send_awaitable` becomes a thin wrapper around `request_all({receiver: content}, timeout=...)`
and is marked deprecated.

### 1.2 `MangoCarrier` (`carrier/mango.py`)

| Method | Implementation |
|---|---|
| `send_to_other` | `context.schedule_instant_message(content, receiver)` (scheduler-tracked) |
| `reply_to_other` | `context.schedule_instant_task(context.reply_to(content, meta))`; `reply_to` echoes `tracking_id` and sets `reply=True` |
| `request_all` | `open_gather(tid, expected=len(requests), reply_type=...)`; `send_message(req, addr, tracking_id=tid)` per receiver; race `collector.wait()` against `scheduler.clock.sleep(timeout)` inside `sleeping_wait()`; `close_gather(tid)` in `finally` (same as mango's `gather`, but with per-receiver content) |
| `schedule_after` | `context.schedule_timestamp_task(coro, now + delay)`; returns the scheduled task for cancellation |
| `sender_of` | `mango.sender_addr(meta)` |
| `now` | `context.current_timestamp` |

Delete `_CarrierRequest`, `_CarrierReply`, `MangoCarrier._pending`, `_resolve_reply`, and
the reply subscription in `_CarrierRole.setup`.

### 1.3 Roles (`carrier/mango.py`)

1. `DistributedOptimizationRole`: in the message filter, **drop orphaned replies** (`meta.get("reply")`
   and `meta.get("tracking_id")`): these are late answers to a `request_all` that already
   timed out. Run `on_exchange_message` with `schedule_instant_task`, not
   `asyncio.create_task`.
2. `CoordinatorRole`: run the coordinator with `schedule_instant_task`; return results as
   `{address: result}` instead of by list position.
3. Algorithm handlers stay free of `await`s between reading and writing their state, so
   parallel handler tasks cannot interleave. Add a docstring rule saying so.

### 1.4 `SimpleCarrier` parity (`carrier/simple.py`)

Add the same methods for the in-process carrier, plus **optional loss injection** so the
algorithm unit tests can check loss tolerance without mango:
- `ActorContainer(loss_rate=0.0, rng=None, delay_s=0.0)`.
- Dropped messages never dispatch, and they still count toward `done_event`.
- `schedule_after` uses `loop.call_later`; `request_all` uses `asyncio.wait(..., timeout)`.

### 1.5 Tests

- `request_all` returns partial results when one receiver drops the request (both
  carriers).
- A late reply after the timeout is dropped and never reaches `on_exchange_message`.
- In a `SimulationWorld` (`ExternalClock`), a `request_all` with timeout 1.0 s and one
  lost message completes at simulation time ≈ t+1.0 without blocking `step_until`.
- `schedule_after` fires at the right simulation time and can be cancelled.
- The 14 existing carrier-based test modules pass unchanged with `loss_rate=0`.

---

## Phase 2: shared round helpers (`algorithm/_rounds.py`, new)

```python
class NeighbourState(Generic[T]):
    """Last value per neighbour; default until the first message arrives."""
    def __init__(self, neighbours, default: T | Callable[[Any], T]): ...
    def receive(self, sender, k: int, value: T) -> bool:   # newer k wins; older ignored
    def complete(self, k: int) -> bool:                    # every neighbour reported >= k
    def values(self) -> dict[Any, T]                       # fresh or stale, never missing
    def ever_heard(self, sender) -> bool
    def staleness(self, sender, k: int) -> int             # rounds since last update

class RoundClock:
    """Advance on 'all arrived' or on the per-round deadline, whichever is first."""
    def __init__(self, carrier, timeout_s: float, on_advance: Callable[[int], None]): ...
    def start_round(self, k: int)        # arms the deadline for round k
    def maybe_advance(self, k: int)      # call after each receive(); advances if complete
    def stop(self)                       # cancel the pending deadline
```

Rules shared by all peer-to-peer algorithms:

1. A message for an older round still updates `NeighbourState` if it is newer than what
   the receiver holds (it is the freshest information available), but it never triggers
   an advance.
2. Messages for future rounds are stored, so a neighbour one round ahead does not
   count as missing.
3. The deadline handler only advances if the node is still in that round. It acts on
   `values()`, the last known value or the default.
4. **Every node stops by its own counter** at `max_iter`. Termination never depends on
   receiving someone else's last message.
5. Optional: if the topology has `EdgeHealth`, wait only for `live_neighbours()` before
   the deadline, so dead peers don't cost a full timeout every round.
6. Lossless runs must not be affected: with no loss every round completes before its
   deadline, so the results must match today's.

---

## Phase 3: per-algorithm changes

Each step also removes `require_lossless_transport` from the matching scenario (Phase 4.4).

### 3.1 ADMM coordinator (`algorithm/admm/core.py`), uses `gather`

- `_x_update_round`:
  `replies = await carrier.request_all({addr: ADMMMessage(v_i, rho)}, timeout=round_timeout, reply_type=ADMMAnswer)`.
  Set `x[i] = replies[addr].x` if present, otherwise keep `x[i]` (default: the initial zeros).
- Track `fresh[i]` per round. **Declare convergence only after a round where every
  participant answered.** Otherwise an all-stale round leaves z unchanged, the dual
  residual drops to 0, and ADMM looks converged when it isn't.
- Per participant, count consecutive misses. Above `max_missing_rounds`, log a warning
  and end the run with `converged = False`.
- Participants (`flex_actor.py:65`, `storage.py:92`) keep calling `reply_to_other`. The
  carrier now routes that through mango's `reply_to`, so the participants need no change.
- Background: with stale x this becomes partially asynchronous ADMM, which converges for
  bounded staleness.

### 3.2 Averaging consensus (`algorithm/consensus/averaging.py`)

- Replace `_message_queue` with a `NeighbourState` of `(lam, p)` keyed by sender.
  Defaults: `lam = initial_lam`, `p = 0`.
- Leader ΔP = `pd − (own p + Σ last known p_j)`. Apply the leader term only once it has
  heard every neighbour at least once, so the default `p = 0` can't cause a huge first
  correction. This also fixes today's understated ΔP when messages are missing.
- Remove the "jump to a later round" heuristic and the termination by a received
  `k >= max_iter` message. Use the `RoundClock` and stop by the node's own counter.
- Separate issue: the lossless balance failure (94.4 %) is a tuning bug, fixed on its own.

### 3.3 Diffusion (`algorithm/diffusion/diffusion.py`)

- `NeighbourState` of `(phi, converged)`. Check `degree`/`weight_rule` on the first
  message from each sender instead of every round.
- Combine step: same weights, using the last known φ_j (default: own initial φ).
- Convergence stop: own flag plus all last known flags are true, **and** every flag was
  updated within the last `patience` rounds, so a flag that is stale because of loss
  can't end the run. Otherwise stop at `max_iter`.

### 3.4 Exact diffusion (`algorithm/diffusion/exact_diffusion.py`)

- Same as 3.3. Document that the correction step assumes synchronous weights, so under
  loss the result is approximate rather than exact. The balance check will show how much.

### 3.5 FDGDM (`algorithm/fdgdm/fdgdm.py`), running sums

Today: `ΔP_i = Σ_j w_ij (g_j − g_i)`. Σ P stays constant only if both ends of every
edge use the same pair (g_i, g_j) in the same round.

New edge protocol:
1. Each edge (i, j) has an **owner**, the end with the smaller address. In round k the
   owner computes `f_ij = w_ij (g_j − g_i)` from its own g_i(k) and the last known g_j,
   and applies `+f_ij` to itself immediately.
2. The owner adds `f_ij` to its running sum `F_ij` and sends `F_ij` to j, attached to its
   normal per-round message. The message count stays the same, but the content now
   differs per receiver.
3. j applies `−(F_ij − F_ij_last_applied)` when the message arrives. A lost message only
   delays these flows until the next running sum arrives.
4. Non-owners still broadcast their gradient g_j every round. Owners keep it in
   `NeighbourState` (default: own gradient, so the flow starts at 0).
5. End of run: owners keep resending the final `F_ij` until j confirms it. Use
   `request_all` with retries; j's reply confirms the sum it applied. Flows still in
   flight are therefore applied before the schedules are reported.
- Tests: with loss, Σ P before projection stays constant within 1e-9 after the end-of-run
  step; without loss the results match today's within numerical tolerance (the order of
  additions changes).

### 3.6 DEED-ADMM (`algorithm/deed_admm/deed_admm.py`)

- λ mixing `λ̃ = λ + β Σ_j (λ_j − λ)`: last known λ_j is fine.
- ξ mixing `ξ̃ = ξ + β Σ_j (ξ_j − ξ)`: each pair's terms cancel, and that is what keeps
  `Σ ξ = Σ (x − y)` true. Use the same owner / running-sum protocol as 3.5, with
  `f_ij = β (ξ_j − ξ_i)`.
- Already stops by its own counter, so `_finish` stays the same. Add the final running-sum
  confirmation from 3.5 before `_finish`.

### 3.7 COHDA

Not used by the scenarios. Its gossip merges working memories, which tolerates loss
apart from termination. Out of scope; note it in the module docstring.

---

## Phase 4: scenario layer (`energy-scheduling-benchmark/scenarios/_common.py`)

### 4.1 Load reporting (`PowerLoadMonitoring` → `PowerLoadAggregator`)

- Keep the push model (loads send on each `PowerUpdateInfo`). The aggregator keeps the
  **last value per load**, keyed by sender. The default before the first report is that
  load's forecast for that hour from the scenario timeseries, passed in at construction.
- On the first report for a timestep, schedule a deadline with
  `schedule_timestamp_task(t + load_timeout)`. Apply the hour when all loads reported or
  at the deadline, using last known values for the missing ones.
- Send with `schedule_instant_message`, not `asyncio.create_task`.
- Alternative, if it turns out simpler: the aggregator `gather`s `LoadRequest` from all
  loads each hour (twice the messages, same keep-last rule).

### 4.2 Collecting results (replaces the shared `schedule_by_aid` dict)

- Generators send `FinishedInfo(aid, schedule)` when they finish, so the schedule now
  travels over mango.
- The leader starts a result deadline when the run starts (estimate
  `max_iter × round_timeout + margin`, configurable). At the deadline, or as soon as
  all have reported, it `gather`s `ScheduleRequest` from the generators that are still
  missing, with `result_retries` attempts.
- A generator that still hasn't answered gets its default schedule (its last
  reported intermediate schedule if any, else `p_min`). The run is marked **degraded**
  in the log and outputs, and the balance check then decides validity as usual.
- `make_finish_callback` / `admm._make_finish_callback`: stop using `carrier._parent`,
  `schedule_by_aid` and `leader_addr_ref`. The finished algorithm sends its own message.
- Optional later step: each generator applies its own set-point instead of the leader
  calling `behavior.act` for everyone.

### 4.3 Central dispatch

Uses the same load reporting (4.1) and the same broadcast of results (4.2 in reverse:
the leader `gather`s acknowledgements from generators). After that it is loss tolerant too.

### 4.4 Remove the loss guards

Remove `require_lossless_transport` per scenario as soon as its algorithm (Phase 3) and
the shared layer (4.1/4.2) are done. Remove `lossless_only` from the argparser once all
seven scenarios are converted.

### 4.5 New shared CLI flags (`build_scenario_argparser`)

| Flag | Default | Meaning |
|---|---|---|
| `--round-timeout` | `3 × --delay-s` | Peer-to-peer round deadline and ADMM `request_all` timeout |
| `--max-missing-rounds` | `20` | Consecutive misses before a participant is treated as gone (ADMM) / logged |
| `--result-retries` | `3` | Retries when collecting final schedules |
| `--load-timeout` | `3 × --delay-s` | Deadline for applying an hour with missing load reports |

### 4.6 Outputs

Write per run: loss rate, seed, round timeout, degraded flag, number of stale
neighbour-rounds per agent, and dropped-message count (from a thin counting wrapper around
`SimpleCommunicationSimulation`). Extend `compare_costs.py` with a `loss_rate` column, so
cost against loss rate can be compared across algorithms.

---

## Phase 5: validation

1. **Lossless regression:** with `--loss-rate 0` every scenario's CSV matches today's
   (FDGDM / DEED-ADMM within 1e-9 because of the running-sum arithmetic). This is the
   gate for every phase.
2. **Unit tests per algorithm** (`SimpleCarrier` with loss injection, rates 0.05 and 0.2,
   5 seeds): the run finishes. Diffusion, consensus and ADMM land close to the lossless
   optimum. FDGDM conserves Σ P before projection, and DEED-ADMM keeps
   `Σ ξ = Σ (x − y)`.
3. **Scenario tests (mango, toy network):** each scenario at `--loss-rate 0.05 --seed 1`
   finishes, and either passes the balance check or is flagged degraded. It never writes
   silent zeros.
4. **Timing:** with `--round-timeout` below the delay, lossless runs show staleness but
   still converge (the setting has an effect but doesn't break runs). Default runs show
   zero stale rounds without loss.
5. **Benchmark sweep:** `run_all_scenarios.sh --loss-rate {0, 0.01, 0.05, 0.1}` on one
   PyPSA-Eur network, 3 seeds each, giving cost and balance gap against loss rate per
   algorithm.

---

## Risks

| Risk | Mitigation |
|---|---|
| Round timeout too short for the delay creates false staleness without loss | Default 3 × delay; Phase 5.4 test; log a warning if `round_timeout < 2 × delay` |
| Deadlines cancelled early stay in `ExternalClock._futures` and cause empty world steps (more steps on large networks) | One deadline per agent per round (not per message); measure step count on a large network; upstream: remove cancelled sleeps from the clock |
| `schedule_instant_task` per message is slower than raw `create_task` on large networks | Measure on `base_s_5_elec_2019` before and after Phase 1; if needed, send a round's messages from one task |
| Changing results of the consensus heuristic (round jumping removed) | Covered by the lossless regression; consensus already fails its balance check, see the separate fix |
| FDGDM/DEED-ADMM end-of-run confirmation never completes if a peer is permanently gone | Bounded retries, then mark degraded |

## Possible mango upstream additions

- `gather` with per-receiver content (removes the custom `request_all` code).
- A round-collect primitive that buffers early arrivals (would replace `RoundClock`).
- An option to drop orphaned replies instead of dispatching them.
- Remove cancelled `ExternalClock.sleep` futures.

## Order and effort

| Phase | Repo | Effort | Depends on |
|---|---|---|---|
| 1 Carrier | distributed-resource-optimization | M | compat plan |
| 2 Round helpers | distributed-resource-optimization | S | 1 |
| 3.1 ADMM | distributed-resource-optimization | S | 1 |
| 3.2–3.4 Consensus, diffusion, exact diffusion | distributed-resource-optimization | M | 2 |
| 3.5–3.6 FDGDM, DEED-ADMM (running sums) | distributed-resource-optimization | L | 2 |
| 4.1–4.3 Scenario layer | energy-scheduling-benchmark | M | 1 |
| 4.4–4.6 Flags, guards, outputs | energy-scheduling-benchmark | S | 3, 4.1–4.3 |
| 5 Validation | both | M | all |

Suggested first slice: Phase 1 + 3.1 + 4.1/4.2. ADMM is the clearest use of `gather` and
proves the carrier changes end to end.

## Open questions

1. Default load value before the first report: the forecast (proposed), zero, or the
   previous hour?
2. Should a participant that stays silent be excluded from the round (topology link set
   to `BROKEN`), or keep counting with its stale value (proposed)?
3. Should generators apply their own set-points (4.2, optional step) as part of this work
   or later?
