# Review: algorithm implementation and mango utilisation

Date: 2026-10-12. Reviewed state: `energy-scheduling-benchmark` @ `b6da45d`
(`add-more-algorithm`), `distributed-resource-optimization` @ `add-more-algorithm`,
`mango-energy-environments` @ `add-pypsa`, mango checkout on
`fix-log-messages-opt-out-old` (installed editable in `.venv`), compared against mango
`development` (`a8bff6e`) and its docs (<https://mango-agents.readthedocs.io/en/development/>).

The fix plan for the two blocking issues (§2.1, §2.2) is in
[`plan-mango-compat-and-loss-rate.md`](plan-mango-compat-and-loss-rate.md).

## 1. What mango `development` offers that matters here

| Feature | API | Relevance |
|---|---|---|
| Tracked request/reply | `send_tracked_message`, `reply_to` | Replaces the carrier's hand-rolled `_CarrierRequest`/`_CarrierReply`. |
| Scatter/gather | `gather(content, receivers, reply_type=, timeout=5.0, min_fraction=1.0)` → `{AgentAddress: reply}` | Returns partial results on timeout (never raises); first reply per sender wins; timeout runs on the scheduler clock, wrapped in `sleeping_wait()`, so it works in the stepped `SimulationWorld`. |
| Low-level gather | `open_gather(tracking_id, expected=, reply_type=)` / `close_gather` | Same collector, but lets you send *different* content per receiver (needed for ADMM). |
| Conversations | `open_conversation(timeout=)`, `join_conversation(meta)`, `conv.send/broadcast/converge/cancel` | Multi-hop protocols; message is delivered to handlers **and** the iterator. |
| Topology | `complete_topology`, `auto_assign`, `topology_neighbors`, `broadcast_to_neighbors`, characteristics, `modify_topology` + `State.BROKEN/INACTIVE` | |
| Link health | `create_topology(edge_health=EdgeHealth(...))`, `context.live_neighbours()`, `neighbour_scores()` | Detects silent neighbours; useful to bound round waits. |
| Scheduler-tracked work | `schedule_instant_task`, `schedule_instant_message`, `schedule_timestamp_task`, `@periodic`, `@on_message` | Only these are seen by the stepped world's settle/termination detection. |
| Simulated loss/delay | `SimpleCommunicationSimulation(loss_percent=, default_delay_s=, rng=)` | `loss_percent` is a **fraction** (`rng.random() >= loss_percent`). |
| Auto environment install | `world.register(agent, suggested_aid=, **install_kwargs)` forwards kwargs to `environment.install(agent, agent_id=aid, **install_kwargs)` | New behaviour; breaks the current scenarios (§2.1). |

Relevant mango internals (verified in source):

* `SimulationWorld.step` settles via `tasks_complete_or_sleeping`, which only waits on
  container/agent inboxes and **scheduler-registered** tasks. Plain `asyncio.create_task`
  work is invisible to it.
* A reply (`meta["reply"]` + `tracking_id`) with no open collector/handler is **not
  dropped**: `_handle_tracked_reply` returns `False` and the message falls through to
  normal role dispatch (`mango/agent/core.py:1224`). Late gather replies therefore reach
  ordinary handlers.

## 2. Findings

### 2.1 Blocking: the benchmark does not run against current mango

* `build_world` calls `create_world(..., record_messages=False)`. `record_messages` only
  exists on the unmerged mango branch `fix-log-messages-opt-out` (hpc/README calls it
  `record-messages-opt-out`); neither `development` nor the installed checkout has it.
  Baseline on this branch: **27 of 125 tests fail**, all with
  `TypeError: create_world() got an unexpected keyword argument 'record_messages'`.
* Behind that, every scenario crashes with `KeyError: 'id'`: mango's `world.register()`
  now auto-installs the agent into the environment, without `id`, and
  `SchedulingBehavior.install` (`mango-energy-environments`,
  `_scheduling_behavior_base.py:128`) does `kwargs["id"]`. The scenarios still call
  `world.environment.install(agent, id=ref)` separately after `register`. The mango branch
  that has `record_messages` also has auto-install, so switching branches alone does not
  help. Same pattern in `mango-energy-environments` tests (and `multi_energy_monee.install`
  does `kwargs["id"]`/`kwargs["type"]`).
* `pyproject.toml` pins `mango-agents>=2.1`, which does not express either requirement.

### 2.2 Blocking: loss configuration

* `--loss-percent` is passed through unchanged to mango's `loss_percent`, which is a
  fraction. `README.md` documents it as a fraction (0–1), but the argparse help,
  `build_world`/`require_lossless_transport` docstrings and the flag name say
  "percentage", and nothing rejects values > 1. `--loss-percent 5` therefore drops
  **every** message.
* `SimpleCommunicationSimulation` uses mango's unseeded module-level `random`, so lossy
  runs are not reproducible and cannot be compared across algorithms.

### 2.3 mango utilisation (non-blocking)

1. **Work runs outside mango's scheduler.** `MangoCarrier.send_to_other/reply_to_other/
   send_awaitable`, `DistributedOptimizationRole._handle_optimization`,
   `CoordinatorRole._handle_start`, `PowerLoadMonitoring`, `PowerLoadAggregator.on_ready`,
   `make_finish_callback`, `admm._make_finish_callback`, `central_dispatch` all use
   `asyncio.create_task`. The stepped world cannot see these tasks; correctness depends on
   incidental event-loop turns, and scheduler-clock timeouts (`sleeping_wait`) cannot work
   inside them. Use `schedule_instant_task` / `schedule_instant_message`.
2. **The carrier re-implements tracked messaging.** `_CarrierRequest`/`_CarrierReply` +
   `MangoCarrier._pending` duplicate `send_tracked_message`/`reply_to`, without timeout or
   cleanup; futures for lost messages never resolve and are never removed.
3. **Results travel through shared Python state.** Schedules reach the leader via the
   shared `schedule_by_aid` dict; the leader calls `behavior.act()` for every generator; the
   finish callback reaches into `carrier._parent`; the leader address goes through a
   `leader_addr_ref` dict. Only `FinishedInfo(aid)` travels over mango. Put the schedule in
   the message (or let the leader `gather` schedules) and let each generator apply its own
   set-point.
4. `CoordinatorRole` maps results to `others()` by list position; return
   `{address: result}` instead.
5. Handlers are wired with `subscribe_message(lambda ...)`; `@on_message(Type)` is available.
6. Every round message carries the full `data` (demand vector), needed only at kick-off.
7. Leader-based algorithms (consensus ΔP) silently require a complete graph; a topology
   characteristic (`has_characteristic="leader"`) would make this explicit.

### 2.4 Algorithm behaviour (verified by running the toy network)

Runs used a throwaway shim around §2.1 (ignore `record_messages`, skip `install` without
`id`); nothing in the repos was changed.

| Scenario, 3 days, no loss | Balance check |
|---|---|
| diffusion | 0.041 % ✓ |
| exact diffusion | 0.170 % ✓ |
| FDGDM | 0.000 % ✓ |
| DEED-ADMM | 0.003 % ✓ |
| ADMM | 0.012 % ✓ |
| **consensus** | **94.4 % ✗** (94.6 % with `--max-iter 2000`) |

* **Consensus fails its balance check without loss.** Thermal set-points stay flat at about
  their minimum (10 / 5 MW) every hour; λ never rises. This is a convergence/tuning bug
  (leader gain / ΔP path), not a messaging problem.
* **Consensus under loss fails silently.** At loss 0.05 the leader participant (`thermal0`)
  never terminates (its `k >= max_iter` messages were lost), so the aggregator never becomes
  ready and no schedule is applied. A CSV is still written (target 0, cost 24 × 3000) with
  no error; `balance_max_rel_gap` stays `None`.
* All other scenarios refuse non-zero loss (`require_lossless_transport`).

### 2.5 Loss sensitivity per algorithm

| Algorithm | Pattern | Effect of a lost message today | Keep last value safe? |
|---|---|---|---|
| ADMM | coordinator `asyncio.gather` over `send_awaitable` | coordinator blocks forever | Yes (partial/async ADMM), but never declare convergence on an incomplete round |
| Averaging consensus | P2P, advance on full queue or on a later-round message | stalls; can miss termination; leader ΔP understated (queue not keyed by sender) | Yes; keep last p_j per sender for ΔP |
| Diffusion | P2P, `len(queue) == len(neighbours)` | deadlock | Yes (bounded staleness) |
| Exact diffusion | as diffusion | deadlock | Approximate only (correction step assumes synchronous W) |
| FDGDM | P2P pairwise gradient exchange | deadlock | **No**: Σ P conservation needs both ends of an edge to use the same round values → use running-sum (cumulative) messages |
| DEED-ADMM | P2P (λ, ξ) | deadlock | λ yes; **ξ tracker no** → running sums |
| Load → aggregator | exact count per timestep | that hour is never applied | Yes; last value per load + deadline |
| Finish notification | single message | leader never ready, nothing dispatched | Use gather/retry; default schedule + mark degraded |

### 2.6 Loss-tolerance design (summary)

* **Coordinator steps** (ADMM x-update, end-of-run schedule collection, load polling): use
  mango `gather`, or `open_gather` with per-receiver content. Missing replies keep the
  previous value, or the default if never set. Drop orphaned late replies
  (`meta["reply"]` with an unknown `tracking_id`) in the role filter.
* **Peer-to-peer rounds**: `gather` is a poor fit (request/reply doubles the message count,
  responders may not have round-k state yet, early broadcasts arrive before the collector
  exists). Use a per-neighbour last-value store (`NeighbourState`, keyed by sender) and a
  round deadline on the scheduler clock. Advance when complete or at the deadline; every
  node stops by its own counter at `max_iter`; wait only for `live_neighbours()`.
* **Carrier additions**: `sender_of(meta)`, scheduler-clock `schedule`/`sleep`, a
  per-receiver gather, all work via `schedule_instant_task`; matching `SimpleCarrier`
  implementations.
* **Possible mango upstream additions**: `gather` with per-receiver content; a
  round-collect primitive that buffers early arrivals; an option to drop orphaned replies;
  rename `loss_percent` (it is a fraction).
