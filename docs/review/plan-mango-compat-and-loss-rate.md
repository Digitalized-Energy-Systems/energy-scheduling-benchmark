# Plan: restore mango compatibility and fix the loss-rate configuration

Scope: the two blocking findings of
[`mango-utilisation-review.md`](mango-utilisation-review.md) (§2.1 mango compatibility,
§2.2 loss configuration). The broader loss-tolerance redesign (§2.5–2.6) is out of scope
here; it builds on this plan.

**This plan does not use mango's `gather`.** It only makes the benchmark run on the target
mango branch and makes the loss setting correct and reproducible; with it, lossy runs are
*configurable*, but the algorithms still cannot *survive* lost messages. `gather` belongs
to the follow-up loss-tolerance plan (review §2.6): coordinator-style steps (ADMM
x-update, end-of-run schedule collection, load polling) via `gather`/`open_gather` with
"keep last value, else default"; peer-to-peer rounds via per-neighbour last values plus
scheduler-clock round deadlines, because `gather` (request/reply) does not fit symmetric
broadcast rounds.

Baseline (this branch, `b6da45d`, installed mango checkout): `pytest tests` →
**98 passed, 27 failed**, all 27 with
`TypeError: create_world() got an unexpected keyword argument 'record_messages'`. Once
mango is on the target branch (A0), the same tests hit `KeyError: 'id'` (§2.1 of the review).

Repos touched: `mango-energy-environments` (A1), `energy-scheduling-benchmark` (A2–A5,
B1–B6); `mango` only switches its checkout to `fix-log-messages-opt-out` (A0). One
branch per repo; no pushes.

---

## Part A: mango compatibility

### A0. Mango target (decided)

* **Target: mango branch `fix-log-messages-opt-out`.** Verified on that branch:
  `create_world(..., record_messages=True, record_message_content=False)`,
  `gather`/`open_gather`/conversations (needed for the later loss-tolerance work), and
  the auto-install of `world.register` into the environment (so A1/A2 are still
  required).
* Switch the workspace's mango checkout from `fix-log-messages-opt-out-old` to
  `fix-log-messages-opt-out` (the editable install in `.venv` follows the checkout; no
  reinstall needed). Done by you or explicitly in the implementation step, since it changes
  the shared mango working copy.

### A1. `mango-energy-environments`: make `install` tolerate auto-install

mango now calls `environment.install(agent, agent_id=aid, **install_kwargs)` for **every**
`world.register`, including agents that represent no component (coordinators, watchers).
`DefaultEnvironment.install` strips `agent_id` and forwards the remaining kwargs to
`behavior.install`.

1. `environments/scheduling/_scheduling_behavior_base.py:120` (`SchedulingBehavior.install`):
   if `"id"` is not in `kwargs`, return without registering anything (log at DEBUG).
   Otherwise behave as today.
2. `environments/restoration/multi_energy_monee.py:209` (`install`): same rule for
   `id`/`type`.
3. Make a second `install` for the same agent/ref idempotent. It already only overwrites
   dict entries; add a test so it stays that way, since callers may still call
   `environment.install` explicitly during the transition.
4. Tests: switch `tests/test_pypsa_scheduling.py`, `test_scheduling.py`,
   `test_restoration.py`, `test_monee_energyflow.py` to
   `world.register(agent, suggested_aid=..., id=ref)` (plus `type=` for monee). Add one
   test that registers a component-less agent (e.g. `watcher`) and one that uses the
   old two-step pattern.
5. Document the new pattern in that repo's README.

### A2. Benchmark: register component agents in one call

1. Add a helper in `scenarios/_common.py`:

   ```python
   def register_component_agent(world, agent, ref: ComponentRef):
       """Register *agent* under ref.component_id and bind it to *ref* in the environment."""
       return world.register(agent, suggested_aid=ref.component_id, id=ref)
   ```

2. Replace every `world.register(...)` + `world.environment.install(agent, id=ref)` pair
   with the helper (22 call sites):
   * `admm.py` 227/228, 236/237, 271/272
   * `central_dispatch.py` 218/219, 239/240, 272/273
   * `consensus.py` 210/211, 219/220, 256/257
   * `deed_admm.py` 204/205, 215/216, 250/251
   * `diffusion.py` 212/213, 221/222, 258/259
   * `exact_diffusion.py` 230/231, 239/240, 276/277
   * `fdgdm.py` 431/432, 444/445, 454/455, 486/487
3. `tests/test_pypsa_behavior.py` (6 sites): same change.
4. Check that nothing relies on the install happening *after* `register` returns (e.g.
   `leader_agent.addr` is read right after `register`, which is unaffected).

### A3. Benchmark: `record_messages`

No code change: `build_world` already calls `create_world(..., record_messages=False)`,
which the target branch supports. (A version-detecting fallback is unnecessary now that
the branch is fixed; revisit only if older mango versions must stay supported.)

### A4. Pin the dependency

* `pyproject.toml`: keep the editable path source (`tool.uv.sources`); raise
  `mango-agents>=2.1` to the first release that contains `fix-log-messages-opt-out` once
  it is released.
* `README.md` and `hpc/README.md`: state that mango must be on `fix-log-messages-opt-out`
  (hpc/README currently names `record-messages-opt-out`, which is wrong).
* Root `CLAUDE.md`: mention the register-with-`id` pattern under "How a scenario runs"
  (step 4).

### A5. Verify

1. `pytest energy-scheduling-benchmark/tests/` → 125 passed.
2. `pytest mango-energy-environments/tests/` and `pytest distributed-resource-optimization/tests/`
   → no new failures.
3. Toy runs of all seven entry points; lossless balance gaps should match the review's
   table (diffusion 0.041 %, exact diffusion 0.170 %, FDGDM 0.000 %, DEED-ADMM 0.003 %,
   ADMM 0.012 %; consensus still fails at 94.4 %, a separate bug).
4. One `base_s_5_elec_2019` 1-day consensus run: peak memory ≈ 0.36 GB (`record_messages=False` takes effect).

---

## Part B: loss configuration

### B1. Fix the unit (decision needed)

The flag says "percent", README says fraction, mango takes a fraction. Options:

| Option | CLI | Pros | Cons |
|---|---|---|---|
| **1 (recommended)** | new `--loss-rate` (fraction, 0–1); `--loss-percent` kept as a deprecated alias that means **percent** (0–100, divided by 100) and warns | names match meanings; scripts passing `0` are unaffected | anyone who followed the README and passed `0.05` now gets 0.05 % (with a warning) |
| 2 | keep `--loss-percent`, documented as fraction; just validate `0 ≤ x ≤ 1` | smallest change | name stays misleading |
| 3 | `--loss-percent` becomes true percent, no new flag | simple | silent meaning change for README users |

Real exposure is small: every scenario except consensus refuses non-zero loss, and
`run_all_scenarios.sh` and the hpc scripts pass `--loss-percent 0`.

Implementation for option 1:

1. `build_scenario_argparser`: add `--loss-rate` (`type=_probability`, rejects values
   outside [0, 1]); keep `--loss-percent` (`type=_percentage`, rejects values outside
   [0, 100]) in a mutually exclusive group; resolve both into one `loss_rate` in
   `run_scenario_main`; emit a `DeprecationWarning` for `--loss-percent`.
2. Rename the internal parameter `loss_percent` → `loss_rate` in `build_world`,
   `require_lossless_transport`, and all seven `execute_test_case` signatures (keep
   `loss_percent=` as a deprecated keyword for one release, converted the same way, so
   external callers do not break).
3. `build_world` validates `0 ≤ loss_rate ≤ 1` and passes it to
   `SimpleCommunicationSimulation(loss_percent=loss_rate, ...)`; the docstring says
   explicitly that mango's argument is a fraction.
4. Log the effective loss rate once at INFO at scenario start.

### B2. Reproducible loss

1. Add `--seed` (int, default `0`) to the shared argparser and a `seed` parameter to
   `build_world` and every `execute_test_case`.
2. Pass `rng=random.Random(seed)` to `SimpleCommunicationSimulation`. mango draws once per
   message in delivery order, so equal seeds and equal message order give identical loss
   patterns.
3. Write the seed and loss rate into the run's outputs (e.g. a `<name>-meta.json` or the
   log), so `compare_costs.py` results can be traced back.

### B3. Make silent failures visible (small, needed for lossy consensus to be trustworthy)

* `PowerLoadAggregator`: if the world finishes and the aggregator never became ready
  (`_ready is False` with `n_finished_required > 0`), log an ERROR naming the missing
  generator aids and treat it like a failed balance check (raise under `--strict`).
  Today such a run writes a CSV with target 0 and no error (review §2.4).

### B4. Tests

* Argparser: `--loss-rate 0.05` → 0.05; `--loss-percent 5` → 0.05 with a
  `DeprecationWarning`; `--loss-rate 5` and `--loss-percent 500` rejected; both flags
  together rejected.
* `build_world`: passes the fraction through; rejects out-of-range values.
* Seed: two consensus toy runs with `loss_rate=0.05, seed=1` produce identical CSVs; with
  `seed=2` the dropped-message count differs (count via a test-only communication-sim
  wrapper).
* Existing tests that pass `loss_percent=5.0` and expect `ValueError` (admm, central
  dispatch, diffusion, exact diffusion, fdgdm, deed-admm): switch to `loss_rate=0.05`;
  they must still raise from `require_lossless_transport`.
* B3: a lossy consensus run where the leader never finishes logs the ERROR and raises
  under `strict=True`.

### B5. Docs

* `README.md` flag table and the CLAUDE.md flag list: `--loss-rate`, `--seed`, the
  deprecated `--loss-percent`.
* `run_all_scenarios.sh`: `--loss-percent 0` → `--loss-rate 0`.
* `hpc/README.md` "Reproducibility": mention `--seed`.

### B6. Verify

1. Full test suites as in A5.
2. `esb-consensus --network toy --loss-rate 0.05 --seed 1` twice → byte-identical CSVs;
   with B3 it now reports the unfinished leader instead of writing silent zeros.
3. `esb-diffusion --loss-rate 0.05` still refuses to run (unchanged until the
   loss-tolerance redesign).

---

## Order and effort

| Step | Repo | Effort | Depends on |
|---|---|---|---|
| A0 | mango (checkout switch) | XS | — |
| A1 | mango-energy-environments | S | A0 |
| A2 | benchmark | S (mechanical) | A1 |
| A3 | — | none (already done) | A0 |
| A4 | benchmark (+ docs) | XS | A0 |
| A5 | all | S | A1–A4 |
| B1 | benchmark | M | decision in B1 |
| B2 | benchmark | S | B1 |
| B3 | benchmark | S | — |
| B4–B6 | benchmark | S | B1–B3 |

Follow-ups outside this plan: the consensus balance bug at 0 % loss, and the
loss-tolerance redesign (carrier scheduler integration, `gather` for coordinator steps,
`NeighbourState` + round deadlines, running sums for FDGDM/DEED-ADMM), both described in
the review.
