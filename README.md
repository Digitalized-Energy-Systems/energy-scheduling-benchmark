# energy-scheduling-benchmark

Benchmark of distributed economic-dispatch algorithms (running as mango agents) against the
central-dispatch optimum on copper-plate networks.

The package is built on top of:

* [mango-agents](https://github.com/OFFIS-DAI/mango) 2.4.0 (PyPI) — the agent framework
* [mango-energy-environments](../mango-energy-environments) — mango environment abstractions
* [distributed-resource-optimization](../distributed-resource-optimization) — the distributed algorithms (consensus + innovations, ADMM, COHDA, …)
* [PyPSA](https://pypsa.org) and [pandapower](https://www.pandapower.org) (SimBench) — sources of the benchmark networks

## Scope of this package

This package is one of three with separate responsibilities:

- **mango-energy-environments** — simulation and representation of energy networks for mango, in
  two modes: **physical** (runs on the real frameworks — pandapower, monee — incl. their physical
  calculations) and **copper-plate** (physics skipped; lines only decide which agents are connected,
  energy moves freely between any actors; used by the benchmark). Networks, components, time series,
  observe/act, binding agents to network elements (one component or a whole bus), network sources (e.g. PyPSA) converted into backend networks,
  generic preparation steps. Existing functionality (monee restoration, pandapower scheduling) is
  kept. No distributed-algorithm or benchmark logic.
- **distributed-resource-optimization** — only the logic of the distributed algorithms, as generic
  as possible (usable for non-energy problems too) and built on mango as far as possible, including
  the generic mango carrier/role. No PyPSA, network or benchmark logic.
- **energy-scheduling-benchmark** — builds the benchmark from the other two: the economic-dispatch
  problem formulation, storage model, central-dispatch optimum, benchmark roles (leader, bus agents),
  concrete datasets, runs and comparisons.

Dependencies point one way only: energy-scheduling-benchmark → the other two; mango-energy-environments
and distributed-resource-optimization never import each other.

## Install

```bash
pip install -e .
```

The `pyproject.toml` wires the two sibling packages (mango-energy-environments,
distributed-resource-optimization) in as editable local dependencies via `tool.uv.sources`;
mango-agents comes from PyPI.

## Networks

Prepare once, run many times. `esb-prepare --network NAME` loads the raw network from
`resources/networks/` (PyPSA `.nc`, or pandapower `.json` for SimBench), converts it into an MEE `Network`, applies the benchmark's preparation recipe
(drop unusable units, quadratic cost incl. a curvature band for storage, storage defaults incl. a
battery ageing cost — Xu et al. 2018 eq. 5 with DEA 2025 values — and a PHS wear cost — Zakeri &
Syri 2015 variable O&M, `datasets.STORAGE_COST`) and saves it as NetCDF to
`resources/prepared/NAME.nc` (committed; the recipe is stored in the file's `meta` attribute). The
prepared file does not depend on the environment that runs it. Names: `toy` (tiny built-in 4-bus
chain), `base_s_1_elec_2019`, `lower_saxony_2025` (PyPSA-Eur Germany unclustered,
2025 weather/demand, cut to Lower Saxony, built by `resources/networks/build_lower_saxony_2025.py`;
offshore wind dropped — it lands in Lower Saxony but serves all of Germany; replaces
`lower_saxony_2030` since 2026-11-11 — its EEG 2030 uplift left almost no dispatch), `synthetic_10`, `synthetic_50` (seeded networks
picked for balanced bus price sensitivities incl. renewables — since renewables are pre-dispatched
the dispatchable buses are not balanced any more; built once by `resources/networks/build_synthetic.py`),
`simbench_lower_saxony` (SimBench EHV cut to Lower Saxony, pandapower JSON, built by
`resources/networks/build_simbench_lower_saxony.py`), `pypsa_eur_germany_100` (PyPSA-Eur Germany, 100
clusters, 2025 weather/demand). Both PyPSA-Eur networks (since 2026-11-12, audit Q-A2): fleet in
service at the end of 2024, renewables topped up to the IRENA 2024 country totals (PV 89.9,
onshore 63.6, offshore 9.2 GW; before, solar was only the MaStR extract's 48.8 GW); PyPSA-Eur
spreads the top-up over the buses by a heuristic, so Lower Saxony gets 10.5 GW PV vs 8.8 GW in
the state's statistics (+20 %, `docs/deviations.md` P7). The PyPSA-Eur configs both PyPSA-Eur networks were built with
are kept next to them: `resources/networks/config.germany-100-2025.yaml`,
`config.lower-saxony-2025.yaml` (run them from a PyPSA-Eur checkout). Lower Saxony (`clusters: [all]`)
also needs `pypsa-eur-powerplant-bus-mapping.patch` (`git apply` in the checkout): upstream
drops every power plant when bus names are OSM ids. Re-run `esb-prepare` after changing the recipe;
`--k`, `--band` and `--out` write a variant (e.g. linear costs: `--k 0 --band 0`) to another file.

## Running

```bash
esb-prepare --network lower_saxony_2025          # once (files for all networks are committed)
esb-run --network lower_saxony_2025 --topology ring --start 2025-01-14 --hours 168 --seed 0
esb-run --network synthetic_10 --start 2019-01-16 # standard winter day (table below), 24 h
esb-run --network path/to/prepared.nc ...        # any prepared file
esb-run --algorithm consensus --tuning paper    # or manual / sweep; --set key=value overrides
esb-run --trace ...   # debugging: mango trace (mango_trace.jsonl + .html viewer) in the run folder; slow, large
esb-tune --network toy --topology ring --jobs 4   # sweep → [consensus.toy.ring.sweep]; --algorithm consensus (default)
esb-compare                      # all runs under outputs/ vs. their central run → outputs/compare.csv
esb-run --help   # --algorithm, --loss-rate (fraction per hop, algorithm messages only), --delay-s, --grid-exchange, --no-storage, --set, --out, --trace, ...
```

### Standard and tuning days

Benchmark runs use 24 h on four **standard days** per network; parameters are tuned (`esb-tune
--start`) on four other **tuning days**, so tuning is not done on the evaluated days. Rule (per
network and meteorological season DJF / MAM / JJA / SON, full days only): the day whose daily
residual load (load − available pre-dispatched units, i.e. renewables + thermal c₁ ≤ 0, MWh — the
same rule as the runs) is closest to the season's median day is the
standard day, the second-closest is the tuning day. The dates live here, in `../run_all.sh`
(standard days, its defaults) and in `../hpc_sweeps.sh` (tuning days), not in code; regenerate with
`datasets.typical_days(load_network(datasets.prepared_path(name)))` after re-preparing a network.

| network | | winter | spring | summer | autumn |
|---|---|---|---|---|---|
| synthetic_10 | standard | 2019-01-16 | 2019-04-25 | 2019-06-13 | 2019-10-09 |
| | tuning | 2019-01-09 | 2019-04-15 | 2019-06-24 | 2019-10-25 |
| synthetic_50 | standard | 2019-01-14 | 2019-03-05 | 2019-06-13 | 2019-09-20 |
| | tuning | 2019-01-15 | 2019-05-03 | 2019-07-02 | 2019-10-02 |
| lower_saxony_2025 | standard | 2025-12-02 | 2025-03-28 | 2025-06-27 | 2025-09-26 |
| | tuning | 2025-12-30 | 2025-04-20 | 2025-08-12 | 2025-11-02 |
| simbench_lower_saxony | standard | 2016-12-18 | 2016-03-15 | 2016-07-01 | 2016-11-04 |
| | tuning | 2016-12-30 | 2016-05-16 | 2016-08-24 | 2016-10-23 |
| pypsa_eur_germany_100 | standard | 2025-12-11 | 2025-03-03 | 2025-06-26 | 2025-09-28 |
| | tuning | 2025-12-27 | 2025-03-28 | 2025-07-12 | 2025-10-31 |
| base_s_1_elec_2019 | standard | 2019-01-28 | 2019-04-16 | 2019-06-26 | 2019-09-04 |
| | tuning | 2019-12-10 | 2019-03-24 | 2019-08-13 | 2019-09-13 |

Consensus is tuned with storage on; that set is used for runs with and without storage.

Each run writes `outputs/<network>/<algorithm>/<topology>_loss<rate>_seed<seed>_from<start>_<hours>h_[no]storage[_<variant>]/`
(consensus: variant = tuning route, plus `_set<hash>` when `--set` is used)
with `schedule.csv` (time × unit MW, `grid_exchange`, `residual`, `soc:<storage>`) and `meta.json` (inputs,
prepared-file SHA-256, total cost (`null` for a failed central solve or a run that scheduled
nothing without grid exchange), `grid_exchange` (import/export MWh, import share, import cost at
4000 €/MWh; with `--grid-exchange` it takes the remaining imbalance after the run — named `slack`
before 2026-11-12), imbalance shortfall/surplus, messages
sent/delivered/lost, iterations, stop reason, `sim_time_s` = simulated time of the stop (time to
solution of a distributed run; communication only — mango's simulation has no computation-time
model, agents compute in zero simulated time; `null` for central), wall time,
`renewable_available_mwh`, `storage_simultaneous_mwh` = MWh charged and discharged in the same
hour, central only, expected 0). Consensus runs also write `trace.csv`: one row per iteration k
(the state at λ after k updates) with `sim_time_s`, `imbalance_mwh` (Σ over hours of
|Σ agents' mismatch|), `cost` (€ of the current schedules, grid exchange not included), `lambda_mean`,
`lambda_spread` (largest per-hour spread of λ across agents). With `--trace` also
`mango_trace.jsonl` and its HTML viewer `mango_trace.html`.

**Renewables are pre-dispatched** (decided 2026-11-10, `../docs/renewables-predispatch-plan.md`):
renewables and thermal units with c₁ ≤ 0 run at their available power p_max; every algorithm
(central included) only dispatches thermal units and storage against the residual load (each bus
agent: its own load minus its own renewables). Renewables are never curtailed: a surplus is
exported by the grid exchange (`--grid-exchange`) or stays in `residual` / `imbalance.surplus_mwh` (islanded).
Consensus keeps λ ≥ 0 (the renewables' marginal cost) so a surplus hour cannot drive λ down
without bound. A window whose renewables cover the load in every hour needs no ED: every
algorithm returns at once with `stop_reason = "renewables_cover"`.

### Algorithms and their parameters

| `--algorithm` | what | parameters |
|---|---|---|
| `central` | the reference optimum (cvxpy, no agents) | none |
| `consensus` | consensus + innovations, Kar & Hug 2012 / Hug, Kar & Wu 2015 (plan `../docs/consensus-plan.md`, deviations `../docs/deviations.md`) | `[consensus]` in `resources/tuned_parameters.toml`: step sizes, tolerances, `max_iter`, `agent_scale` |
| `dummy` | merit order in the leader, a plumbing test | none |

Parameters come in **tuning routes**, one set per network × topology (`--set key=value` overrides
any value; the route and the values used are stored in `meta.json`):

| route | TOML section | when it is used |
|---|---|---|
| `paper` | `[consensus]` — the papers' values, in MW / €/MWh as printed | `--tuning paper`, or the fallback (with a warning on stderr) when no tuned set exists |
| `manual` | `[consensus.<network>.<topology>.manual]` — hand-tuned, with a `note` | default when present |
| `sweep` | `[consensus.<network>.<topology>.sweep]` — written by `esb-tune` | default when there is no manual set (decided 2026-11-12) |

`esb-tune --network N --topology T --start DAY … [--jobs N] [--no-write]` sweeps the algorithm's
`sweep_grid` (32 candidates per algorithm; consensus: constant 4 κ × 4 β, decaying 4 α₀ × 2 β₀ ×
α exponent {0.6, 0.98}; α × 1/max scaled bus sensitivity, β × 1/λ_max(L) of the bus topology) on
a fixed budget with the stop rule off; a sweep run ends as `diverged` once |λ| > 4000 €/MWh, and a
candidate is pruned once one window scores worse than the best of its variant. The best candidate
that runs through every window is written whatever its gap (the gap per window is in its `note`);
nothing is written if every candidate fails. Use `--no-write` for trial sweeps and check
`git diff resources/tuned_parameters.toml` after a real one.

Per-agent steps (`agent_scale`, decided 2026-11-11, `../docs/per-agent-steps-plan.md`): `"none"`
(default, the paper route) = one α, β for all agents as in the papers; `"capacity"` (sweep sets) =
each bus runs with s_j·α and s_j·β, s_j = median / the bus's dispatchable capacity, so big buses
take smaller steps (both scaled together: the optimum stays the papers' one); the factors always
come from the network with storage, also in `--no-storage` runs, so one tuned set fits both.

`esb-compare [--root outputs] [--out FILE]` reads every `<root>/<network>/<algorithm>/<run>/meta.json`
(read-only), pairs each run with the central run of the same network, window, storage and grid-exchange
setting (only run folders of the current code: older ones are rerun, no fallbacks), and writes one row per run to `<root>/compare.csv` (signed cost gap in %, central cost,
imbalance shortfall/surplus, iterations, stop reason, messages, wall time, window start/hours)
plus a printed markdown table. Missing runs go to stderr: runs without
their central run (skipped) and, per network, windows an algorithm setting lacks that other runs
have. The batch over the standard grid is `../run_all.sh` (workspace root): all `esb-run`s, then
`esb-compare`.

## World

One mango agent per bus; it owns every component at its bus (units, storage and its load) and
reaches them through MEE's copper-plate behaviour (`register(..., bus=b)`). A `leader` sits at the
first bus / position 0; the grid exchange is no agent (it takes the remaining imbalance after the
run; the slack agent had no role and was removed 2026-11-12). The leader knows no demand: it
only learns it from the bus agents' messages. Topologies: `grid` (buses joined by lines),
`ring` (over the buses), `complete`; non-neighbours are reached hop by hop, every hop counted.
`--loss-rate p` loses each hop of the **algorithms' own messages** (consensus λ, dummy offers) with
probability p; control messages (start, stop, converged reports) always arrive (decided 2026-11-11,
`../docs/message-loss-plan.md`). Lossy consensus runs end each round for every agent at
t0 + (k+1)·2·`delay_s`: a neighbour's λ missing by then is a failed link of that round (Kar & Hug
2012 §IV); meta `messages` then also reports `rounds_by_deadline` and `missing_edges`. Lossless
runs wait for every λ as before.

The benchmark runs in mango's simulated world (discrete event, simulated delays and loss). The
same agents also run unchanged on real mango TCP containers: `tests/test_tcp.py` spreads the toy
network's consensus over three local containers (grid topology, routed messages, DRO
`json_codec()`) and reaches the simulated run's cost within 1e-4 (audit 2026-11-12 Q-A8). There
time is wall-clock and there is no loss model.

## Adding an algorithm

One ESB module per algorithm, registered in `run.ALGORITHMS` (decided 2026-11-12, audit Q-A4):
`setup(params, network, full, args)` returns `(behavior, agents, routes) -> outcome` — it adds the
algorithm's mango roles (DRO `DistributedOptimizationRole` / `CoordinatorRole`) to the world's
agents and returns `outcome() -> outputs.Outcome` (stop reason, iterations, message counts,
`sim_time_s`, optional per-iteration trace). Its parameters are the `[<algorithm>]` section of
`resources/tuned_parameters.toml` (`--tuning`, `--set` work for every algorithm); with
`sweep_grid`, `SWEEP_RUN` and `VARIANT` it can be tuned by `esb-tune --algorithm <name>`. Its
start / stop / report messages subclass DRO's `ControlMessage`, so they stay reliable on lossy
links. `consensus.py` is the worked example, `dummy.py` the smallest one.

## Package layout

```
src/energy_scheduling_benchmark/
├── problem.py    # problem rules shared by all: pre-dispatched units, grid import price, no-dispatch window
├── datasets.py   # benchmark networks + preparation recipe; esb-prepare
├── central.py    # central dispatch: the reference optimum (cvxpy)
├── storage.py    # shared storage model + price response (central dispatch + algorithms)
├── parameters.py # algorithm parameters per route (tuned_parameters.toml)
├── consensus.py  # consensus + innovations: bus actor, sweep grid, setup
├── dummy.py      # dummy merit-order algorithm (test fixture), setup
├── tune.py       # esb-tune: parameter sweep for one algorithm × network × topology
├── compare.py    # esb-compare: cost gap of every run vs. its central run
├── world.py      # mango world: bus agents, topologies, counted lossy links, leader
├── run.py        # esb-run: one run end-to-end; ALGORITHMS (the algorithm interface)
└── outputs.py    # schedule.csv + meta.json + trace.csv; Outcome, IterationTrace
resources/
├── tuned_parameters.toml  # per-algorithm parameters
├── networks/     # raw inputs (PyPSA .nc, pandapower .json), build_*.py, PyPSA-Eur configs
└── prepared/     # prepared networks (esb-prepare), read by esb-run
```
