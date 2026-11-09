# energy-scheduling-benchmark

Benchmark of distributed economic-dispatch algorithms (running as mango agents) against the
central-dispatch optimum on copper-plate networks.

The package is built on top of:

* [mango-agents](https://github.com/OFFIS-DAI/mango) 2.4.0 (PyPI) — the agent framework
* [mango-energy-environments](../mango-energy-environments) — mango environment abstractions
* [distributed-resource-optimization](../distributed-resource-optimization) — COHDA / ADMM / averaging consensus
* [PyPSA](https://pypsa.org) — source of the benchmark networks

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
(drop unusable units, quadratic cost incl. a curvature band for renewables/storage, storage defaults) and saves it as NetCDF to
`resources/prepared/NAME.nc` (committed; the recipe is stored in the file's `meta` attribute). The
prepared file does not depend on the environment that runs it. Names: `toy` (tiny built-in 4-bus
chain), `base_s_1_elec_2019`, `lower_saxony_2030` (built by
`resources/networks/build_lower_saxony_2030.py`), `synthetic_10`, `synthetic_50` (seeded networks
with balanced bus price sensitivities, built once by `resources/networks/build_synthetic.py`),
`simbench_lower_saxony` (SimBench EHV cut to Lower Saxony, pandapower JSON, built by
`resources/networks/build_simbench_lower_saxony.py`), `pypsa_eur_germany_100` (PyPSA-Eur Germany, 100
clusters, 2025 weather/demand/fleet). The PyPSA-Eur configs both PyPSA-Eur networks were built with
are kept next to them: `resources/networks/config.germany-100-2025.yaml`,
`config.lower-saxony-2030.yaml` (run them from a PyPSA-Eur checkout). LS2030 (`clusters: [all]`)
also needs `pypsa-eur-powerplant-bus-mapping.patch` (`git apply` in the checkout): upstream
drops every power plant when bus names are OSM ids. Re-run `esb-prepare` after changing the recipe;
`--k`, `--band` and `--out` write a variant (e.g. linear costs: `--k 0 --band 0`) to another file.

## Running

```bash
esb-prepare --network lower_saxony_2030          # once (files for all networks are committed)
esb-run --network lower_saxony_2030 --topology ring --start 2019-01-14 --hours 168 --seed 0
esb-run --network synthetic_10 --day winter      # standard day: winter|spring|summer|autumn
esb-run --network path/to/prepared.nc ...        # any prepared file
esb-run --algorithm consensus --tuning paper    # or manual / sweep; --set key=value overrides
esb-tune --network toy --topology ring --steps constant   # sweep → [consensus.toy.ring.sweep]
esb-run --help   # --algorithm, --loss-rate (fraction per hop), --delay-s, --slack, --no-storage, --set, --out, ...
```

Standard days (`--day`, also `esb-tune --day`): per network and season the 24 h day whose residual
load (load − available renewables) is closest to the season's median day
(`datasets.STANDARD_DAYS`, kept in sync with `datasets.typical_days` by a test).

Each run writes `outputs/<network>/<algorithm>/<topology>_loss<rate>_seed<seed>_from<start>_<hours>h_[no]storage[_<variant>]/`
(consensus: variant = tuning route, plus `_set<hash>` when `--set` is used)
with `schedule.csv` (time × unit MW, `slack`, `residual`, `soc:<storage>`) and `meta.json` (inputs,
prepared-file SHA-256, total cost (`null` for a failed central solve or a run that scheduled
nothing without slack), slack use, imbalance shortfall/surplus, messages
sent/delivered/lost, iterations, stop reason, wall time). Algorithms so far: `central` (the
reference optimum), `consensus` (consensus + innovations, Kar & Hug 2012 / Hug, Kar & Wu 2015;
plan `../docs/consensus-plan.md`, deviations `../docs/deviations.md`) and `dummy` (merit order in
the leader, a plumbing test). No compare script yet (cost gap vs. central across runs): only
`esb-tune` computes the gap. Consensus parameters (step sizes, tolerances, `max_iter`) come from
`resources/tuned_parameters.toml`, one set per tuning route and network × topology:
`--tuning paper` = the papers' values (`[consensus]`, used in MW / €/MWh as printed), `manual` =
hand-tuned `[consensus.<network>.<topology>.manual]` (default when present), `sweep` = written by
`esb-tune` (grid over the step sizes, best converged cost gap vs central). `--set key=value`
overrides any value; the route and values used are stored in `meta.json`.

## World

One mango agent per bus; it owns every component at its bus (units, storage and its load) and
reaches them through MEE's copper-plate behaviour (`register(..., bus=b)`). A `leader` (and with
`--slack` a `slack` agent) sits at the first bus / position 0. The leader knows no demand: it
only learns it from the bus agents' messages. Topologies: `grid` (buses joined by lines),
`ring` (over the buses), `complete`; non-neighbours are reached hop by hop, every hop lossy and counted.

## Package layout

```
src/energy_scheduling_benchmark/
├── datasets.py   # benchmark networks + preparation recipe; esb-prepare
├── central.py    # central dispatch: the reference optimum (cvxpy)
├── storage.py    # shared storage model + price response (central dispatch + algorithms)
├── consensus.py  # consensus bus actor + parameters (tuned_parameters.toml)
├── tune.py       # esb-tune: step-size sweep for one network × topology
├── world.py      # mango world: bus agents, topologies, counted lossy links, leader/slack
├── dummy.py      # dummy merit-order algorithm (test fixture)
├── run.py        # esb-run: one run end-to-end
└── outputs.py    # schedule.csv + meta.json
resources/
├── tuned_parameters.toml  # per-algorithm parameters
├── networks/     # raw PyPSA inputs + build_*.py
└── prepared/     # prepared networks (esb-prepare), read by esb-run
```
