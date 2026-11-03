# energy-scheduling-benchmark

Benchmark of distributed economic-dispatch algorithms (running as mango agents) against the
central-dispatch optimum on copper-plate networks.

The package is built on top of:

* [mango-agents](../mango) — the agent framework
* [mango-energy-environments](../mango-energy-environments) — mango environment abstractions
* [distributed-resource-optimization](../distributed-resource-optimization) — COHDA / ADMM / averaging consensus
* [PyPSA](https://pypsa.org) — power-system data model (replaces `PowerSystems.jl`)

## Scope of this package

This package is one of three with separate responsibilities:

- **mango-energy-environments** — simulation and representation of energy networks for mango, in
  two modes: **physical** (runs on the real frameworks — pandapower, monee — incl. their physical
  calculations) and **copper-plate** (physics skipped; lines only decide which agents are connected,
  energy moves freely between any actors; used by the benchmark). Networks, components, time series,
  observe/act, component agents, network sources (e.g. PyPSA) converted into backend networks,
  generic preparation steps. Existing functionality (monee restoration, pandapower scheduling) is
  kept. No distributed-algorithm or benchmark logic.
- **distributed-resource-optimization** — only the logic of the distributed algorithms, as generic
  as possible (usable for non-energy problems too) and built on mango as far as possible, including
  the generic mango carrier/role. No PyPSA, network or benchmark logic.
- **energy-scheduling-benchmark** — builds the benchmark from the other two: the economic-dispatch
  problem formulation, storage model, central-dispatch optimum, benchmark roles (leader, loads),
  concrete datasets, runs and comparisons.

Dependencies point one way only: energy-scheduling-benchmark → the other two; mango-energy-environments
and distributed-resource-optimization never import each other.

## Install

```bash
pip install -e .
```

The `pyproject.toml` wires the three sibling mango packages in as editable
local dependencies via `tool.uv.sources`.

## Networks

`datasets.load(name)` loads a raw PyPSA network from `resources/networks/`, converts it into an MEE
`Network` and applies the benchmark's preparation recipe in memory (drop unusable units, quadratic
cost, storage defaults); the result does not depend on the environment that runs it. Names: `toy` (tiny built-in 4-bus chain), `base_s_1_elec_2019`,
`lower_saxony_2030` (built by `resources/networks/build_lower_saxony_2030.py`).

## Running

```bash
esb-run --network lower_saxony_2030 --topology ring --start 2019-01-14 --hours 168 --seed 0
esb-run --help   # --algorithm, --loss-rate (fraction per hop), --delay-s, --slack, --out, ...
```

Each run writes `outputs/<network>/<algorithm>/<topology>_loss<rate>_seed<seed>_from<start>_<hours>h/`
with `schedule.csv` (time × unit MW, `slack`, `residual`) and `meta.json` (inputs, network SHA-256,
total cost, messages sent/delivered/lost, iterations, stop reason, wall time). The only algorithm so
far is `dummy` (merit order in the leader, a plumbing test).

## Package layout

```
src/energy_scheduling_benchmark/
├── datasets.py   # benchmark networks + preparation recipe
├── world.py      # mango world: topologies, counted lossy links, leader/slack
├── dummy.py      # dummy merit-order algorithm (test fixture)
├── run.py        # esb-run: one run end-to-end
├── outputs.py    # schedule.csv + meta.json
└── plotting.py   # matplotlib plotting utilities
```
