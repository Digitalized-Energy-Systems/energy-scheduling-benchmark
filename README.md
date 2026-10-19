# energy-scheduling-benchmark

Multi-agent energy scheduling benchmark scenarios — a Python port and
extension of the original `EnergySchedulingBenchmark.jl`.

The package is built on top of:

* [mango-agents](../mango) — the agent framework
* [mango-energy-environments](../mango-energy-environments) — mango environment abstractions (`PyPSABehavior`, component classification)
* [distributed-resource-optimization](../distributed-resource-optimization) — the distributed algorithms (consensus, ADMM, diffusion, FDGDM, DEED-ADMM, COHDA)
* [PyPSA](https://pypsa.org) — power-system data model (replaces `PowerSystems.jl`)
* [Pyomo](https://pyomo.readthedocs.io) — LP modelling layer (replaces JuMP); any
  Pyomo-compatible LP solver works, but HiGHS (via `highspy`) is preferred.

## Install

```bash
pip install -e .
```

The `pyproject.toml` wires the three sibling packages in as editable
local dependencies via `tool.uv.sources`.

## Choosing a network

Scenarios work out of the box on any PyPSA network.  You do not need to
hand-build one — the loader layer accepts all of the following:

| Source                              | Example                                         |
|-------------------------------------|-------------------------------------------------|
| Built-in PyPSA example              | `load_scenario("ac-dc-meshed")`                 |
| NetCDF file                         | `load_scenario("/path/to/grid.nc")`             |
| HDF5 file                           | `load_scenario("/path/to/grid.h5")`             |
| CSV folder                          | `load_scenario("/path/to/csv_folder/")`         |
| Excel workbook                      | `load_scenario("/path/to/grid.xlsx")`           |
| PYPOWER IEEE test case              | `load_scenario("case14")` (requires `pypower`)  |
| Existing `pypsa.Network`            | `load_scenario(my_net)`                         |
| Toy 5-bus fixture (tests / smoke)   | `build_toy_network()`                           |

The loader returns a `ScenarioData` bundle (network + auto-extracted
timeseries + start datetime).  Feed it straight into the behavior:

```python
from energy_scheduling_benchmark import (
    PyPSABehavior, available_examples, load_scenario,
)

print(available_examples())           # ['ac-dc-meshed', 'model-energy', ...]
scenario = load_scenario("storage-hvdc")
behavior = PyPSABehavior.from_scenario(scenario)
```

Timeseries are pulled automatically from the network's native
`generators_t.p_max_pu`, `loads_t.p_set`, and `storage_units_t.p_set`
DataFrames.  Generator classification (thermal vs renewable) follows the
PyPSA `carrier` attribute.

PyPSA-Eur extracts for benchmarking live in `resources/networks/` (see
`resources/networks/print_network.py` for how to regenerate them from the
sibling `pypsa-eur/` workflow; `resources/networks/extract_lower_saxony.py`
builds the `lower_saxony_2030.nc` regional extract — see
`docs/lower-saxony-2030-workflow.md` in the monorepo root).

## Running the scenarios

Seven scenario entry points are installed:

| Entry point            | Algorithm                                                          |
|------------------------|--------------------------------------------------------------------|
| `esb-central-dispatch` | Centralised full-horizon Pyomo/HiGHS LP incl. storage (lower bound)|
| `esb-consensus`        | Averaging consensus with gradient (Jian et al. 2020)               |
| `esb-admm`             | Exchange ADMM with proximal actors (Boyd et al. 2011)              |
| `esb-diffusion`        | Adapt-then-combine diffusion (Ces et al. 2025)                     |
| `esb-exact-diffusion`  | Bias-corrected exact diffusion (Yuan et al. 2018 / Ces et al. 2025)|
| `esb-fdgdm`            | Fast distributed gradient descent (Bai et al. 2022)                |
| `esb-deed-admm`        | DEED-ADMM (Zhu et al. 2025)                                        |

All seven share the same CLI flags: `--network` (default `toy`),
`--name-base`, `--out-dir` (directory for the outputs, created if missing;
default: the current directory), `--simulate-days` (default `3`), `--delay-s`
(comms delay, default `0.02`), `--loss-rate` (probability that a message is lost, a
fraction from 0 to 1, e.g. `0.05` for 5 %; default `0`), `--seed` (seed for
the loss draws, default `0`, so lossy runs are reproducible), `--cost-model`
(`linear`, the default PyPSA marginal costs, or `quadratic`: thermal units add
`c₂·P²` with `c₂ = 0.11·marginal_cost/p_nom`, so their marginal cost is 22 % higher
at full output — k = 0.11 from Wood & Wollenberg's Example 4E; renewables and
storage stay linear; central dispatch then solves a QP with HiGHS), `--topology`
(communication graph, see below; default `complete`), `--tuned-params` (see
below), `--log-level`,
plus `--balance-tol` (max per-timestep |generation − demand| / demand before a
finished run is flagged as not converged, default `0.01`) and `--strict` (exit
with an error instead of writing outputs when that check fails, or when the
distributed schedule never finished).

### Tuned parameters

Algorithm parameters that are tuned rather than derived — iteration budgets
(`max_iter`), ADMM `rho`, consensus `alpha` / `leader_damping`, DEED-ADMM `gamma`,
diffusion `tol`, exact diffusion's `weight_rule` — live in
`src/energy_scheduling_benchmark/tuned_parameters.toml`, per scenario, with
optional sections per topology and per network:

```toml
[consensus]                    # every network
max_iter = 500

[consensus.topology.ring]      # sparse graphs need more rounds
max_iter = 3000

[network."base_s_2_elec_de-at-storage-2023".deed_admm]
max_iter = 2000
```

Later sections override earlier ones in that order, and an explicit flag
(`--max-iter 800`) overrides the file. `--tuned-params FILE` reads another file.
Each run's `<name>-meta.json` records the values used and, under
`dimensions.tuned_from`, the section each one came from. Values fixed by a paper
(FDGDM / DEED-ADMM `tol_ratio`) and values derived from the network (step sizes,
ε bands) stay in code.

### Communication topology

`--topology` sets who talks to whom (built by
`mango_energy_environments...scheduling.topology`):

* `complete` (default): every agent talks to every other one.
* `ring`: the algorithm's agents form a cycle ordered by component id.
* `grid`: follows the electrical network. Agents on one bus are all connected;
  every agent on bus A talks to every agent on bus B when a line, link or
  transformer joins A and B. Buses without algorithm agents are passed through,
  and disconnected islands are bridged so the graph stays connected.

In `ring` and `grid` the loads (and units outside the algorithm) attach via their
bus. The leader (the first load) stays where it is: every message, including
load reports, kick-off and results, travels the shortest path through the graph
and takes hops × `--delay-s`, with `--loss-rate` applied per hop. The algorithms
adapt to sparse graphs:

* diffusion / exact diffusion use per-neighbour weights from the neighbours'
  degrees; exact diffusion scales its step by the Perron entry (Ces et al. eq. 31)
  where the weight matrix is not doubly stochastic;
* DEED-ADMM uses β = 1/(max degree + 1) (W = I − βL stays doubly stochastic);
* the convergence stops of diffusion, exact diffusion, FDGDM and DEED-ADMM need
  every agent converged (a "converged radius" reaching the graph diameter);
  FDGDM and DEED-ADMM then stop in one common round so their pairwise flows
  stay conserved;
* consensus followers that are not the leader's neighbours send it their power
  each round, so the leader still sees the system-wide imbalance ΔP.

With `complete` the outputs are byte-identical to the code before `--topology`.

### Message loss

Every scenario runs under message loss (`--loss-rate`). A value that was not
updated in time keeps its last received value, or a default if it was never
received:

* **Algorithm rounds** (consensus, diffusion, exact diffusion, FDGDM,
  DEED-ADMM) advance when all neighbours reported *or* after `--round-timeout`
  seconds of simulation time (default 3 × `--delay-s` when `--loss-rate` > 0;
  lossless runs have no round deadline, as in the papers); a missing neighbour
  contributes its last value. Pairwise exchanges whose sum must be conserved
  (FDGDM's power transfers, DEED-ADMM's ξ tracker, exact diffusion's mixing)
  use running sums, so a lost message delays a transfer instead of losing it.
* **ADMM** collects each round's answers with mango's `gather` and a timeout;
  a participant that did not answer keeps its previous value. A participant
  that misses `--max-missing-rounds` rounds in a row (default 20; `0`
  disables) ends the run as not converged.
* **Leader**: the kick-off is acknowledged and retried; the hourly set-points
  are applied at a deadline (`--load-timeout` seconds into the hour, default
  3 × `--delay-s`) even if load reports are missing, counting a missing load
  with its forecast; lost results
  (schedules, central dispatch's unit reports) are requested again with
  mango's `gather` every `--result-poll-s` seconds. After `--result-retries`
  requests a missing result is given up, the run is logged as degraded and
  the balance check decides whether it is valid.

Without loss none of this changes a run: rounds and requests complete before
their deadlines, and results match the lossless implementation.

The scenarios need mango `development` (the message-log opt-out is merged
there): it provides `create_world(record_messages=...)`, and its
`world.register(agent, ..., id=ref)` installs the agent in the environment in
the same call.

Algorithm-specific flags (see `<entry-point> --help`):

| Entry point           | Extra flags (defaults)                                                         |
|-----------------------|--------------------------------------------------------------------------------|
| `esb-consensus`       | `--max-iter 500`, `--alpha 0.2`, `--leader-damping 0.5`                        |
| `esb-admm`            | `--rho 0.1`, `--max-iter 2000`                                                 |
| `esb-diffusion`       | `--max-iter 2000`, `--tol 1e-3`                                                |
| `esb-exact-diffusion` | `--weight-rule averaging`, `--max-iter 6000`, `--tol 1e-3`                     |
| `esb-fdgdm`           | `--max-iter 300`, `--tol-ratio 0.000222` (Eq. 24 stop, 0.01/45 × p_nom; 0 = off) |
| `esb-deed-admm`       | `--gamma 0.05`, `--max-iter 500`, `--tol-ratio 0.000222` (stop, as FDGDM; 0 = off) |

### Optional slack node

The networks are islanded, so by default every scenario must match demand with its
own units, and runs that miss by more than 1 % are excluded from the cost
comparison. `--slack` adds a slack node (the point of common coupling of Bai et
al. 2022) to all seven scenarios: it imports any shortfall at
`--slack-import-price` (default 4000 €/MWh, the EU day-ahead price cap) and exports any surplus at
`--slack-export-price` (default 0 €/MWh). Both are added to `cost:total` and
written as `slack:import`/`slack:export` columns; central dispatch gets the same
slack inside its LP. Off by default. See [`docs/slack-node.md`](docs/slack-node.md).

```bash
esb-fdgdm --network resources/networks/lower_saxony_2030.nc --name-base fdgdm --slack
```

`esb-central-dispatch` solves one LP over the whole simulated horizon, with
perfect foresight of demand and renewable availability (the same information
the distributed leaders get) and SOC-constrained storage using the same model
and cyclic terminal condition as the distributed storage actors. Its cost is
therefore a lower bound for every balanced distributed run.

```bash
# Toy 5-bus fixture (default network)
esb-consensus --name-base consensus_out

# Any PyPSA example by name
esb-central-dispatch --network ac-dc-meshed --name-base ac_dc_central

# Or load from a file / run as a module
python -m energy_scheduling_benchmark.scenarios.diffusion \
    --network /path/to/mynet.nc --name-base mynet_diffusion
```

Each run produces (in `--out-dir`):

* `<name>-df.csv`          — per-agent timeseries (one row per hourly snapshot) incl. a `cost:total` column
  (plus `slack:import`/`slack:export` with `--slack`)
* `<name>-stacked.pdf`     — generation stacked by carrier, with the demand target
* `<name>-observation.pdf` — per-unit power as a grid of per-carrier subplots
* `<name>-balance.pdf`     — total generation vs demand, plus the residual
* `<name>-cost.pdf`        — dispatch cost per timestep
* `<name>-meta.json`       — run dimensions (all CLI options, scenario, topology),
  total cost, balance gap, rounds/iterations and stop reason, delivered messages,
  wall time, loss diagnostics (degraded units, hours with missing loads, ADMM
  missed answers) and the git SHA (+ dirty flag) of each package

To run all seven scenarios against the PyPSA-Eur extracts and compare total
costs, use `../run_all_scenarios.sh` from the repo root (writes
`results/cost_comparison.csv` via `compare_costs.py`).

To run the same batch on the University of Oldenburg HPC cluster (ROSA) as SLURM
job arrays, follow [`hpc/README.md`](hpc/README.md).

## Package layout

```
src/energy_scheduling_benchmark/
├── networks.py         # PyPSA network loaders (examples, files, PYPOWER, toy fixture)
├── dispatch/           # Pyomo economic dispatch (single-step + full-horizon LP)
├── plotting.py         # matplotlib plotting (replacement for CairoMakie)
└── scenarios/
    ├── _common.py              # shared scenario skeleton (argparser, roles,
    │                           #  balance verification, slack, cost computation)
    ├── central_dispatch.py     # central full-horizon LP (lower bound)
    ├── consensus.py            # averaging consensus
    ├── admm.py                 # exchange ADMM
    ├── diffusion.py            # classical diffusion
    ├── exact_diffusion.py      # exact (bias-corrected) diffusion
    ├── fdgdm.py                # fast distributed gradient descent
    └── deed_admm.py            # DEED-ADMM
```

`PyPSABehavior`, `SchedulingBehavior`, `ComponentRef`, `PowerUpdateInfo`, and the component-type
constants are defined in `mango-energy-environments` and re-exported here for
convenience.

## Tests

```bash
pytest tests/
```
