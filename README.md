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
sibling `pypsa-eur/` workflow).

## Running the scenarios

Seven scenario entry points are installed:

| Entry point            | Algorithm                                                          |
|------------------------|--------------------------------------------------------------------|
| `esb-central-dispatch` | Centralised Pyomo/HiGHS LP dispatch (baseline)                     |
| `esb-consensus`        | Averaging consensus with gradient (Jian et al. 2020)               |
| `esb-admm`             | Exchange ADMM with proximal actors (Boyd et al. 2011)              |
| `esb-diffusion`        | Adapt-then-combine diffusion (Ces et al. 2025)                     |
| `esb-exact-diffusion`  | Bias-corrected exact diffusion (Yuan et al. 2018 / Ces et al. 2025)|
| `esb-fdgdm`            | Fast distributed gradient descent (Bai et al. 2022)                |
| `esb-deed-admm`        | DEED-ADMM (Zhu et al. 2025)                                        |

All seven share the same CLI flags: `--network`, `--name-base`,
`--simulate-days`, `--delay-s` (comms delay), `--loss-percent` (packet loss),
`--log-level`.  Every distributed algorithm except averaging consensus
requires lossless transport and refuses to run with `--loss-percent != 0`.

```bash
# Toy 5-bus fixture (default network)
esb-consensus --name-base consensus_out

# Any PyPSA example by name
esb-central-dispatch --network ac-dc-meshed --name-base ac_dc_central

# Or load from a file / run as a module
python -m energy_scheduling_benchmark.scenarios.diffusion \
    --network /path/to/mynet.nc --name-base mynet_diffusion
```

Each run produces:

* `<name>-observation.pdf` — grid of mango recordings
* `<name>-stacked.pdf`     — stacked-area dispatch plot with demand target
* `<name>-cost.pdf`        — cost per timestep
* `<name>-df.csv`          — per-agent timeseries incl. a `cost:total` column

To run all seven scenarios against the PyPSA-Eur extracts and compare total
costs, use `../run_all_scenarios.sh` from the repo root (writes
`results/cost_comparison.csv` via `compare_costs.py`).

## Package layout

```
src/energy_scheduling_benchmark/
├── networks.py         # PyPSA network loaders (examples, files, PYPOWER, toy fixture)
├── dispatch/           # Pyomo economic dispatch (central LP)
├── plotting.py         # matplotlib plotting (replacement for CairoMakie)
└── scenarios/
    ├── _common.py              # shared scenario skeleton (argparser, roles,
    │                           #  balance verification, cost computation)
    ├── central_dispatch.py     # central LP baseline
    ├── consensus.py            # averaging consensus
    ├── admm.py                 # exchange ADMM
    ├── diffusion.py            # classical diffusion
    ├── exact_diffusion.py      # exact (bias-corrected) diffusion
    ├── fdgdm.py                # fast distributed gradient descent
    └── deed_admm.py            # DEED-ADMM
```

`PyPSABehavior`, `SchedulingBehavior`, `ComponentRef`, and the component-type
constants are defined in `mango-energy-environments` and re-exported here for
convenience.

## Tests

```bash
pytest tests/
```
