# energy-scheduling-benchmark

Multi-agent energy scheduling benchmark scenarios — Python port of
[`EnergySchedulingBenchmark.jl`](./EnergySchedulingBenchmark.jl).

The package is built on top of:

* [mango-agents](../mango) — the agent framework
* [mango-energy-environments](../mango-energy-environments) — mango environment abstractions
* [distributed-resource-optimization](../mango-optimization) — COHDA / ADMM / averaging consensus
* [PyPSA](https://pypsa.org) — power-system data model (replaces `PowerSystems.jl`)
* [Pyomo](https://pyomo.readthedocs.io) — LP modelling layer (replaces JuMP); any
  Pyomo-compatible LP solver works, but HiGHS (via `highspy`) is preferred.

## Install

```bash
pip install -e .
```

The `pyproject.toml` wires the three sibling mango packages in as editable
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
| PYPOWER IEEE test case              | `load_pypower_case(118)`                        |
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

## Running the scenarios

Two scenarios mirror the Julia ones:

```bash
# Toy 5-bus fixture (default)
python -m energy_scheduling_benchmark.scenarios.consensus \
    --name-base consensus_lossless --loss-percent 0.0

# Any PyPSA example by name
python -m energy_scheduling_benchmark.scenarios.central_dispatch \
    --network ac-dc-meshed --name-base ac_dc_central

# Or load from a file
python -m energy_scheduling_benchmark.scenarios.consensus \
    --network /path/to/mynet.nc --name-base mynet_consensus
```

Or via the installed entry points:

```bash
esb-consensus --network storage-hvdc --name-base storage_consensus
esb-central-dispatch --network toy --name-base central_agent_withlosses
```

Each run produces:

* `<name>-observation.pdf` — grid of mango recordings
* `<name>-stacked.pdf`     — stacked-area dispatch plot
* `<name>-df.csv`          — per-agent timeseries

## Package layout

```
src/energy_scheduling_benchmark/
├── environment/        # PyPSABehavior — mango Behavior backed by PyPSA
├── networks.py         # PyPSA network loaders (examples, files, PYPOWER)
├── dispatch/           # Pyomo economic dispatch
├── plotting.py         # matplotlib replacement for CairoMakie
└── scenarios/
    ├── consensus.py            # distributed consensus
    └── central_dispatch.py     # central LP
```
