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

## Running the scenarios

Two scenarios mirror the Julia ones:

```bash
# Distributed consensus economic dispatch (200 iterations)
python -m energy_scheduling_benchmark.scenarios.consensus \
    --name-base consensus_lossless --loss-percent 0.0

# Central LP dispatch (Pyomo + HiGHS)
python -m energy_scheduling_benchmark.scenarios.central_dispatch \
    --name-base central_agent_lossless --loss-percent 0.0
```

Or via the installed entry points:

```bash
esb-consensus --name-base consensus_withlosses
esb-central-dispatch --name-base central_agent_withlosses
```

Each run produces:

* `<name>-observation.pdf` — grid of mango recordings
* `<name>-stacked.pdf`     — stacked-area dispatch plot
* `<name>-df.csv`          — per-agent timeseries

## Package layout

```
src/energy_scheduling_benchmark/
├── environment/        # PyPSABehavior — mango Behavior backed by PyPSA
├── dispatch/           # Pyomo economic dispatch
├── plotting.py         # matplotlib replacement for CairoMakie
└── scenarios/
    ├── consensus.py            # distributed consensus
    └── central_dispatch.py     # central LP
```
