"""Shared boilerplate for the distributed-optimization scenarios.

Consensus, ADMM, diffusion, FDGDM, and DEED-ADMM all follow the same shape:
load agents push per-timestep demand to a leader, the leader kicks off a
distributed optimization run exactly once, and once every generator reports
back it applies the resulting per-timestep schedule as the simulation clock
advances. This module holds that shared shape so each scenario only needs to
supply the algorithm-specific pieces (actor construction, start-message
payload, and the attribute path to read the finished schedule from).


The code lives in ``_model`` (units, demand, cost model), ``_world`` (scenario and
world setup, topology, loss), ``_roles`` (messages and roles), ``_outputs`` (cost,
meta and output files) and ``_cli`` (argparser, ``run_scenario_main``); this module
re-exports them so ``from ._common import ...`` keeps working.
"""

from __future__ import annotations

from energy_scheduling_benchmark.dispatch import SlackSpec  # noqa: F401
from energy_scheduling_benchmark.networks import (  # noqa: F401
    ScenarioData,
    build_toy_network,
    load_scenario,
)
from energy_scheduling_benchmark.scenarios._cli import (  # noqa: F401
    _bounded_float,
    _complete_meta,
    _non_negative_int,
    build_scenario_argparser,
    run_scenario_main,
)
from energy_scheduling_benchmark.scenarios._model import (  # noqa: F401
    COST_MODELS,
    QUADRATIC_K,
    DemandHorizon,
    EpsilonScaling,
    StorageParams,
    apply_cost_model,
    build_demand_horizon,
    build_p_max_vec,
    build_unit_bounds,
    capacity_scaled_epsilon,
    collect_generator_refs,
    filter_and_cache_statics,
    price_sensitivity,
    quadratic_cost_coefficient,
    thermal_cost_quad,
)
from energy_scheduling_benchmark.scenarios._outputs import (  # noqa: F401
    OUTPUT_SUFFIXES,
    _git_state,
    _run_meta,
    _slack_columns,
    build_group_map,
    carrier_style,
    compute_overall_cost,
    compute_slack,
    file_sha256,
    write_scenario_outputs,
)
from energy_scheduling_benchmark.scenarios._roles import (  # noqa: F401
    GeneratorResultRole,
    KickOff,
    KickOffAck,
    OptimizationFinishedInfo,
    PowerLoadAggregator,
    PowerLoadInfo,
    PowerLoadMonitoring,
    ScheduleRequest,
    generator_agent,
    make_finish_callback,
)
from energy_scheduling_benchmark.scenarios._world import (  # noqa: F401
    _REFS_BY_WORLD,
    AgentError,
    CommunicationTopology,
    LossTolerance,
    _AgentErrorWatch,
    _clip_scenario,
    build_world,
    install_topology,
    loss_tolerance,
    participant_graph,
    register_component_agent,
    resolve_loss_rate,
    resolve_scenario,
    run_simulation,
)

__all__ = [
    "build_toy_network",
    "load_scenario",
    "resolve_scenario",
    "build_world",
    "register_component_agent",
    "resolve_loss_rate",
    "_clip_scenario",
    "compute_overall_cost",
    "compute_slack",
    "SlackSpec",
    "build_group_map",
    "carrier_style",
    "build_scenario_argparser",
    "LossTolerance",
    "loss_tolerance",
    "generator_agent",
    "filter_and_cache_statics",
    "collect_generator_refs",
    "build_demand_horizon",
    "DemandHorizon",
    "build_p_max_vec",
    "build_unit_bounds",
    "capacity_scaled_epsilon",
    "EpsilonScaling",
    "StorageParams",
    "write_scenario_outputs",
    "run_scenario_main",
    "PowerLoadInfo",
    "OptimizationFinishedInfo",
    "ScheduleRequest",
    "KickOff",
    "KickOffAck",
    "PowerLoadMonitoring",
    "PowerLoadAggregator",
    "GeneratorResultRole",
    "make_finish_callback",
]
