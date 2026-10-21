"""Tuned algorithm parameters, kept in one TOML file instead of code defaults.

Parameters that are *tuned* rather than derived (iteration budgets, ADMM ρ,
consensus α / leader damping, DEED-ADMM γ, stop tolerances, ...) live in
``tuned_parameters.toml`` next to this module, so every result's settings are
visible in one place and reproducible. Layout (all sections optional; keys are
the scenario CLI flags' ``dest`` names, e.g. ``max_iter`` for ``--max-iter``)::

    [consensus]                         # scenario defaults, every network
    max_iter = 500

    [consensus.topology.ring]           # per topology
    max_iter = 3000

    [network."base_s_2_elec_de-at-storage-2023".consensus]          # per network
    alpha = 0.2

    [network."base_s_2_elec_de-at-storage-2023".consensus.topology.ring]
    max_iter = 1500

Later sections override earlier ones in exactly that order; an explicit CLI
flag overrides them all. Scenario keys are the module names (``deed_admm``),
network keys the ``--network`` value reduced to its file stem (``toy``,
``base_s_2_elec_de-at-storage-2023``).
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

#: The packaged parameter file.
DEFAULT_PATH = Path(__file__).with_name("tuned_parameters.toml")


def network_key(network: str) -> str:
    """``--network`` reduced to the key used in the file (file stem or name)."""
    path = Path(str(network))
    if path.suffix in {".nc", ".h5", ".xlsx"}:
        return path.stem
    return path.name or str(network)


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the parameter file (*path* or :data:`DEFAULT_PATH`)."""
    with open(path or DEFAULT_PATH, "rb") as f:
        return tomllib.load(f)


def _flat(section: Any) -> dict[str, Any]:
    """Plain values of a section (sub-tables such as ``topology`` dropped)."""
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in section.items() if not isinstance(v, dict)}


def resolve(
    config: dict[str, Any], scenario: str, network: str, topology: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parameters for one run and, per parameter, the section it came from."""
    net = config.get("network", {}).get(network_key(network), {})
    scen = config.get(scenario, {})
    net_scen = net.get(scenario, {}) if isinstance(net, dict) else {}
    layers = [
        (scenario, scen),
        (f"{scenario}.topology.{topology}", scen.get("topology", {}).get(topology, {})),
        (f'network."{network_key(network)}".{scenario}', net_scen),
        (
            f'network."{network_key(network)}".{scenario}.topology.{topology}',
            net_scen.get("topology", {}).get(topology, {}) if net_scen else {},
        ),
    ]
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, section in layers:
        for key, value in _flat(section).items():
            values[key] = value
            sources[key] = name
    return values, sources


def apply(
    parser: argparse.ArgumentParser, argv: list[str] | None, default_name_base: str
) -> argparse.Namespace:
    """Parse *argv* with the tuned parameters of *default_name_base* as defaults.

    Explicit flags win over the file. ``args.tuned_from`` maps every value
    taken from the file to the section it came from (see :func:`resolve`).
    """
    # Tuned parameters: a first parse finds the network/topology, the file's
    # values become the parser defaults, and the second parse lets explicit
    # flags win.
    first = parser.parse_args(argv)
    config = load(first.tuned_params)
    tuned, tuned_from = resolve(
        config, default_name_base, first.network, first.topology
    )
    source = first.tuned_params or DEFAULT_PATH
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(tuned) - set(actions))
    if unknown:
        parser.error(
            f"{source}: unknown parameter(s) {unknown} for {default_name_base}"
        )
    # set_defaults() skips argparse's own checks: apply each flag's type and
    # choices to the file's values here.
    for key, value in list(tuned.items()):
        action = actions[key]
        if action.type is not None:
            try:
                value = action.type(str(value))
            except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
                parser.error(f"{source}: {key} = {tuned[key]!r}: {exc}")
        elif isinstance(action.const, bool) and not isinstance(value, bool):
            parser.error(f"{source}: {key} must be true or false, got {value!r}")
        if action.choices is not None and value not in action.choices:
            parser.error(
                f"{source}: {key} = {value!r} is not one of {list(action.choices)}"
            )
        tuned[key] = value
    parser.set_defaults(**tuned)
    args = parser.parse_args(argv)
    # Flags given on the command line: parse once more with every default
    # suppressed, so only explicit flags appear.
    saved = [(action, action.default) for action in parser._actions]
    saved_parser_defaults = dict(parser._defaults)
    parser._defaults.clear()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    try:
        explicit = set(vars(parser.parse_args(argv)))
    finally:
        for action, default in saved:
            action.default = default
        parser._defaults.update(saved_parser_defaults)
    args.tuned_from = {key: tuned_from[key] for key in tuned if key not in explicit}
    return args
