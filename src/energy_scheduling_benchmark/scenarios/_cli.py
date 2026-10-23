"""The shared scenario command line: argparser and ``run_scenario_main``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from mango_energy_environments.environments.scheduling.topology import (
    TOPOLOGIES,
)

from energy_scheduling_benchmark import (
    tuned_params,
)
from energy_scheduling_benchmark.dispatch import SlackSpec
from energy_scheduling_benchmark.networks import (
    available_examples,
)
from energy_scheduling_benchmark.scenarios._model import (
    COST_MODELS,
)
from energy_scheduling_benchmark.scenarios._outputs import (
    OUTPUT_SUFFIXES,
    _git_state,
    file_sha256,
)
from energy_scheduling_benchmark.scenarios._world import (
    resolve_scenario,
)


def _non_negative_int(text: str) -> int:
    """argparse ``type=`` for an integer ``>= 0``."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _bounded_float(lo: float, hi: float) -> Callable[[str], float]:
    """argparse ``type=`` that parses a float and rejects values outside ``[lo, hi]``."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"must be in [{lo:g}, {hi:g}], got {value:g}"
            )
        return value

    return parse


def build_scenario_argparser(
    description: str | None,
    *,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser shared by all scenario entry points.

    *extra_args*, if given, is called with the parser to add algorithm-specific
    flags (e.g. DEED-ADMM's ``--gamma``/``--max-iter``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--network",
        type=str,
        default="toy",
        help=(
            "Network source. Either 'toy' (the built-in 5-bus fixture), "
            "a PyPSA example name "
            f"({', '.join(available_examples())}), or a path to a "
            ".nc/.h5/.xlsx file or CSV folder."
        ),
    )
    parser.add_argument("--delay-s", type=float, default=0.02, help="Comms delay (s).")
    parser.add_argument(
        "--loss-rate",
        type=_bounded_float(0.0, 1.0),
        default=0.0,
        help=(
            "Probability that a message is lost, as a fraction in [0, 1] "
            "(e.g. 0.05 for 5%%; default 0)."
        ),
    )
    parser.add_argument(
        "--tuned-params",
        type=Path,
        default=None,
        help=(
            "TOML file with the tuned algorithm parameters (iteration budgets, rho, "
            "alpha, ...) per scenario, topology and network; default: the packaged "
            "tuned_parameters.toml. Explicit flags override it."
        ),
    )
    parser.add_argument(
        "--topology",
        choices=TOPOLOGIES,
        default="complete",
        help=(
            "Communication graph: 'complete' (default, everyone talks to everyone) "
            "or 'ring' (algorithm agents in a cycle ordered by id). Loads attach "
            "via their bus; every message travels the shortest path (hops x "
            "--delay-s, loss per hop)."
        ),
    )
    parser.add_argument(
        "--cost-model",
        choices=COST_MODELS,
        default="linear",
        help=(
            "Generator cost: 'linear' (PyPSA marginal cost, default) or 'quadratic' "
            "(+ c2·P² with c2 = 0.11·marginal_cost/p_nom, decision Q17)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the message-loss draws (default 0), so lossy runs are reproducible.",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=None,
        help=(
            "Seconds (simulation time) a round waits for missing messages before "
            "advancing with the last received values (default: 3 x --delay-s under "
            "loss; none without loss, so lossless rounds wait for every neighbour as "
            "in the papers)."
        ),
    )
    parser.add_argument(
        "--result-poll-s",
        type=float,
        default=300.0,
        help=(
            "Interval (simulation seconds) between the leader's requests for "
            "results that never arrived (default 300)."
        ),
    )
    parser.add_argument(
        "--result-retries",
        type=int,
        default=12,
        help=(
            "Requests for a missing result before the run continues without it "
            "and is marked degraded (default 12)."
        ),
    )
    parser.add_argument(
        "--load-timeout",
        type=_bounded_float(0.0, float("inf")),
        default=None,
        help=(
            "Seconds (simulation time) after an hour starts before it is applied "
            "with incomplete load reports (missing loads count with their forecast; "
            "default 3 x --delay-s)."
        ),
    )
    parser.add_argument(
        "--max-missing-rounds",
        type=_non_negative_int,
        default=20,
        help=(
            "ADMM only: consecutive rounds a participant may miss before the "
            "coordinator stops the run as not converged (default 20; 0 disables)."
        ),
    )
    parser.add_argument("--name-base", type=str, default=default_name_base)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for the CSV/PDF outputs (created if missing).",
    )
    parser.add_argument("--simulate-days", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--slack",
        action="store_true",
        help=(
            "Add a slack node (the paper's point of common coupling) that "
            "imports any shortfall and exports any surplus, priced into "
            "cost:total and written as slack:import/slack:export columns. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--slack-import-price",
        type=float,
        default=SlackSpec().import_price,
        help="Price per MWh the slack supplies, €/MWh (default: %(default)s).",
    )
    parser.add_argument(
        "--slack-export-price",
        type=float,
        default=SlackSpec().export_price,
        help="Price per MWh the slack absorbs, €/MWh (default: %(default)s).",
    )
    parser.add_argument(
        "--balance-tol",
        type=float,
        default=0.01,
        help=(
            "Max allowed per-timestep |generation - demand| / demand "
            "before a finished run is flagged as not converged (default: "
            "0.01, i.e. 1%%)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with an error instead of writing CSV/PDF outputs if "
            "the power-balance check fails (default: log a warning and "
            "write the outputs anyway)."
        ),
    )
    if extra_args is not None:
        extra_args(parser)
    return parser


def run_scenario_main(
    execute_test_case: Callable[..., Any],
    *,
    doc: str | None,
    default_name_base: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_kwargs: Sequence[str] = (),
    argv: list[str] | None = None,
) -> None:
    """Shared ``main()`` body: parse CLI args, resolve the scenario, run it.

    *extra_kwargs* names CLI args added via *extra_args* to forward to
    *execute_test_case* verbatim, e.g. ``("gamma", "max_iter")`` for
    DEED-ADMM's ``--gamma``/``--max-iter``.
    """
    parser = build_scenario_argparser(
        doc, default_name_base=default_name_base, extra_args=extra_args
    )
    args = tuned_params.apply(parser, argv, default_name_base)

    slack = None
    if args.slack:
        if args.strict:
            parser.error(
                "--strict cannot be combined with --slack: the slack closes "
                "every power-balance gap, so there is nothing left to enforce."
            )
        slack = SlackSpec(
            import_price=args.slack_import_price,
            export_price=args.slack_export_price,
        )

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenario = resolve_scenario(args.network, simulate_days=args.simulate_days)

    kwargs = {name: getattr(args, name) for name in extra_kwargs}
    # Every output path is derived from name_base, so prefixing it with the
    # output directory routes all CSV/PDF writes there.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A run that fails (crash, timeout, --strict) must not leave the previous
    # run's outputs behind under its name: compare_costs.py would read them.
    for suffix in OUTPUT_SUFFIXES:
        (args.out_dir / f"{args.name_base}-{suffix}").unlink(missing_ok=True)
    started = perf_counter()
    asyncio.run(
        execute_test_case(
            scenario=scenario,
            delay_s=args.delay_s,
            loss_rate=args.loss_rate,
            seed=args.seed,
            round_timeout=args.round_timeout,
            result_poll_s=args.result_poll_s,
            result_retries=args.result_retries,
            load_timeout=args.load_timeout,
            max_missing_rounds=args.max_missing_rounds or None,
            name_base=str(args.out_dir / args.name_base),
            simulate_days=args.simulate_days,
            cost_model=args.cost_model,
            topology=args.topology,
            slack=slack,
            balance_tol=args.balance_tol,
            strict=args.strict,
            **kwargs,
        )
    )
    _complete_meta(
        args.out_dir / f"{args.name_base}-meta.json",
        scenario=default_name_base,
        args=args,
        wall_time_s=perf_counter() - started,
    )


def _complete_meta(
    path: Path, *, scenario: str, args: argparse.Namespace, wall_time_s: float
) -> None:
    """Add run dimensions, git SHAs and wall time to a written meta sidecar."""
    if not path.exists():
        return  # the run failed before writing outputs (e.g. --strict)
    import distributed_resource_optimization
    import mango
    import mango_energy_environments

    import energy_scheduling_benchmark

    meta = json.loads(path.read_text(encoding="utf-8"))
    dims = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    dims.update(
        {
            "scenario": scenario,
        }
    )
    meta = {
        "dimensions": dims,
        **meta,
        "wall_time_s": wall_time_s,
        "inputs": {"network_sha256": file_sha256(args.network)},
        "git": {
            m.__name__: _git_state(m)
            for m in (
                energy_scheduling_benchmark,
                distributed_resource_optimization,
                mango_energy_environments,
                mango,
            )
        },
    }
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
