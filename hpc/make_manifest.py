#!/usr/bin/env python3
"""Write the run manifest of a benchmark sweep (TODO Phase 6, decision Q9).

One TSV per scenario under ``--out`` (default ``hpc-manifests/<stamp>/``), one line
per run: ``run_id``, ``network`` and the scenario's CLI arguments. Every run gets a
unique ``--name-base`` inside ``results/<network-stem>/``; ``compare_costs.py``
groups runs by the dimensions recorded in their ``-meta.json``, not by file name.

Full product per network: 7 scenarios × topology {complete, ring, grid} × cost
model {linear, quadratic} × slack {off, on} × simulated days {1, 3, 7} × loss rate
{0, 0.01, 0.05, 0.1} × seeds. Q9: loss 0 runs once plus one determinism repeat
(same seed); every loss > 0 cell runs with ``--seeds`` seeds (default 5).
Pruned (invalid, not just uninteresting): central dispatch has no communication
topology and no message loss, so it runs on ``complete`` at loss 0 only.

Usage (from anywhere)::

    hpc/make_manifest.py --all                       # every .nc in resources/networks/
    hpc/make_manifest.py base_s_2_elec_de-at-storage-2023.nc --days 1 --loss-rates 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

ESB_DIR = Path(__file__).resolve().parents[1]
NETWORKS_DIR = ESB_DIR / "resources" / "networks"

SCENARIOS = (
    "central-dispatch",
    "consensus",
    "admm",
    "diffusion",
    "exact-diffusion",
    "fdgdm",
    "deed-admm",
)
TOPOLOGIES = ("complete", "ring", "grid")
COST_MODELS = ("linear", "quadratic")
SLACK = ("off", "on")
DAYS = (1, 3, 7)
LOSS_RATES = (0.0, 0.01, 0.05, 0.1)


def runs(
    networks: Sequence[str],
    *,
    scenarios: Sequence[str] = SCENARIOS,
    topologies: Sequence[str] = TOPOLOGIES,
    cost_models: Sequence[str] = COST_MODELS,
    slack: Sequence[str] = SLACK,
    days: Sequence[int] = DAYS,
    loss_rates: Sequence[float] = LOSS_RATES,
    seeds: int = 5,
    determinism_repeat: bool = True,
) -> Iterator[dict]:
    """Every run of the sweep as ``{scenario, network, run_id, args}``."""
    for network, scenario in itertools.product(networks, scenarios):
        central = scenario == "central-dispatch"
        for topology, cost_model, slack_on, n_days, loss in itertools.product(
            ("complete",) if central else topologies,
            cost_models,
            slack,
            days,
            (0.0,) if central else loss_rates,
        ):
            if loss == 0.0:
                draws = [(0, "")] + ([(0, "-rep")] if determinism_repeat else [])
            else:
                draws = [(seed, "") for seed in range(1, seeds + 1)]
            for seed, suffix in draws:
                run_id = (
                    f"{scenario}__{topology}_{cost_model}_slack-{slack_on}_{n_days}d"
                    f"_loss{loss:g}_s{seed}{suffix}"
                )
                args = [
                    "--topology", topology,
                    "--cost-model", cost_model,
                    "--simulate-days", str(n_days),
                    "--loss-rate", f"{loss:g}",
                    "--seed", str(seed),
                ]  # fmt: skip
                if slack_on == "on":
                    args.append("--slack")
                yield {
                    "scenario": scenario,
                    "network": network,
                    "run_id": run_id,
                    "args": args,
                }


def write(manifest_dir: Path, all_runs: Sequence[dict]) -> dict[str, int]:
    """One ``<scenario>.tsv`` per scenario; returns the line count per scenario."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for scenario in SCENARIOS:
        rows = [r for r in all_runs if r["scenario"] == scenario]
        if not rows:
            continue
        with open(manifest_dir / f"{scenario}.tsv", "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            for r in rows:
                writer.writerow([r["run_id"], r["network"], *r["args"]])
        counts[scenario] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "networks", nargs="*", help="Network files in resources/networks/ (or `toy`)."
    )
    parser.add_argument(
        "--all", action="store_true", help="Every .nc in resources/networks/."
    )
    parser.add_argument("--out", type=Path, default=None, help="Manifest directory.")
    parser.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS
    )
    parser.add_argument(
        "--topologies", nargs="+", default=list(TOPOLOGIES), choices=TOPOLOGIES
    )
    parser.add_argument(
        "--cost-models", nargs="+", default=list(COST_MODELS), choices=COST_MODELS
    )
    parser.add_argument("--slack", nargs="+", default=list(SLACK), choices=SLACK)
    parser.add_argument("--days", nargs="+", type=int, default=list(DAYS))
    parser.add_argument("--loss-rates", nargs="+", type=float, default=list(LOSS_RATES))
    parser.add_argument(
        "--seeds", type=int, default=5, help="Seeds per loss > 0 cell (Q9: 5)."
    )
    parser.add_argument(
        "--no-determinism-repeat",
        action="store_true",
        help="Skip the second loss-0 run (Q9 repeats it once to check determinism).",
    )
    args = parser.parse_args(argv)

    networks = list(args.networks)
    if args.all:
        networks += sorted(p.name for p in NETWORKS_DIR.glob("*.nc"))
    if not networks:
        parser.error("no networks: pass network files, `toy` or --all")
    for name in networks:
        if (
            name != "toy"
            and not (NETWORKS_DIR / name).is_file()
            and not Path(name).is_file()
        ):
            parser.error(f"network not found: {name}")

    all_runs = list(
        runs(
            networks,
            scenarios=args.scenarios,
            topologies=args.topologies,
            cost_models=args.cost_models,
            slack=args.slack,
            days=args.days,
            loss_rates=args.loss_rates,
            seeds=args.seeds,
            determinism_repeat=not args.no_determinism_repeat,
        )
    )
    out = args.out or (
        ESB_DIR.parent / "hpc-manifests" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    counts = write(out, all_runs)
    print(f"Manifest: {out}")
    for scenario, n in counts.items():
        print(f"  {scenario:18s} {n:6d} runs")
    print(
        f"  {'total':18s} {sum(counts.values()):6d} runs over {len(networks)} network(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
