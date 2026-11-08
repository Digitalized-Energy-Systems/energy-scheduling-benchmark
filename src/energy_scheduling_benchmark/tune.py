"""esb-tune: sweep consensus step sizes for one network × topology (plan §7 I15–I19).

Runs central once per window, then consensus for every candidate (decaying: α₀ × β₀ grid;
constant: κ × γ grid) on each window. α₀ and κ are factors of 1 / (the largest bus price
sensitivity Σ 1/(2c₂)), because constant steps are only stable below 2 / that sensitivity.
Each candidate runs a fixed budget (``--max-iter``) with the stop rule off (KH12 style), so
small stable steps are not cut short by the step-change stop. The candidate with the smallest
worst-window |cost gap| + relative imbalance vs central wins and is written into
``resources/tuned_parameters.toml`` as ``[consensus.<network>.<topology>.sweep]`` — only if its
|gap| < ``--max-gap`` (1 %) on every window; the note also records the best of the other variant.

    esb-tune --network lower_saxony_2030 --topology ring --start 2019-01-14 --start 2019-06-10
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import itertools
import json
import re
import tempfile

from mango_energy_environments import LOAD, load_network

from . import consensus, datasets
from .run import build_parser, run, window


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


async def _meta(argv: list[str]) -> dict:
    out = await run(build_parser().parse_args(argv))
    return json.loads((out / "meta.json").read_text())


def score(meta: dict, central: dict, demand_mwh: float) -> float | None:
    """|cost gap| + imbalance / demand (both relative); None unless the run used its whole
    budget (stop rule off, so "max_iter"; "stalled" = diverged)."""
    if meta["stop_reason"] != "max_iter" or meta["total_cost"] is None:
        return None
    gap = meta["total_cost"] / central["total_cost"] - 1
    imbalance = meta["imbalance"]["shortfall_mwh"] + meta["imbalance"]["surplus_mwh"]
    return abs(gap) + imbalance / demand_mwh


def write_section(name: str, values: dict) -> None:
    """Replace (or append) table ``[name]`` in the parameter file; other text is kept.

    Locked, so parallel sweeps (one per topology) do not lose each other's sections.
    """
    with open(consensus.PARAMETERS, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        text = f.read()
        text = re.sub(
            rf"\n\[{re.escape(name)}\]\n(?:[^\[\n].*\n|\n(?!\[))*", "\n", text
        )
        lines = [f"[{name}]"] + [f"{k} = {json.dumps(v)}" for k, v in values.items()]
        f.seek(0)
        f.write(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n")
        f.truncate()


def max_sensitivity(network) -> float:
    """Largest bus price sensitivity Σ 1/(2c₂) (MW per €/MWh): constant steps need
    α < 2 / it at that bus, so the α₀ / κ grids are factors of its inverse."""
    per_bus: dict[str, float] = {}
    for c in network.components.values():
        if c.kind != LOAD:
            per_bus[c.bus] = per_bus.get(c.bus, 0.0) + 1 / (2 * c.cost[1])
    return max(per_bus.values())


def _grid(args: argparse.Namespace, sensitivity: float) -> list[dict]:
    """α₀ and κ = factor / *sensitivity* (rounded to 4 digits); β₀, γ as given."""

    def scaled(text):
        return [float(f"{f / sensitivity:.4g}") for f in _floats(text)]

    decaying = [
        {"steps": "decaying", "alpha0": a, "beta0": b}
        for a, b in itertools.product(scaled(args.alpha0), _floats(args.beta0))
    ]
    constant = [
        {"steps": "constant", "kappa": k, "gamma": g}
        for k, g in itertools.product(scaled(args.kappa), _floats(args.gamma))
    ]
    return {"decaying": decaying, "constant": constant, "both": decaying + constant}[
        args.steps
    ]


def _note(variant: str, best: tuple, hours: int) -> str:
    s, iterations, values, gaps = best
    shown = ", ".join(f"{k}={v}" for k, v in values.items() if k != "steps")
    windows = "; ".join(f"{start}: {gap:+.4%}" for start, gap in gaps.items())
    return (
        f"{variant} {shown} — {hours} h gap {windows}, score {s:.2e}, {iterations} it."
    )


async def sweep(args: argparse.Namespace) -> dict | None:
    """A candidate counts only if it converges on every window; its score and iterations are
    the worst over the windows."""
    base = ["--network", args.network, "--topology", args.topology]
    base += ["--hours", str(args.hours)]
    raw = load_network(datasets.prepared_path(args.network))
    sensitivity = max_sensitivity(raw)
    print(f"max bus dP/dλ = {sensitivity:.4g} MW per €/MWh", flush=True)
    starts = args.start or [None]
    best: dict[str, tuple] = {}  # variant -> (score, iterations, values, {start: gap})
    with tempfile.TemporaryDirectory() as tmp:
        windows = {}
        for w, start in enumerate(starts):
            common = base + ([] if start is None else ["--start", start])
            net = window(raw, start, args.hours)
            demand = sum(
                c.p_max.sum() for c in net.components.values() if c.kind == LOAD
            )
            central = await _meta(
                ["--algorithm", "central", *common, "--out", f"{tmp}/central{w}"]
            )
            windows[str(net.time_index[0])] = (common, central, float(demand))
        for i, values in enumerate(_grid(args, sensitivity)):
            sets = [
                x
                # stop rule off (max |Δλ| < −1 never holds): every candidate gets the same
                # budget; small stable steps would otherwise trip the step-change stop at once
                for k, v in (
                    values | {"max_iter": args.max_iter, "eps_lambda": -1}
                ).items()
                for x in ("--set", f"{k}={v}")
            ]
            scores, iterations, gaps = [], [], {}
            for w, (start, (common, central, demand)) in enumerate(windows.items()):
                meta = await _meta(
                    ["--algorithm", "consensus", "--tuning", "paper", *common, *sets]
                    + ["--out", f"{tmp}/{i}_{w}"]
                )
                scores.append(score(meta, central, demand))
                iterations.append(meta["iterations"] or args.max_iter)
                if meta["total_cost"] is not None:
                    gaps[start] = meta["total_cost"] / central["total_cost"] - 1
                print(
                    values,
                    start,
                    meta["stop_reason"],
                    meta["iterations"],
                    scores[-1],
                    flush=True,
                )
            if None in scores:
                continue
            candidate = (max(scores), max(iterations), values, gaps)
            variant = values["steps"]
            if variant not in best or candidate[:2] < best[variant][:2]:
                best[variant] = candidate
    usable = {
        v: b for v, b in best.items() if max(map(abs, b[3].values())) < args.max_gap
    }
    if not usable:
        print(f"no usable candidate (|gap| < {args.max_gap:.1%} on every window)")
        for v, b in best.items():
            print("best", v, _note(v, b, args.hours))
        return None
    best = usable
    winner = min(best, key=lambda v: best[v][:2])
    values = best[winner][2] | {
        "note": f"esb-tune (grid × 1/{sensitivity:.4g} MW per €/MWh) best: "
        + " | ".join(
            _note(v, best[v], args.hours)
            for v in sorted(best, key=lambda v: v != winner)
        )
    }
    if args.write:
        write_section(f"consensus.{args.network}.{args.topology}.sweep", values)
    print("best:", values)
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", default="toy")
    parser.add_argument(
        "--topology", default="grid", choices=("grid", "complete", "ring")
    )
    parser.add_argument(
        "--steps", default="both", choices=("decaying", "constant", "both")
    )
    parser.add_argument(
        "--start", action="append", help="window start; repeat for several windows"
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument(
        "--alpha0", default="0.1,0.3,1,1.5", help="decaying grid, × 1/max bus dP/dλ"
    )
    parser.add_argument(
        "--beta0", default="0.003,0.01,0.03,0.1,0.2", help="decaying grid (× alpha0)"
    )
    parser.add_argument(
        "--kappa", default="0.1,0.3,1,1.5", help="constant grid, × 1/max bus dP/dλ"
    )
    parser.add_argument("--gamma", default="1,4,16", help="constant grid")
    parser.add_argument(
        "--max-iter", type=int, default=3000, help="fixed budget per candidate"
    )
    parser.add_argument(
        "--max-gap", type=float, default=0.01, help="write only if |gap| < this"
    )
    parser.add_argument(
        "--no-write", dest="write", action="store_false", help="only print the best set"
    )
    asyncio.run(sweep(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
