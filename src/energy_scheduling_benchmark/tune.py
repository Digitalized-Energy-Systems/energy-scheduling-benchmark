"""esb-tune: sweep consensus step sizes for one network × topology (plan §7 I15).

Runs central once per window, then consensus for every candidate (decaying: α₀ grid; constant:
κ × γ grid) on each window. Among the candidates that stop "converged" on every window it picks
the smallest worst-window |cost gap| + relative imbalance vs central (ties: fewer iterations)
and writes it into ``resources/tuned_parameters.toml`` as
``[consensus.<network>.<topology>.sweep]``; the note also records the best of the other variant.

    esb-tune --network lower_saxony_2030 --topology ring --start 2019-01-14 --start 2019-06-10
"""

from __future__ import annotations

import argparse
import asyncio
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
    """|cost gap| + imbalance / demand (both relative); None unless the run converged."""
    if meta["stop_reason"] != "converged" or meta["total_cost"] is None:
        return None
    gap = meta["total_cost"] / central["total_cost"] - 1
    imbalance = meta["imbalance"]["shortfall_mwh"] + meta["imbalance"]["surplus_mwh"]
    return abs(gap) + imbalance / demand_mwh


def write_section(name: str, values: dict) -> None:
    """Replace (or append) table ``[name]`` in the parameter file; other text is kept."""
    text = consensus.PARAMETERS.read_text()
    text = re.sub(rf"\n\[{re.escape(name)}\]\n(?:[^\[\n].*\n|\n(?!\[))*", "\n", text)
    lines = [f"[{name}]"] + [f"{k} = {json.dumps(v)}" for k, v in values.items()]
    consensus.PARAMETERS.write_text(
        text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    )


def _grid(args: argparse.Namespace) -> list[dict]:
    decaying = [{"steps": "decaying", "alpha0": a} for a in _floats(args.alpha0)]
    constant = [
        {"steps": "constant", "kappa": k, "gamma": g}
        for k, g in itertools.product(_floats(args.kappa), _floats(args.gamma))
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
        for i, values in enumerate(_grid(args)):
            sets = [
                x
                for k, v in (values | {"max_iter": args.max_iter}).items()
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
                    values, start, meta["stop_reason"], meta["iterations"], scores[-1]
                )
            if None in scores:
                continue
            candidate = (max(scores), max(iterations), values, gaps)
            variant = values["steps"]
            if variant not in best or candidate[:2] < best[variant][:2]:
                best[variant] = candidate
    if not best:
        print("no candidate converged on every window")
        return None
    winner = min(best, key=lambda v: best[v][:2])
    values = best[winner][2] | {
        "note": "esb-tune best: "
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
    parser.add_argument("--alpha0", default="0.1,0.3,1,3,10,30", help="decaying grid")
    parser.add_argument("--kappa", default="0.003,0.01,0.03,0.1", help="constant grid")
    parser.add_argument("--gamma", default="1,4,16", help="constant grid")
    parser.add_argument("--max-iter", type=int, default=3000, help="per candidate")
    parser.add_argument(
        "--no-write", dest="write", action="store_false", help="only print the best set"
    )
    asyncio.run(sweep(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
