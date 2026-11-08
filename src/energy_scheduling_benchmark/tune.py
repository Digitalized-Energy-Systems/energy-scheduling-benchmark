"""esb-tune: sweep consensus step sizes for one network × topology (plan §7 I15).

Runs central once, then consensus for every candidate (decaying: α₀ grid; constant: κ × γ grid)
on a short window. Among the runs that stop "converged" it picks the smallest
|cost gap| + relative imbalance vs central (ties: fewer iterations) and writes it into
``resources/tuned_parameters.toml`` as ``[consensus.<network>.<topology>.sweep]``.

    esb-tune --network toy --topology ring --steps constant --hours 24
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


async def sweep(args: argparse.Namespace) -> dict | None:
    common = [
        "--network",
        args.network,
        "--topology",
        args.topology,
        "--hours",
        str(args.hours),
    ]
    if args.start:
        common += ["--start", args.start]
    network = window(
        load_network(datasets.prepared_path(args.network)), args.start, args.hours
    )
    demand_mwh = float(
        sum(c.p_max.sum() for c in network.components.values() if c.kind == LOAD)
    )
    if args.steps == "decaying":
        grid = [{"alpha0": a} for a in _floats(args.alpha0)]
    else:
        grid = [
            {"kappa": k, "gamma": g}
            for k, g in itertools.product(_floats(args.kappa), _floats(args.gamma))
        ]
    with tempfile.TemporaryDirectory() as tmp:
        central = await _meta(
            ["--algorithm", "central", *common, "--out", f"{tmp}/central"]
        )
        best = None
        for i, values in enumerate(grid):
            values = {"steps": args.steps, **values, "max_iter": args.max_iter}
            sets = [x for k, v in values.items() for x in ("--set", f"{k}={v}")]
            meta = await _meta(
                [
                    "--algorithm",
                    "consensus",
                    "--tuning",
                    "paper",
                    *common,
                    *sets,
                    "--out",
                    f"{tmp}/{i}",
                ]
            )
            s = score(meta, central, demand_mwh)
            print(values, meta["stop_reason"], meta["iterations"], "score", s)
            if s is not None and (best is None or (s, meta["iterations"]) < best[:2]):
                best = (s, meta["iterations"], values, meta)
    if best is None:
        print("no candidate converged")
        return None
    s, iterations, values, meta = best
    gap = meta["total_cost"] / central["total_cost"] - 1
    values = {k: v for k, v in values.items() if k != "max_iter"} | {
        "note": f"esb-tune {meta['inputs']['start']} {args.hours} h: gap {gap:+.4%}, "
        f"score {s:.2e}, {iterations} iterations"
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
    parser.add_argument("--steps", default="constant", choices=("decaying", "constant"))
    parser.add_argument("--start")
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
