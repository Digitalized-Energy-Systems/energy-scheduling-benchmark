"""esb-tune: sweep an algorithm's parameters for one network × topology (plan §7 I15–I19; audit
2026-11-12 Q-A1).

Runs central once per window, then every candidate of the algorithm's ``sweep_grid`` on each
window, each with a fixed budget (``--max-iter``) plus the algorithm's ``SWEEP_RUN`` settings
(consensus: stop rule off, so small stable steps are not cut short; divergence abort). Score = the
worst window's |cost gap| + relative imbalance vs central; the best candidate per variant
(``VARIANT``, consensus: ``steps``) is kept, the overall best is written into
``resources/tuned_parameters.toml`` as ``[<algorithm>.<network>.<topology>.sweep]`` whatever its
gap (decision 2026-11-09); the note records the gap per window and the best of each variant.
Nothing is written only if no candidate runs through every window.

Exact pruning (T3): the score is the worst window, so once one window scores worse than the best
complete candidate of the same variant, the candidate cannot win and its other windows are skipped.
Ties go to the earlier candidate, so the result does not depend on ``--jobs`` (T7: no tie-break by
iterations — with a fixed budget all run to ``--max-iter``). Runs go to ``--jobs`` processes (T2).

    esb-tune --network lower_saxony_2025 --topology ring --jobs 16 --start 2025-01-14 ...
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from mango_energy_environments import LOAD, load_network

from . import consensus, datasets
from .outputs import RENEWABLES_COVER
from .run import build_parser, run, window, without_storage

#: algorithm → module with ``sweep_grid(network, full, topology)``, ``SWEEP_RUN``, ``VARIANT``
ALGORITHMS = {"consensus": consensus}


def _meta(argv: list[str]) -> dict:
    """One esb-run (in a worker process) → its meta.json."""
    out = asyncio.run(run(build_parser().parse_args(argv)))
    return json.loads((out / "meta.json").read_text())


def gap(cost: float, central_cost: float) -> float:
    """Relative cost gap; 0 when equal (a window the renewables cover can cost 0 in both)."""
    return 0.0 if cost == central_cost else cost / central_cost - 1


def score(meta: dict, central: dict, demand_mwh: float) -> float | None:
    """|cost gap| + imbalance / demand (both relative); None unless the run used its whole
    budget ("max_iter"; "stalled"/"diverged" = failed) or needed no ED at all (a window the
    renewables cover: the same for every candidate). Surplus counts only where it differs from
    central's: islanded, an unabsorbable renewable surplus is part of the optimum (S1)."""
    if (
        meta["stop_reason"] not in ("max_iter", RENEWABLES_COVER)
        or meta["total_cost"] is None
    ):
        return None
    surplus = meta["imbalance"]["surplus_mwh"] - central["imbalance"]["surplus_mwh"]
    imbalance = meta["imbalance"]["shortfall_mwh"] + abs(surplus)
    return abs(gap(meta["total_cost"], central["total_cost"])) + imbalance / demand_mwh


def write_section(name: str, values: dict) -> None:
    """Replace (or append) table ``[name]`` in the parameter file; other text is kept.

    Locked, so parallel sweeps (one per network × topology) do not lose each other's sections.
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


def _note(variant, best: tuple, hours: int) -> str:
    s, values, gaps = best
    shown = ", ".join(f"{k}={v}" for k, v in values.items() if k != "steps")
    windows = "; ".join(f"{start}: {g:+.4%}" for start, g in gaps.items())
    return f"{variant} {shown} — {hours} h gap {windows}, score {s:.2e}"


def sweep(args: argparse.Namespace) -> dict | None:
    """The best candidate that runs through every window (see the module docstring)."""
    alg = ALGORITHMS[args.algorithm]
    base = ["--network", args.network, "--topology", args.topology]
    base += ["--hours", str(args.hours)] + ([] if args.storage else ["--no-storage"])
    raw = full = load_network(datasets.prepared_path(args.network))
    if not args.storage:
        raw = without_storage(raw)
    candidates, grid_note = alg.sweep_grid(raw, full, args.topology)
    print(f"{len(candidates)} candidates; {grid_note}", flush=True)
    fixed = alg.SWEEP_RUN | {"max_iter": args.max_iter}
    sets = [x for k, v in fixed.items() for x in ("--set", f"{k}={v}")]
    starts = args.start or [None]
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(args.jobs) as pool:
        common = [base + ([] if s is None else ["--start", s]) for s in starts]
        centrals = [
            pool.submit(_meta, ["--algorithm", "central", *c, "--out", f"{tmp}/c{w}"])
            for w, c in enumerate(common)
        ]
        windows = []  # (label, central meta, demand MWh)
        for w, start in enumerate(starts):
            net = window(raw, start, args.hours)
            demand = sum(
                c.p_max.sum() for c in net.components.values() if c.kind == LOAD
            )
            windows.append(
                (str(net.time_index[0]), centrals[w].result(), float(demand))
            )

        def submit(i: int, w: int):
            values = [
                x for k, v in candidates[i].items() for x in ("--set", f"{k}={v}")
            ]
            argv = ["--algorithm", args.algorithm, "--tuning", "paper", *common[w]]
            future = pool.submit(
                _meta, argv + values + sets + ["--out", f"{tmp}/{i}_{w}"]
            )
            pending[future] = (i, w)

        pending: dict = {}
        scores: dict[int, list[float]] = {i: [] for i in range(len(candidates))}
        gaps: dict[int, dict[str, float]] = {i: {} for i in range(len(candidates))}
        best: dict = {}  # variant -> (score, candidate index)
        failed = pruned = 0
        fresh = iter(range(len(candidates)))

        def refill() -> None:
            # depth first: at most --jobs runs in flight, a candidate's next window goes before a
            # new candidate, so a best is known early and pruning bites
            while len(pending) < args.jobs and (i := next(fresh, None)) is not None:
                submit(i, 0)

        refill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                i, w = pending.pop(future)
                meta = future.result()
                label, central, demand = windows[w]
                s = score(meta, central, demand)
                if meta["total_cost"] is not None:
                    gaps[i][label] = gap(meta["total_cost"], central["total_cost"])
                print(
                    candidates[i],
                    label,
                    meta["stop_reason"],
                    meta["iterations"],
                    s,
                    flush=True,
                )
                if s is None:
                    failed += 1
                    continue
                scores[i].append(s)
                worst, variant = max(scores[i]), candidates[i].get(alg.VARIANT)
                if len(scores[i]) == len(windows):
                    if variant not in best or (worst, i) < best[variant]:
                        best[variant] = (worst, i)
                elif variant in best and worst > best[variant][0]:
                    pruned += 1  # cannot beat the best of its variant any more
                else:
                    submit(i, w + 1)
            refill()
    print(f"{failed} candidates failed a window, {pruned} pruned", flush=True)
    if not best:
        print("no candidate ran through every window (each diverged/stalled on one)")
        return None
    ranked = sorted(best, key=lambda v: best[v])
    notes = [
        _note(v, (best[v][0], candidates[best[v][1]], gaps[best[v][1]]), args.hours)
        for v in ranked
    ]
    values = candidates[best[ranked[0]][1]] | {
        "note": f"esb-tune ({grid_note}, {args.max_iter} it.) best: "
        + " | ".join(notes)
    }
    if args.write:
        write_section(f"{args.algorithm}.{args.network}.{args.topology}.sweep", values)
    print("best:", values)
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algorithm", default="consensus", choices=ALGORITHMS)
    parser.add_argument("--network", default="toy")
    parser.add_argument(
        "--topology", default="grid", choices=("grid", "complete", "ring")
    )
    parser.add_argument(
        "--start",
        action="append",
        help="window start; repeat for several windows (tuning days: README)",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument(
        "--max-iter", type=int, default=3000, help="fixed budget per candidate"
    )
    parser.add_argument("--jobs", type=int, default=1, help="parallel runs (processes)")
    parser.add_argument(
        "--no-storage",
        dest="storage",
        action="store_false",
        help="tune without the network's storage units",
    )
    parser.add_argument(
        "--no-write", dest="write", action="store_false", help="only print the best set"
    )
    sweep(parser.parse_args(argv))


if __name__ == "__main__":
    main()
