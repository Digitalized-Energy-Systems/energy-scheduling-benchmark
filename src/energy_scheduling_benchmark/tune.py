"""esb-tune: sweep an algorithm's parameters for one network × topology (plan §7 I15–I19; audit
2026-11-12 Q-A1).

Runs central once per window, then every candidate of the algorithm's ``sweep_grid`` on each
window, each with a fixed budget (``--max-iter``) plus the algorithm's ``SWEEP_RUN`` settings
(both: stop rule on with tolerances 10× below the run's, 2026-11-17 D-142/D-143; consensus also a
divergence abort). Score = |cost gap| + relative imbalance vs central; a window's rank
(``rank``) is the score (consensus: its step-change stop can fire far off the optimum) or, with
``RANK_BY_ITERATIONS`` (admm, D-140), runs that met their stop rule first, by iterations; a
candidate counts with its worst window, and the best candidate per variant
(``VARIANT``, consensus: ``steps``) is kept, the overall best is written into
``resources/tuned_parameters.toml`` as ``[<algorithm>.<network>.<topology>.sweep]`` whatever its
gap (decision 2026-11-09); the note records the gap per window and the best of each variant.
Nothing is written only if no candidate runs through every window.

Exact pruning (T3): the score is the worst window, so once one window scores worse than the best
complete candidate of the same variant, the candidate cannot win and its other windows are skipped.
Ties go to the earlier candidate, so the result does not depend on ``--jobs`` (T7; iterations
rank only with ``RANK_BY_ITERATIONS``, D-140). Runs go to ``--jobs`` processes (T2).

Records (S4b, D-147–D-152): every run is kept in the sweep folder
``<--out>/<algorithm>/<network>/<topology>_[no]storage/`` (emptied first, a re-sweep replaces it)
as ``central_<day>/`` and ``<candidate>_<day>/`` (esb-run output: meta, trace, schedule), plus
``sweep.csv`` (one row per candidate × day; ``skipped`` = not run: after a failed day or pruned)
and ``sweep.json`` (args, central cost per day, best per variant, chosen set) for plots.

    esb-tune --network lower_saxony_2025 --topology ring --jobs 16 --start 2025-01-14 ...
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd
from mango_energy_environments import LOAD, load_network

from . import algorithms, datasets, parameters
from .outputs import RENEWABLES_COVER
from .run import build_parser, run, window, without_storage

#: the tunable algorithms: modules with ``sweep_grid(network, full, topology)``, ``SWEEP_RUN``,
#: ``VARIANT``, ``RANK_BY_ITERATIONS`` (algorithm interface, :mod:`.algorithms`)
ALGORITHMS = {
    n: m for n, m in algorithms.ALGORITHMS.items() if hasattr(m, "sweep_grid")
}


def _meta(argv: list[str]) -> tuple[dict, str]:
    """One esb-run (in a worker process) → its meta.json and its run folder."""
    out = asyncio.run(run(build_parser().parse_args(argv)))
    return json.loads((out / "meta.json").read_text()), str(out)


def gap(cost: float, central_cost: float) -> float:
    """Relative cost gap; 0 when equal (a window the renewables cover can cost 0 in both)."""
    return 0.0 if cost == central_cost else cost / central_cost - 1


def score(meta: dict, central: dict, demand_mwh: float) -> float | None:
    """|cost gap| + imbalance / demand (both relative); None unless the run used its whole
    budget ("max_iter"), met its stop rule ("converged"; "stalled"/"diverged" = failed) or needed no ED at all (a window the
    renewables cover: the same for every candidate). Surplus counts only where it differs from
    central's: islanded, an unabsorbable renewable surplus is part of the optimum (S1)."""
    if (
        meta["stop_reason"] not in ("max_iter", "converged", RENEWABLES_COVER)
        or meta["total_cost"] is None
    ):
        return None
    surplus = meta["imbalance"]["surplus_mwh"] - central["imbalance"]["surplus_mwh"]
    imbalance = meta["imbalance"]["shortfall_mwh"] + abs(surplus)
    return abs(gap(meta["total_cost"], central["total_cost"])) + imbalance / demand_mwh


def rank(meta: dict, s: float, by_iterations: bool) -> tuple:
    """Sort key of one window. ``by_iterations`` (the algorithm's ``RANK_BY_ITERATIONS``): a run
    that met its stop rule (or needed no ED) beats every run that used the whole budget and ranks
    by iterations — at the noise floor the score cannot tell a fast ρ from a slow one (A3,
    2026-11-16 D-140); the others rank by score. Otherwise every window ranks by score."""
    # 2026-11-16 D-141: fastest converged wins, no gap band (Jonas); at the sweep tolerance, D-142.
    # 2026-11-17 D-143: consensus ranks by score, its step-change stop can fire far off the optimum.
    if not by_iterations or meta["stop_reason"] == "max_iter":
        return (1, s)
    return (0, meta["iterations"], s)


def write_section(name: str, values: dict) -> None:
    """Replace table ``[name]`` in place (its header + the key lines that follow it) or append
    it; every other line, comments after the table included, is kept.

    Locked (where the filesystem has locks, ``parameters.lock``), so parallel sweeps (one per
    network × topology) do not lose each other's sections.
    """
    section = "\n".join(
        [f"[{name}]"] + [f"{k} = {json.dumps(v)}" for k, v in values.items()]
    )
    with open(parameters.PARAMETERS, "r+") as f:
        parameters.lock(f, fcntl.LOCK_EX)
        text = f.read()
        text, n = re.subn(
            rf"^\[{re.escape(name)}\]\n(?:[^\[#\n].*\n?)*",
            lambda _: section + "\n",  # a function: the note may contain backslashes
            text,
            flags=re.M,
        )
        if not n:
            text = text.rstrip("\n") + "\n\n" + section + "\n"
        f.seek(0)
        f.write(text)
        f.truncate()


def _note(variant, best: tuple, hours: int) -> str:
    key, values, gaps = best
    shown = ", ".join(f"{k}={v}" for k, v in values.items() if k != "steps")
    windows = "; ".join(f"{start}: {g:+.4%}" for start, g in gaps.items())
    its = f", converged within {key[1]} it." if key[0] == 0 else ""
    return f"{variant} {shown} — {hours} h gap {windows}, score {key[-1]:.2e}{its}"


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
    # 2026-11-17 D-147/D-149: every run is kept here for plots (was a temporary directory)
    store = Path(args.out, args.algorithm, args.network, args.topology)
    store = store.with_name(f"{store.name}_{'' if args.storage else 'no'}storage")
    shutil.rmtree(store, ignore_errors=True)
    days = ["default" if s is None else f"{pd.Timestamp(s):%Y%m%dT%H}" for s in starts]
    with ProcessPoolExecutor(args.jobs) as pool:
        common = [base + ([] if s is None else ["--start", s]) for s in starts]
        centrals = [
            pool.submit(
                _meta,
                ["--algorithm", "central", *c, "--out", f"{store}/central_{days[w]}"],
            )
            for w, c in enumerate(common)
        ]
        windows = []  # (label, central meta, demand MWh, central folder)
        for w, start in enumerate(starts):
            net = window(raw, start, args.hours)
            demand = sum(
                c.p_max.sum() for c in net.components.values() if c.kind == LOAD
            )
            central, out = centrals[w].result()
            windows.append((str(net.time_index[0]), central, float(demand), out))

        def submit(i: int, w: int):
            values = [
                x for k, v in candidates[i].items() for x in ("--set", f"{k}={v}")
            ]
            argv = ["--algorithm", args.algorithm, "--tuning", "paper", *common[w]]
            future = pool.submit(
                _meta, argv + values + sets + ["--out", f"{store}/{i}_{days[w]}"]
            )
            pending[future] = (i, w)

        pending: dict = {}
        scores: dict[int, list[tuple]] = {i: [] for i in range(len(candidates))}
        gaps: dict[int, dict[str, float]] = {i: {} for i in range(len(candidates))}
        best: dict = {}  # variant -> (rank key of the worst window, candidate index)
        failed = pruned = 0
        rows: dict[tuple, dict] = {}  # (candidate, window) -> sweep.csv row (D-148)
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
                meta, out = future.result()
                label, central, demand, _ = windows[w]
                s = score(meta, central, demand)
                if meta["total_cost"] is not None:
                    gaps[i][label] = gap(meta["total_cost"], central["total_cost"])
                rows[i, w] = {
                    "status": "failed" if s is None else "ok",
                    "stop_reason": meta["stop_reason"],
                    "iterations": meta["iterations"],
                    "total_cost": meta["total_cost"],
                    "gap": gaps[i].get(label),
                    "score": s,
                    "sim_time_s": meta["sim_time_s"],
                    "wall_time_s": meta["wall_time_s"],
                    "folder": os.path.relpath(out, store),
                }
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
                scores[i].append(rank(meta, s, alg.RANK_BY_ITERATIONS))
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
    pd.DataFrame(
        [
            {"candidate": i, **c, "day": windows[w][0]}
            | rows.get((i, w), {"status": "skipped"})
            for i, c in enumerate(candidates)
            for w in range(len(windows))
        ]
    ).to_csv(store / "sweep.csv", index=False)
    values = None
    if not best:
        print("no candidate ran through every window (each diverged/stalled on one)")
    else:
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
            write_section(
                f"{args.algorithm}.{args.network}.{args.topology}.sweep", values
            )
        print("best:", values)
    record = {
        "args": vars(args),
        "grid_note": grid_note,
        "fixed": fixed,
        "central": {
            label: {
                "total_cost": central["total_cost"],
                "demand_mwh": demand,
                "folder": os.path.relpath(out, store),
            }
            for label, central, demand, out in windows
        },
        "best": [
            {"variant": v, "candidate": i, "rank": key} for v, (key, i) in best.items()
        ],
        "failed": failed,
        "pruned": pruned,
        "chosen": values,
    }
    (store / "sweep.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"records: {store}", flush=True)
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
        "--out",
        default="outputs/sweeps",
        help="records root: every run + sweep.csv/.json (D-149)",
    )
    parser.add_argument(
        "--no-write", dest="write", action="store_false", help="only print the best set"
    )
    sweep(parser.parse_args(argv))


if __name__ == "__main__":
    main()
