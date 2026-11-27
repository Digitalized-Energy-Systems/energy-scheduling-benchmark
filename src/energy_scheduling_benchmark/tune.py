"""esb-tune: sweep an algorithm's parameters for one network × topology (plan §7 I15–I19; audit
2026-11-12 Q-A1).

Selection by the literature (plan S4d, ``docs/tuning-selection-plan.md``, D-194–D-199;
Beiranvand et al. 2017 §4.3–4.4, Deb 2000): runs central once per window, then every candidate
of the algorithm's ``sweep_grid`` on each window with the same budget for every algorithm
(``--max-iter``, default ``BUDGET`` = 10 000, D-198) and the algorithm's ``SWEEP_RUN`` settings
(its own stop rule off, D-196; consensus also a divergence abort). A window is *solved* once
|gap| ≤ ``EPS`` and imbalance/demand ≤ ``EPS`` hold to the end of the run (``solved_at``, from
``trace.csv``; D-195/D-196); a candidate counts with its worst window. The winner (``select``):
tier 1 the solved candidate with the fewest iterations; tier 2 (none solved) the smallest
|gap| among candidates with imbalance ≤ ``EPS``; tier 3 the smallest imbalance (D-197). The
winner per variant (``VARIANT``, consensus: ``steps``) goes into the note, the overall winner is
written into ``resources/tuned_parameters.toml`` as ``[<algorithm>.<network>.<topology>.sweep]``
with ``solved`` = tier 1 (D-199) whatever its gap (decision 2026-11-09). Nothing is written only
if no candidate runs through every window.

Exact pruning (T3): once a variant has a solved candidate, another one stops after a window that
is unsolved or needs more iterations than that one — it cannot win. Ties go to the earlier
candidate, so the result does not depend on ``--jobs`` (T7). Runs go to ``--jobs`` processes (T2).

Pilot (level 2, plan ``docs/sweep-scale-plan.md``, D-261; Choi et al. 2019 §4: "an initial set
of experiments over a relatively large search space" picks the final one): an algorithm with
``PILOT_KEY`` (diffusion, exact diffusion) first runs ``sweep_grid(..., pilot=True)`` — its step
over four decades — for ``--pilot-iter`` iterations on the first window; a run is stable when it
uses that budget and its imbalance has not grown (``stable``). Per variant the largest step up to
which every pilot value is stable (``pilot_top``) tops the final grid
(``sweep_grid(..., top=...)``). Pilot runs: ``pilot_<i>/`` + ``pilot.csv``; their budget is in the
note and ``sweep.json`` (Sivaprasad et al. 2020: report the tuning effort). Every note flags a
winner on the edge of its variant's grid (Choi et al. 2019 App. E, D-262).

Records (S4b, D-147–D-152): every run is kept in the sweep folder
``<--out>/<algorithm>/<network>/<topology>_[no]storage/`` (a re-sweep replaces it only with
``--replace``, else esb-tune stops, D-154)
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
#: ``VARIANT`` (algorithm interface, :mod:`.algorithms`)
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


#: 2026-11-23 D-195: a window is solved when |gap| and imbalance/demand both stay ≤ EPS
EPS = 1e-3
#: 2026-11-23 D-198: the same sweep budget for every algorithm (= the runs' max_iter)
BUDGET = 10_000
#: 2026-11-27 D-261: iterations of a pilot run (level 2, Choi et al. 2019 §4)
PILOT_ITER = 1000


def stable(meta: dict, trace: pd.DataFrame | None) -> bool:
    """A pilot run is stable when it used its whole budget (or needed no ED) and its imbalance at
    the end is no larger than at the start (D-261)."""
    if meta["stop_reason"] == RENEWABLES_COVER:
        return True
    if meta["stop_reason"] not in ("max_iter", "converged") or trace is None:
        return False
    return bool(trace.imbalance_mwh.iloc[-1] <= trace.imbalance_mwh.iloc[0])


def pilot_top(runs: list[tuple]) -> dict:
    """``[(variant, step, stable)]`` → variant → the largest step up to which every pilot step
    is stable (a step tried in several candidates counts as stable only if all are); the
    smallest step if none is (D-261)."""
    top = {}
    for variant in {v for v, _, _ in runs}:
        ok: dict[float, bool] = {}
        for v, x, good in runs:
            if v == variant:
                ok[x] = ok.get(x, True) and good
        steps = sorted(ok)
        top[variant] = steps[0]
        for x in steps:
            if not ok[x]:
                break
            top[variant] = x
    return top


def grid_edges(candidates: list[dict], i: int, variant_key) -> list[str]:
    """Numeric parameters whose value in candidate *i* is the smallest or largest of its variant's
    grid — the winner should lie inside the grid (Choi et al. 2019 App. E, D-262)."""
    peers = [
        c for c in candidates if c.get(variant_key) == candidates[i].get(variant_key)
    ]
    edges = []
    for k, x in candidates[i].items():
        if k == variant_key or isinstance(x, bool) or not isinstance(x, (int, float)):
            continue
        values = [c[k] for c in peers if k in c]
        # a range needs ≥ 3 values: with 2 (e.g. μ_IS ∈ {0, 0.5}) every value is an edge
        if len(set(values)) > 2 and x in (min(values), max(values)):
            edges.append(k)
    return edges


def solved_at(trace: pd.DataFrame, central: dict, demand_mwh: float) -> int | None:
    """The first iteration from which |gap| ≤ ``EPS`` and imbalance/demand ≤ ``EPS`` hold to
    the end of *trace* (imbalance as ``score``), None if the last row misses the target (D-196)."""
    cost = central["total_cost"]
    gaps = (trace.cost / cost - 1).abs() if cost else trace.cost.abs()
    off = (
        trace.shortfall_mwh
        + (trace.surplus_mwh - central["imbalance"]["surplus_mwh"]).abs()
    )
    ok = ((gaps <= EPS) & (off / demand_mwh <= EPS)).to_numpy()
    if not len(ok) or not ok[-1]:
        return None
    misses = (~ok).nonzero()[0]
    return int(trace.iteration.iloc[misses[-1] + 1 if len(misses) else 0])


def select(done: dict[int, tuple[int | None, float, float]]) -> tuple[int, int]:
    """(winner, tier) of ``{candidate: (worst-window iterations to the target or None, worst
    |gap|, worst imbalance/demand)}``: tier 1 the fewest iterations among the solved, tier 2
    the smallest |gap| among those with imbalance ≤ ``EPS``, tier 3 the smallest imbalance;
    then the earlier candidate (T7)."""
    # 2026-11-23 D-194/D-197 (Beiranvand §4.3: unsolved = unsuccessful; Deb 2000: feasible first);
    # replaces the score + CLOSE rule of D-182/D-183
    solved = [(it, g, i) for i, (it, g, _) in done.items() if it is not None]
    if solved:
        return min(solved)[2], 1
    feasible = [(g, v, i) for i, (_, g, v) in done.items() if v <= EPS]
    if feasible:
        return min(feasible)[2], 2
    return min((v, g, i) for i, (_, g, v) in done.items())[2], 3


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


def _note(variant, best: tuple, hours: int, edges: list[str] = ()) -> str:
    (its, _, off), values, gaps = best
    shown = ", ".join(f"{k}={v}" for k, v in values.items() if k != "steps")
    windows = "; ".join(f"{start}: {g:+.4%}" for start, g in gaps.items())
    status = f"solved within {its} it." if its is not None else "not solved"
    edge = f", at the grid edge in {'/'.join(edges)}" if edges else ""  # D-262
    return f"{variant} {shown} — {hours} h gap {windows}, imbalance ≤ {off:.1e}, {status}{edge}"


def sweep(args: argparse.Namespace) -> dict | None:
    """The best candidate that runs through every window (see the module docstring)."""
    alg = ALGORITHMS[args.algorithm]
    base = ["--network", args.network, "--topology", args.topology]
    base += ["--hours", str(args.hours)] + ([] if args.storage else ["--no-storage"])
    raw = full = load_network(datasets.prepared_path(args.network))
    if not args.storage:
        raw = without_storage(raw)
    # 2026-11-23 D-198: one budget for every algorithm (was per algorithm, D-178)
    args.max_iter = args.max_iter or BUDGET
    fixed = alg.SWEEP_RUN | {"max_iter": args.max_iter}
    sets = [x for k, v in fixed.items() for x in ("--set", f"{k}={v}")]
    starts = args.start or [None]
    # 2026-11-17 D-147/D-149: every run is kept here for plots (was a temporary directory)
    store = Path(args.out, args.algorithm, args.network, args.topology)
    store = store.with_name(f"{store.name}_{'' if args.storage else 'no'}storage")
    # 2026-11-17 D-154: never delete earlier records unasked (a local check wiped them silently)
    if store.exists() and not args.replace:
        raise SystemExit(
            f"records exist: {store}\n  pass --replace to overwrite them, or --out elsewhere"
        )
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

        # 2026-11-27 D-261: pilot on the first window, then the grid under its largest stable step
        top, pilot = None, None
        if getattr(args, "pilot", True) and hasattr(alg, "PILOT_KEY"):
            trial, _ = alg.sweep_grid(raw, full, args.topology, pilot=True)
            iters = getattr(args, "pilot_iter", PILOT_ITER)
            short = alg.SWEEP_RUN | {"max_iter": iters}
            short_sets = [x for k, v in short.items() for x in ("--set", f"{k}={v}")]
            futures = [
                pool.submit(
                    _meta,
                    ["--algorithm", args.algorithm, "--tuning", "paper", *common[0]]
                    + [x for k, v in c.items() for x in ("--set", f"{k}={v}")]
                    + short_sets
                    + ["--out", f"{store}/pilot_{i}"],
                )
                for i, c in enumerate(trial)
            ]
            runs, prows = [], []
            for i, (c, future) in enumerate(zip(trial, futures)):
                meta, out = future.result()
                trace = Path(out, "trace.csv")
                ok = stable(meta, pd.read_csv(trace) if trace.exists() else None)
                runs.append((c.get(alg.VARIANT), c[alg.PILOT_KEY], ok))
                prows.append(
                    {"candidate": i, **c, "stop_reason": meta["stop_reason"]}
                    | {"stable": ok, "folder": os.path.relpath(out, store)}
                )
                print("pilot", c, meta["stop_reason"], "stable" if ok else "unstable")
            pd.DataFrame(prows).to_csv(store / "pilot.csv", index=False)
            top = pilot_top(runs)
            pilot = {"runs": len(trial), "iterations": iters, "top": top}
        candidates, grid_note = alg.sweep_grid(
            raw, full, args.topology, **({} if top is None else {"top": top})
        )
        if pilot:
            grid_note += f"; pilot {pilot['runs']} × {pilot['iterations']} it. on day 1"
        print(f"{len(candidates)} candidates; {grid_note}", flush=True)

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
        results: dict[int, list[tuple]] = {i: [] for i in range(len(candidates))}
        gaps: dict[int, dict[str, float]] = {i: {} for i in range(len(candidates))}
        complete: dict = {}  # variant -> {candidate: (iterations|None, |gap|, imbalance), worst}
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
                its = off = None
                if s is not None:
                    g = abs(gap(meta["total_cost"], central["total_cost"]))
                    off = s - g  # imbalance / demand
                    trace = Path(out, "trace.csv")
                    if trace.exists():
                        its = solved_at(pd.read_csv(trace), central, demand)
                    elif g <= EPS and off <= EPS:
                        its = 0  # no iterations: the renewables cover the window
                if meta["total_cost"] is not None:
                    gaps[i][label] = gap(meta["total_cost"], central["total_cost"])
                rows[i, w] = {
                    "status": "failed" if s is None else "ok",
                    "stop_reason": meta["stop_reason"],
                    "iterations": meta["iterations"],
                    "solved_at": its,
                    "total_cost": meta["total_cost"],
                    "gap": gaps[i].get(label),
                    "imbalance": off,
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
                    its,
                    s,
                    flush=True,
                )
                if s is None:
                    failed += 1
                    continue
                results[i].append((its, g, off))
                variant = candidates[i].get(alg.VARIANT)
                solved = [
                    v[0] for v in complete.get(variant, {}).values() if v[0] is not None
                ]
                if len(results[i]) == len(windows):
                    r = results[i]
                    unsolved = any(x is None for x, _, _ in r)
                    complete.setdefault(variant, {})[i] = (
                        None if unsolved else max(x for x, _, _ in r),
                        max(x for _, x, _ in r),
                        max(x for _, _, x in r),
                    )
                elif solved and (its is None or its > min(solved)):
                    pruned += 1  # cannot beat the variant's solved best (D-194)
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
    values = chosen = tier = None
    every = {i: k for d in complete.values() for i, k in d.items()}
    best = {v: select(d)[0] for v, d in complete.items()}  # variant -> its winner
    if not best:
        print("no candidate ran through every window (each diverged/stalled on one)")
    else:
        chosen, tier = select(every)
        # the chosen set's variant first
        ranked = sorted(best, key=lambda v: best[v] != chosen)
        notes = [
            _note(
                v,
                (every[best[v]], candidates[best[v]], gaps[best[v]]),
                args.hours,
                grid_edges(candidates, best[v], alg.VARIANT),
            )
            for v in ranked
        ]
        # 2026-11-23 D-199 (Beiranvand §5.3: a failed target counts as failure)
        status = (
            f"solved (ε {EPS:g})"
            if tier == 1
            else f"unsuccessful: no candidate within ε {EPS:g} on every tuning day, tier {tier}"
        )
        values = candidates[chosen] | {
            "solved": tier == 1,
            "note": f"esb-tune ({grid_note}, {args.max_iter} it.) {status}; best: "
            + " | ".join(notes),
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
        "pilot": None
        if pilot is None
        else pilot | {"top": {str(k): v for k, v in top.items()}},
        "central": {
            label: {
                "total_cost": central["total_cost"],
                "demand_mwh": demand,
                "folder": os.path.relpath(out, store),
            }
            for label, central, demand, out in windows
        },
        "eps": EPS,
        "best": [
            {
                "variant": v,
                "candidate": i,
                "iterations": every[i][0],
                "gap": every[i][1],
                "imbalance": every[i][2],
            }
            for v, i in best.items()
        ],
        "chosen_candidate": chosen,
        "tier": tier,
        "solved": tier == 1 if tier else None,
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
        "--max-iter",
        type=int,
        default=None,
        help="fixed budget per candidate (default: the algorithm's SWEEP_RUN max_iter, else 3000)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="parallel runs (processes)")
    parser.add_argument(
        "--no-pilot",
        dest="pilot",
        action="store_false",
        help="skip the pilot of algorithms with PILOT_KEY (grid from the bound only, D-261)",
    )
    parser.add_argument(
        "--pilot-iter",
        type=int,
        default=PILOT_ITER,
        help=f"iterations per pilot run (default {PILOT_ITER}, D-261)",
    )
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
        "--replace",
        action="store_true",
        help="overwrite this sweep's existing records (else esb-tune stops, D-154)",
    )
    parser.add_argument(
        "--no-write", dest="write", action="store_false", help="only print the best set"
    )
    sweep(parser.parse_args(argv))


if __name__ == "__main__":
    main()
