"""esb-compare: cost gap of every run vs. its central optimum (TODO Next #10, decided 2026-11-09).

Read-only over ``<root>/<network>/<algorithm>/<run>/meta.json``. Each run is paired with the
central run of the same network, window (start, hours), storage and slack setting; one row per
non-central run goes into a CSV (default ``outputs/compare.csv``) and a printed markdown table.
Missing runs are reported, derived from what exists: a run without its central run (row
skipped), and per network every window an algorithm setting lacks that another run has.

Usage::

    esb-compare                       # outputs/ → outputs/compare.csv
    esb-compare --root runs --out runs/compare.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_runs(root: Path) -> pd.DataFrame:
    """One row per ``meta.json`` under *root* (network/algorithm from the folder names)."""
    rows = []
    for path in sorted(root.glob("*/*/*/meta.json")):
        meta = json.loads(path.read_text())
        i = meta["inputs"]
        network, algorithm, name = path.parts[-4:-1]
        rows.append(
            {
                "network": network,
                "algorithm": algorithm,
                # folder suffix after [no]storage: consensus tuning route (+ --set hash)
                "variant": name.partition("storage")[2].lstrip("_"),  # "" for old names
                "topology": i["topology"],
                "loss_rate": i["loss_rate"],
                "seed": i["seed"],
                "start": i["start"],
                "hours": i["hours"],
                "storage": i.get("storage", True),  # older runs had no --no-storage
                "slack": i["slack"],
                "total_cost": meta["total_cost"],
                "shortfall_mwh": meta["imbalance"]["shortfall_mwh"],
                "surplus_mwh": meta["imbalance"]["surplus_mwh"],
                "iterations": meta["iterations"],
                "stop_reason": meta["stop_reason"],
                "messages_sent": meta["messages"]["sent"],
                "messages_delivered": meta["messages"]["delivered"],
                "messages_lost": meta["messages"]["lost"],
                "wall_time_s": meta["wall_time_s"],
                "run": str(path.parent),
            }
        )
    return pd.DataFrame(rows)


WINDOW = ["network", "start", "hours", "storage", "slack"]
SETTING = ["algorithm", "variant", "topology", "loss_rate", "seed"]


def compare(runs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rows with ``central_cost`` + ``gap_pct``, and the missing-run notices."""
    missing = []
    settings = (
        runs.copy()
    )  # central does not depend on topology, loss or seed: one setting
    settings.loc[settings.algorithm == "central", ["topology", "loss_rate", "seed"]] = (
        "any",
        0.0,
        0,
    )
    for network, net_runs in settings.groupby("network"):
        windows = set(net_runs[WINDOW[1:]].itertuples(index=False))
        for (alg, variant, topology, loss, seed), own in net_runs.groupby(SETTING):
            label = f"{network} {alg}{f'[{variant}]' if variant else ''} {topology}_loss{loss:g}_seed{seed}"
            for w in sorted(windows - set(own[WINDOW[1:]].itertuples(index=False))):
                missing.append(f"{label}: {_window(w)}")
    central = runs[runs.algorithm == "central"].drop_duplicates(WINDOW)  # any topology
    rows = runs[runs.algorithm != "central"].merge(
        central[[*WINDOW, "total_cost"]].rename(columns={"total_cost": "central_cost"}),
        on=WINDOW,
        how="left",
        indicator=True,
    )
    for run in rows[rows._merge == "left_only"].run:
        missing.append(f"no central run for {run} (skipped)")
    rows = rows[rows._merge == "both"].drop(columns="_merge")
    rows.insert(
        rows.columns.get_loc("total_cost") + 1,
        "gap_pct",
        (rows.total_cost / rows.central_cost - 1) * 100,  # NaN if a cost is null
    )
    rows.insert(
        rows.columns.get_loc("total_cost"), "central_cost", rows.pop("central_cost")
    )
    return rows.reset_index(drop=True), missing


def _window(w) -> str:
    start, hours, storage, slack = w
    return f"{start} {hours} h {'' if storage else 'no '}storage{' slack' if slack else ''}"


def markdown(df: pd.DataFrame) -> str:
    cols = ["network", "algorithm", "variant", "topology", "loss_rate", "start",
            "storage", "gap_pct", "shortfall_mwh", "surplus_mwh", "iterations", "stop_reason",
            "messages_sent", "wall_time_s"]  # fmt: skip
    cells = df[cols].map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v))
    lines = [cols, ["---"] * len(cols), *cells.values.tolist()]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="outputs", help="run folders (esb-run --out)")
    parser.add_argument("--out", help="CSV path (default: <root>/compare.csv)")
    args = parser.parse_args(argv)
    runs = load_runs(Path(args.root))
    if runs.empty:
        raise SystemExit(f"no runs under {args.root}/<network>/<algorithm>/<run>/")
    rows, missing = compare(runs)
    out = Path(args.out or Path(args.root) / "compare.csv")
    rows.to_csv(out, index=False)
    for line in missing:
        print(f"missing: {line}", file=sys.stderr)
    print(markdown(rows))
    print(f"\n{len(rows)} runs → {out}; {len(missing)} missing")


if __name__ == "__main__":
    main()
