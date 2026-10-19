#!/usr/bin/env python3
"""Acceptance test of a sweep (decision Q19): were the inputs certainly correct?

No result is compared with a local run. For every line of a sweep manifest
(``make_manifest.py``) the run's ``<run_id>-meta.json`` must exist and record

* the scenario and network of the line, and exactly the line's CLI settings
  (topology, cost model, days, loss rate, seed, slack);
* the network file's SHA-256 from ``expected.json`` (written with the manifest);
* every repository at the HEAD recorded in ``expected.json``, with the same
  dirty flag (a dirty tree is reported: its SHA cannot vouch for the changes).

Usage::

    hpc/verify_inputs.py hpc-manifests/<stamp>                  # results/ next to the repos
    hpc/verify_inputs.py hpc-manifests/<stamp> --results /path/to/results

Exit code 0 if every run passed, 1 otherwise (missing runs included unless
``--allow-missing``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ESB_DIR = Path(__file__).resolve().parents[1]

FLAG_DIMS = {
    "--topology": ("topology", str),
    "--cost-model": ("cost_model", str),
    "--simulate-days": ("simulate_days", int),
    "--loss-rate": ("loss_rate", float),
    "--seed": ("seed", int),
}


def expected_dims(flags: list[str]) -> dict:
    """The meta ``dimensions`` a manifest line's flags must produce."""
    dims: dict = {"slack": "--slack" in flags}
    for i, flag in enumerate(flags):
        if flag in FLAG_DIMS:
            key, cast = FLAG_DIMS[flag]
            dims[key] = cast(flags[i + 1])
    return dims


def check_run(
    meta: dict, scenario: str, network: str, flags: list[str], expected: dict
) -> list[str]:
    """Problems of one run (empty if its inputs are as expected)."""
    problems = []
    dims = meta.get("dimensions", {})
    if dims.get("scenario") != scenario.replace("-", "_"):
        problems.append(f"scenario {dims.get('scenario')!r} != {scenario!r}")
    if Path(str(dims.get("network", ""))).name != Path(network).name:
        problems.append(f"network {dims.get('network')!r} != {network!r}")
    for key, value in expected_dims(flags).items():
        if dims.get(key) != value:
            problems.append(f"{key} {dims.get(key)!r} != {value!r}")
    want_sha = expected.get("networks", {}).get(network)
    got_sha = meta.get("inputs", {}).get("network_sha256")
    if want_sha != got_sha:
        problems.append(f"network sha256 {str(got_sha)[:12]} != {str(want_sha)[:12]}")
    for pkg, want in expected.get("git", {}).items():
        got = meta.get("git", {}).get(pkg, {})
        if got.get("sha") != want.get("sha"):
            problems.append(
                f"{pkg} at {str(got.get('sha'))[:10]} != {str(want.get('sha'))[:10]}"
            )
        elif got.get("dirty") != want.get("dirty"):
            problems.append(f"{pkg} dirty={got.get('dirty')} != {want.get('dirty')}")
    return problems


def verify(manifest_dir: Path, results_dir: Path) -> tuple[int, list[str], list[str]]:
    """``(runs checked, missing run ids, problem lines)``."""
    expected = json.loads((manifest_dir / "expected.json").read_text())
    checked, missing, problems = 0, [], []
    for manifest in sorted(manifest_dir.glob("*.tsv")):
        scenario = manifest.stem
        with open(manifest, newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if not row:
                    continue
                run_id, network, *flags = row
                stem = Path(network).stem if Path(network).suffix else network
                meta_path = results_dir / stem / f"{run_id}-meta.json"
                checked += 1
                if not meta_path.is_file():
                    missing.append(f"{stem}/{run_id}")
                    continue
                meta = json.loads(meta_path.read_text())
                for p in check_run(meta, scenario, network, flags, expected):
                    problems.append(f"{stem}/{run_id}: {p}")
    return checked, missing, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("--results", type=Path, default=ESB_DIR.parent / "results")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    expected = json.loads((args.manifest_dir / "expected.json").read_text())
    dirty = [pkg for pkg, g in expected.get("git", {}).items() if g.get("dirty")]
    checked, missing, problems = verify(args.manifest_dir, args.results)
    print(
        f"Checked {checked} runs: {checked - len(missing) - len({p.split(':')[0] for p in problems})} ok, "
        f"{len(missing)} missing, {len({p.split(':')[0] for p in problems})} with wrong inputs"
    )
    if dirty:
        print(
            f"WARNING: expected.json records uncommitted changes in {dirty}; "
            "their SHAs do not pin the code that ran."
        )
    for line in problems[:50]:
        print("  " + line)
    if len(problems) > 50:
        print(f"  ... {len(problems) - 50} more")
    for run in missing[:20]:
        print("  missing: " + run)
    if len(missing) > 20:
        print(f"  ... {len(missing) - 20} more missing")
    failed = bool(problems) or (bool(missing) and not args.allow_missing)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
