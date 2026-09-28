#!/usr/bin/env bash
# One-time setup of the Python environment on the Oldenburg HPC cluster.
# Run on the LOGIN node (compute nodes have no internet access for pip).
#
# Usage:
#   energy-scheduling-benchmark/hpc/setup_env.sh           # venv from the cluster's Python module
#   energy-scheduling-benchmark/hpc/setup_env.sh --conda   # Python 3.14 via Miniforge3 (matches the workstation)
#
# Creates $ESB_ROOT/.venv — the same location run_all_scenarios.sh expects — and installs
# the pinned third-party dependencies plus the four sibling packages in editable mode.
# Re-running it deletes and rebuilds the venv.

set -eo pipefail # no -u: /etc/profile and Lmod are not nounset-safe

ESB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ESB_ROOT/.venv"
MODE="${1:-module}"

for pkg in mango mango-energy-environments distributed-resource-optimization energy-scheduling-benchmark; do
    if [[ ! -f "$ESB_ROOT/$pkg/pyproject.toml" ]]; then
        echo "Missing $ESB_ROOT/$pkg — copy the whole monorepo, not just this package (see hpc/README.md)." >&2
        exit 1
    fi
done

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile
fi
module purge || true

case "$MODE" in
    module)
        module load hpc-env/13.1
        module load Python
        PYTHON=python3
        ;;
    --conda)
        module load Miniforge3
        CONDA_PREFIX_DIR="$ESB_ROOT/.conda-py314"
        if [[ ! -x "$CONDA_PREFIX_DIR/bin/python" ]]; then
            conda create -y -p "$CONDA_PREFIX_DIR" -c conda-forge python=3.14
        fi
        PYTHON="$CONDA_PREFIX_DIR/bin/python"
        ;;
    *)
        echo "Unknown option: $MODE (expected nothing or --conda)" >&2
        exit 1
        ;;
esac

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "$("$PYTHON" --version) is too old: requirements.txt pins need Python >= 3.11." >&2
    echo "Check 'module avail Python' for a newer version, or re-run with --conda." >&2
    exit 1
fi
echo "Using $("$PYTHON" --version) at $(command -v "$PYTHON")"

rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"

# Remember the modules the venv was built against so env.sh can load the same ones.
# The conda interpreter is self-contained, so it needs none.
if [[ "$MODE" == "module" ]]; then
    if ! module -t list 2>&1 | grep -E '^(hpc-env|Python)/' > "$VENV/esb-modules"; then
        echo "Could not record the loaded hpc-env/Python modules ('module -t list' output below)." >&2
        module -t list >&2
        exit 1
    fi
else
    : > "$VENV/esb-modules"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ESB_ROOT/energy-scheduling-benchmark/requirements.txt"
# mango first, so the siblings resolve it locally instead of pulling mango-agents from PyPI.
python -m pip install -e "$ESB_ROOT/mango"
python -m pip install -e "$ESB_ROOT/mango-energy-environments"
python -m pip install -e "$ESB_ROOT/distributed-resource-optimization"
python -m pip install -e "$ESB_ROOT/energy-scheduling-benchmark[dev]"

python -c "import highspy, pypsa, netCDF4, mango, distributed_resource_optimization, energy_scheduling_benchmark"
echo
echo "Environment ready at $VENV (modules: $(tr '\n' ' ' < "$VENV/esb-modules"))"
echo "Next: run the smoke test — see hpc/README.md, step 5."
