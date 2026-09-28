# Runtime environment for benchmark jobs on the Oldenburg HPC cluster (ROSA).
#
# Source this (do not execute it) at the top of every job script, or in an
# interactive shell on a compute node:
#
#   source "$ESB_ROOT/energy-scheduling-benchmark/hpc/env.sh"
#
# ESB_ROOT is the monorepo root (the directory holding mango/, energy-scheduling-benchmark/,
# compare_costs.py, ...). If unset it is derived from this file's location.

if [[ -z "${ESB_ROOT:-}" ]]; then
    ESB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export ESB_ROOT

# `module` is a shell function set up by the login profile; batch shells started
# with --export=NONE may not have it yet.
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile
fi

# setup_env.sh records exactly which modules the venv was built against. Loading the
# same ones keeps the venv's interpreter and its shared libraries in sync.
if [[ -f "$ESB_ROOT/.venv/esb-modules" ]]; then
    module purge >/dev/null 2>&1 || true
    while read -r mod; do
        [[ -n "$mod" ]] && module load "$mod"
    done < "$ESB_ROOT/.venv/esb-modules"
fi

if [[ ! -f "$ESB_ROOT/.venv/bin/activate" ]]; then
    echo "env.sh: no venv at $ESB_ROOT/.venv — run hpc/setup_env.sh on the login node first" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$ESB_ROOT/.venv/bin/activate"

# One run is one single-threaded asyncio process. Without pinning, BLAS/HiGHS detect
# all 128 cores of a node and oversubscribe it when many array tasks share the node.
threads="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"

# Headless plotting; keep matplotlib's font cache node-local instead of hammering $HOME.
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

export PYTHONUNBUFFERED=1
unset threads
