#!/bin/bash
# Runs compare_costs.py once all scenario arrays of a sweep have finished.
# submit_all.sh submits this with --dependency=afterany:<all array job ids>.
#
# Usage: sbatch energy-scheduling-benchmark/hpc/aggregate.sbatch <network-stem> ...
#
#SBATCH --job-name=esb-aggregate
#SBATCH --partition=rosa.p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -eo pipefail # no -u: /etc/profile, Lmod and venv activate are not nounset-safe

export ESB_ROOT="${ESB_ROOT:-$SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source "$ESB_ROOT/energy-scheduling-benchmark/hpc/env.sh"

python "$ESB_ROOT/compare_costs.py" "$@"
