#!/bin/bash
# SLURM array job: one scenario against a list of networks, one array task per network.
#
# Normally submitted by submit_all.sh. Manual use (from the monorepo root, which
# becomes $SLURM_SUBMIT_DIR):
#
#   sbatch --array=1-3 energy-scheduling-benchmark/hpc/run_scenario.sbatch \
#       esb-admm networks.txt [extra scenario flags...]
#
# networks.txt holds one network per line: a file name from
# energy-scheduling-benchmark/resources/networks/ (e.g. base_s_5_elec_2019.nc),
# an absolute path to a network file, or `toy`. Task N runs line N.
# Output lands in $ESB_ROOT/results/<network-stem>/<scenario>-*, the same layout as
# run_all_scenarios.sh, so compare_costs.py works unchanged.
#
#SBATCH --job-name=esb
#SBATCH --partition=rosa.p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=NONE

set -eo pipefail # no -u: /etc/profile, Lmod and venv activate are not nounset-safe

SCENARIO="${1:?usage: run_scenario.sbatch <esb-command> <networks-file> [extra flags...]}"
NETWORK_LIST="${2:?usage: run_scenario.sbatch <esb-command> <networks-file> [extra flags...]}"
shift 2

export ESB_ROOT="${ESB_ROOT:-$SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source "$ESB_ROOT/energy-scheduling-benchmark/hpc/env.sh"

[[ "$NETWORK_LIST" = /* ]] || NETWORK_LIST="$ESB_ROOT/$NETWORK_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
NETWORK="$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$NETWORK_LIST" | sed -n "${TASK_ID}p")"
if [[ -z "$NETWORK" ]]; then
    echo "No network on line $TASK_ID of $NETWORK_LIST" >&2
    exit 1
fi

NETWORKS_DIR="$ESB_ROOT/energy-scheduling-benchmark/resources/networks"
if [[ -f "$NETWORKS_DIR/$NETWORK" ]]; then
    NETWORK_ARG="$NETWORKS_DIR/$NETWORK"
else
    NETWORK_ARG="$NETWORK" # absolute path, or a built-in name such as `toy`
fi
STEM="$(basename "$NETWORK")"
STEM="${STEM%.*}"
NAME="${SCENARIO#esb-}"

OUT_DIR="$ESB_ROOT/results/$STEM"
mkdir -p "$OUT_DIR"
# Scenarios write <name-base>-* relative to the cwd; every task uses a distinct
# name-base, so tasks sharing a network directory never collide.
cd "$OUT_DIR"

echo "job=${SLURM_JOB_ID:-local} task=$TASK_ID node=$(hostname) scenario=$SCENARIO network=$NETWORK_ARG"
echo "out=$OUT_DIR/$NAME-* python=$(python --version 2>&1) extra=$*"

# `srun` makes SLURM account memory/CPU per step, so `seff`/`sacct` report it.
srun "$SCENARIO" --network "$NETWORK_ARG" --loss-percent 0 --name-base "$NAME" "$@"
