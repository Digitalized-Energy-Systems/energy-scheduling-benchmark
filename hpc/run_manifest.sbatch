#!/bin/bash
# SLURM array job: one line of a sweep manifest per array task (see make_manifest.py).
#
# Normally submitted by submit_sweep.sh. Manual use (from the monorepo root, which
# becomes $SLURM_SUBMIT_DIR):
#
#   sbatch --array=1-1000 energy-scheduling-benchmark/hpc/run_manifest.sbatch \
#       hpc-manifests/<stamp>/consensus.tsv [line-offset]
#
# Task N runs line N + offset of the manifest. The manifest's file name names the
# scenario (consensus.tsv -> esb-consensus); each line is
# `run_id <TAB> network <TAB> scenario flags...`. Output lands in
# $ESB_ROOT/results/<network-stem>/<run_id>-*; every run also writes
# <run_id>-meta.json, which compare_costs.py collects into results/runs.csv.
#
#SBATCH --job-name=esb-sweep
#SBATCH --partition=rosa.p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=NONE

set -eo pipefail # no -u: /etc/profile, Lmod and venv activate are not nounset-safe

MANIFEST="${1:?usage: run_manifest.sbatch <manifest.tsv> [line-offset]}"
OFFSET="${2:-0}"

export ESB_ROOT="${ESB_ROOT:-$SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source "$ESB_ROOT/energy-scheduling-benchmark/hpc/env.sh"

[[ "$MANIFEST" = /* ]] || MANIFEST="$ESB_ROOT/$MANIFEST"
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-1} + OFFSET ))
LINE="$(sed -n "${LINE_NO}p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then
    echo "No line $LINE_NO in $MANIFEST" >&2
    exit 1
fi
IFS=$'\t' read -r -a FIELDS <<< "$LINE"
RUN_ID="${FIELDS[0]}"
NETWORK="${FIELDS[1]}"
FLAGS=("${FIELDS[@]:2}")

SCENARIO="esb-$(basename "$MANIFEST" .tsv)"
NETWORKS_DIR="$ESB_ROOT/energy-scheduling-benchmark/resources/networks"
if [[ -f "$NETWORKS_DIR/$NETWORK" ]]; then
    NETWORK_ARG="$NETWORKS_DIR/$NETWORK"
else
    NETWORK_ARG="$NETWORK" # absolute path, or a built-in name such as `toy`
fi
STEM="$(basename "$NETWORK")"
STEM="${STEM%.*}"
OUT_DIR="$ESB_ROOT/results/$STEM"

echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-1} line=$LINE_NO node=$(hostname)"
echo "scenario=$SCENARIO network=$NETWORK_ARG run=$RUN_ID flags=${FLAGS[*]}"

if ! command -v "$SCENARIO" >/dev/null; then
    echo "$SCENARIO not found in the venv ($VIRTUAL_ENV) — re-run hpc/setup_env.sh on the login node" >&2
    exit 1
fi

# See run_scenario.sbatch for why srun needs --export=ALL.
srun --export=ALL "$SCENARIO" --network "$NETWORK_ARG" --name-base "$RUN_ID" \
    --out-dir "$OUT_DIR" "${FLAGS[@]}"
