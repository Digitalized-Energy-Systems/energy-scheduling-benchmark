#!/usr/bin/env bash
# Submit a sweep manifest (make_manifest.py) as SLURM job arrays, one or more per
# scenario (split at ESB_ARRAY_MAX lines), then compare_costs.py as a dependent job.
# Run on the ROSA login node, from anywhere.
#
# Usage:
#   energy-scheduling-benchmark/hpc/make_manifest.py --all            # writes hpc-manifests/<stamp>/
#   energy-scheduling-benchmark/hpc/submit_sweep.sh hpc-manifests/<stamp>
#
# Environment overrides:
#   ESB_PARTITION=mpcb.p      partition for all arrays (default rosa.p)
#   ESB_ARRAY_MAX=1000        max tasks per array (default 1000; check `scontrol show config | grep MaxArraySize`)
#   ESB_MAIL=you@uol.de       e-mail on END/FAIL of each array
#   ESB_DRY_RUN=1             print the sbatch commands instead of running them

set -euo pipefail

ESB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_DIR="$ESB_ROOT/energy-scheduling-benchmark/hpc"
LOG_DIR="$ESB_ROOT/hpc-logs"

# Per-scenario requests "memory walltime" for the longest run in the manifest
# (7 days on lower_saxony_2030). Generous first guesses; tighten with `seff`
# after the first sweep and from the probe table in hpc/README.md.
declare -A RESOURCES=(
    [central-dispatch]="4G 02:00:00"
    [consensus]="32G 24:00:00"
    [admm]="32G 24:00:00"
    [diffusion]="32G 24:00:00"
    [exact-diffusion]="32G 24:00:00"
    [fdgdm]="32G 24:00:00"
    [deed-admm]="32G 24:00:00"
)

MANIFEST_DIR="${1:?usage: submit_sweep.sh <manifest-dir>}"
[[ "$MANIFEST_DIR" = /* ]] || MANIFEST_DIR="$(cd "$MANIFEST_DIR" && pwd)"
PARTITION="${ESB_PARTITION:-rosa.p}"
ARRAY_MAX="${ESB_ARRAY_MAX:-1000}"

if [[ ! -f "$ESB_ROOT/.venv/bin/activate" ]]; then
    echo "No venv at $ESB_ROOT/.venv — run $HPC_DIR/setup_env.sh first." >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

MAIL_ARGS=()
[[ -n "${ESB_MAIL:-}" ]] && MAIL_ARGS=(--mail-type=END,FAIL "--mail-user=$ESB_MAIL")

run_sbatch() {
    if [[ -n "${ESB_DRY_RUN:-}" ]]; then
        echo "sbatch $*" >&2
        echo "DRYRUN"
    else
        sbatch --parsable "$@"
    fi
}

cd "$ESB_ROOT" # becomes $SLURM_SUBMIT_DIR, from which the job scripts derive ESB_ROOT

JOB_IDS=()
STEMS=()
for manifest in "$MANIFEST_DIR"/*.tsv; do
    scenario="$(basename "$manifest" .tsv)"
    if [[ -z "${RESOURCES[$scenario]:-}" ]]; then
        echo "Unknown scenario manifest: $manifest" >&2
        exit 1
    fi
    read -r mem time <<< "${RESOURCES[$scenario]}"
    lines=$(wc -l < "$manifest")
    offset=0
    while (( offset < lines )); do
        count=$(( lines - offset < ARRAY_MAX ? lines - offset : ARRAY_MAX ))
        job_id="$(run_sbatch \
            --job-name="esb-$scenario" \
            --partition="$PARTITION" \
            --array="1-$count" \
            --mem="$mem" \
            --time="$time" \
            --output="$LOG_DIR/%x-%A_%a.out" \
            "${MAIL_ARGS[@]}" \
            "$HPC_DIR/run_manifest.sbatch" "$manifest" "$offset")"
        job_id="${job_id%%;*}"
        JOB_IDS+=("$job_id")
        printf '  %-18s lines %6d-%-6d job %-10s mem=%-5s time=%s\n' \
            "$scenario" $(( offset + 1 )) $(( offset + count )) "$job_id" "$mem" "$time"
        offset=$(( offset + count ))
    done
    while IFS=$'\t' read -r _ network _; do
        stem="$(basename "$network")"
        STEMS+=("${stem%.*}")
    done < "$manifest"
done

mapfile -t STEMS < <(printf '%s\n' "${STEMS[@]}" | sort -u)
DEPENDENCY="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
agg_id="$(run_sbatch \
    --partition="$PARTITION" \
    --dependency="$DEPENDENCY" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HPC_DIR/aggregate.sbatch" "${STEMS[@]}")"
echo "  compare_costs.py   job ${agg_id%%;*} (runs after all arrays finish)"
echo
echo "Logs:    $LOG_DIR/"
echo "Results: $ESB_ROOT/results/<network>/<run_id>-*  and  $ESB_ROOT/results/runs.csv"
