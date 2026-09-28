#!/usr/bin/env bash
# Cluster counterpart of run_all_scenarios.sh: submits all seven scenarios against one
# or more networks as SLURM job arrays, then compare_costs.py as a dependent job.
# Run on the ROSA login node.
#
# Usage:
#   energy-scheduling-benchmark/hpc/submit_all.sh --smoke                 # toy network, express partition, 1 day
#   energy-scheduling-benchmark/hpc/submit_all.sh --all [-- flags]        # every .nc in resources/networks/
#   energy-scheduling-benchmark/hpc/submit_all.sh a.nc b.nc [-- flags]    # specific networks
#
# Everything after `--` is passed to every scenario, e.g. `-- --simulate-days 1`.
#
# Environment overrides:
#   ESB_SCENARIOS="esb-admm esb-fdgdm"   only submit these scenarios
#   ESB_PARTITION=mpcb.p                 partition for all arrays (default rosa.p)
#   ESB_MAIL=you@uol.de                  e-mail on END/FAIL of each array
#   ESB_DRY_RUN=1                        print the sbatch commands instead of running them

set -euo pipefail

ESB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_DIR="$ESB_ROOT/energy-scheduling-benchmark/hpc"
NETWORKS_DIR="$ESB_ROOT/energy-scheduling-benchmark/resources/networks"
LOG_DIR="$ESB_ROOT/hpc-logs"

# Per-scenario resource requests: "memory walltime". These are deliberately generous
# first guesses — after the first sweep, tighten them with `seff <jobid>_<task>`.
# esb-admm is large because mango retains every message's content: it grows by
# ~35 MB per ADMM iteration (~70 GB at --max-iter 2000 on lower_saxony_2030).
declare -A RESOURCES=(
    [esb-central-dispatch]="4G 01:00:00"
    [esb-consensus]="32G 24:00:00"
    [esb-admm]="96G 24:00:00"
    [esb-diffusion]="32G 24:00:00"
    [esb-exact-diffusion]="32G 24:00:00"
    [esb-fdgdm]="32G 24:00:00"
    [esb-deed-admm]="32G 24:00:00"
)
ALL_SCENARIOS=(esb-consensus esb-central-dispatch esb-admm esb-diffusion esb-exact-diffusion esb-fdgdm esb-deed-admm)

PARTITION="${ESB_PARTITION:-rosa.p}"
NETWORKS=()
EXTRA=()
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --all) mapfile -t NETWORKS < <(find "$NETWORKS_DIR" -maxdepth 1 -name '*.nc' -printf '%f\n' | sort) ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)
            name="$(basename "$1")"
            if [[ ! -f "$NETWORKS_DIR/$name" ]]; then
                echo "Network file not found: $NETWORKS_DIR/$name" >&2
                exit 1
            fi
            NETWORKS+=("$name")
            ;;
    esac
    shift
done

if [[ $SMOKE -eq 1 ]]; then
    NETWORKS=(toy)
    PARTITION="${ESB_PARTITION:-rosa_express.p}"
    EXTRA=(--simulate-days 1 "${EXTRA[@]}")
fi

if [[ ${#NETWORKS[@]} -eq 0 ]]; then
    echo "No networks selected. Pass --smoke, --all, or network file names (see --help)." >&2
    exit 1
fi
if [[ ! -f "$ESB_ROOT/.venv/bin/activate" ]]; then
    echo "No venv at $ESB_ROOT/.venv — run $HPC_DIR/setup_env.sh first." >&2
    exit 1
fi

read -r -a SCENARIOS <<< "${ESB_SCENARIOS:-${ALL_SCENARIOS[*]}}"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
NETWORK_LIST="$LOG_DIR/networks-$STAMP.txt"
printf '%s\n' "${NETWORKS[@]}" > "$NETWORK_LIST"
N=${#NETWORKS[@]}

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

echo "Networks ($N): ${NETWORKS[*]}"
echo "Partition: $PARTITION   extra flags: ${EXTRA[*]:-(none)}"
JOB_IDS=()
for scenario in "${SCENARIOS[@]}"; do
    if [[ -z "${RESOURCES[$scenario]:-}" ]]; then
        echo "Unknown scenario: $scenario" >&2
        exit 1
    fi
    read -r mem time <<< "${RESOURCES[$scenario]}"
    if [[ $SMOKE -eq 1 ]]; then
        mem=8G
        time=01:00:00 # rosa_express.p allows at most 2 h
    fi
    job_id="$(run_sbatch \
        --job-name="$scenario" \
        --partition="$PARTITION" \
        --array="1-$N" \
        --mem="$mem" \
        --time="$time" \
        --output="$LOG_DIR/%x-%A_%a.out" \
        "${MAIL_ARGS[@]}" \
        "$HPC_DIR/run_scenario.sbatch" "$scenario" "$NETWORK_LIST" "${EXTRA[@]}")"
    job_id="${job_id%%;*}"
    JOB_IDS+=("$job_id")
    printf '  %-22s job %-10s mem=%-5s time=%s\n' "$scenario" "$job_id" "$mem" "$time"
done

STEMS=()
for net in "${NETWORKS[@]}"; do
    stem="$(basename "$net")"
    STEMS+=("${stem%.*}")
done
DEPENDENCY="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
agg_id="$(run_sbatch \
    --partition="$PARTITION" \
    --dependency="$DEPENDENCY" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HPC_DIR/aggregate.sbatch" "${STEMS[@]}")"
echo "  compare_costs.py       job ${agg_id%%;*} (runs after all arrays finish)"
echo
echo "Logs:    $LOG_DIR/"
echo "Results: $ESB_ROOT/results/<network>/  and  $ESB_ROOT/results/cost_comparison.csv"
echo "Monitor: squeue --me    |    sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS    |    seff <jobid>_<task>"
