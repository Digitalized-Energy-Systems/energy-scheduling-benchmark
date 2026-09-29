# Running the benchmark on the Oldenburg HPC cluster (ROSA)

Step-by-step guide for running the seven scheduling scenarios on **ROSA**, the
multi-purpose HPC cluster of the University of Oldenburg (it replaced CARL/EDDY).
The scripts in this directory are the cluster counterpart of the local
`run_all_scenarios.sh`: same networks, same flags, same `results/<network>/` layout, so
`compare_costs.py` works unchanged.

Background, measurements and the reasoning behind the resource numbers are in the
design notes at `docs/hpc-oldenburg-plan.md` in the monorepo root. Cluster
documentation: <https://hpcwiki.uni-oldenburg.de> (the old `wiki.hpcuser.uni-oldenburg.de`
describes CARL and is outdated).

## What needs to be done — overview

| # | Step | Where | Once / every time |
|---|------|-------|-------------------|
| 1 | Get an HPC account | web form | once |
| 2 | Log in via SSH (VPN from off-campus) | workstation | every time |
| 3 | Copy the monorepo to `$WORK` | workstation → cluster | after every code change |
| 4 | Build the Python environment (`setup_env.sh`) | login node | once (and when dependencies change) |
| 5 | Smoke test (`submit_all.sh --smoke`) | login node → compute nodes | once |
| 6 | Submit the real sweep (`submit_all.sh --all`) | login node → compute nodes | per experiment |
| 7 | Monitor, re-run failures, tune resources | login node | per experiment |
| 8 | Copy results back / archive them | cluster → workstation | per experiment |

Nothing in the code has to change: the scenarios are single-process Python programs,
and parallelism on the cluster comes from running many of them at once as SLURM
**job arrays** (one array per scenario, one array task per network).

## Files in this directory

| File | Purpose |
|------|---------|
| `setup_env.sh` | One-time: creates `<root>/.venv` from the cluster's Python module (or Python 3.14 via Miniforge3 with `--conda`) and installs all four packages editable |
| `env.sh` | Sourced by every job: loads the recorded modules, activates the venv, pins BLAS/HiGHS threads, sets headless matplotlib |
| `submit_all.sh` | Submits all scenarios × networks as job arrays plus a dependent `compare_costs.py` job |
| `run_scenario.sbatch` | The array job: task *N* runs one scenario on line *N* of a network list |
| `aggregate.sbatch` | Runs `compare_costs.py` after the arrays finish |

---

## 1. Get an HPC account

1. Go to <https://konto.uol.de/> → *Application* → **Zugang zu HPC** and select your
   research group. Activation usually takes about a day.
2. Your cluster login is your university account (`abcd1234`).
3. For questions (quota, which partitions your group may use): `hpcsupport@uol.de`.

## 2. Log in

From outside the campus network, connect to the university **VPN** first.

```bash
ssh abcd1234@rosa.hpc.uni-oldenburg.de
```

Tip: add an entry to `~/.ssh/config` on your workstation and copy your key over once:

```
Host rosa
    HostName rosa.hpc.uni-oldenburg.de
    User abcd1234
```

```bash
ssh-copy-id rosa
```

The **login node** is for editing, installing and submitting only. Anything that
computes runs on compute nodes via SLURM (`sbatch`/`srun`).

### Where to put things

| Variable | Backup | Use it for |
|----------|--------|-----------|
| `$HOME` (`/user/abcd1234`) | yes | notes, personal copies of job scripts |
| `$WORK` (`/dss/work/abcd1234`) | **no** | **the monorepo, the venv, all job output** — fast parallel file system |
| `$DATA` (`/nfs/data/abcd1234`) | yes | archiving finished results (read-only on compute nodes!) |

Everything below assumes the monorepo lives at `$WORK/esb`. Check the real path once
with `ssh rosa 'echo $WORK'`.

## 3. Copy the monorepo to the cluster

The scripts need the **whole monorepo layout**, not just this package:

```
$WORK/esb/
  mango/  mango-energy-environments/  distributed-resource-optimization/
  energy-scheduling-benchmark/        # contains hpc/ and resources/networks/*.nc
  compare_costs.py  run_all_scenarios.sh
```

**Recommended: rsync from the workstation.** It copies your current working state
(uncommitted changes and unpushed branches included), which a `git clone` would not:

```bash
# on the workstation, in the monorepo root (the folder containing CLAUDE.md)
rsync -av \
  --exclude '.venv/' --exclude 'pypsa-eur/' --exclude '_archive/' --exclude 'results/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude '.claude/' --exclude '.idea/' \
  ./ rosa:/dss/work/abcd1234/esb/
```

That is roughly 100 MB. `pypsa-eur/` (28 GB) is not needed for running scenarios.
Re-run the same command after every code change — the packages are installed in
editable mode, so no reinstall is needed unless dependencies change. (`--delete` is
deliberately left out: it would not touch the excluded `.venv/`, but it would delete
cluster-side results/logs if you ever stop excluding them.)

Alternative: `git clone` the four repositories into `$WORK/esb` on the login node (it
has internet access; you need an SSH key registered with GitHub) and check out the
branch you want. `compare_costs.py` and `run_all_scenarios.sh` live in the unversioned
monorepo root, so copy those two with `scp` in that case.

## 4. Build the Python environment (once)

On the **login node** (compute nodes have no internet, so `pip` only works here):

```bash
cd $WORK/esb
energy-scheduling-benchmark/hpc/setup_env.sh
```

This loads `hpc-env/13.1` + the newest `Python` module, creates `$WORK/esb/.venv`
(the same place `run_all_scenarios.sh` expects), installs the pinned versions from
`energy-scheduling-benchmark/requirements.txt`, then `mango`,
`mango-energy-environments`, `distributed-resource-optimization` and
`energy-scheduling-benchmark` in editable mode, and records the loaded modules in
`.venv/esb-modules` so every job loads exactly the same ones.

The pinned requirements need **Python ≥ 3.11**; the script stops with a message if the
module is older. In that case, or if you want the exact interpreter of the workstation
(3.14), use Miniforge3 instead:

```bash
energy-scheduling-benchmark/hpc/setup_env.sh --conda
```

Afterwards check the file-count quota — a venv is ~30 000 files and `$WORK` has an
inode limit:

```bash
lastquota
```

## 5. Smoke test

**a) Interactively**, to see errors directly (express partition, max. 2 h):

```bash
cd $WORK/esb
srun -p rosa_express.p --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 --pty bash -l
# now on a compute node:
source energy-scheduling-benchmark/hpc/env.sh
esb-central-dispatch --network toy --name-base smoke --simulate-days 1
pytest energy-scheduling-benchmark/tests/ -q     # optional, recommended on first setup
exit
```

It should produce `smoke-df.csv` and a few PDFs in the current directory (delete them
afterwards).

**b) Through the batch system**, all seven scenarios on the toy network:

```bash
cd $WORK/esb
energy-scheduling-benchmark/hpc/submit_all.sh --smoke
squeue --me
```

When all jobs are gone from `squeue`, you should have `results/toy/<scenario>-df.csv`
for every scenario, logs in `hpc-logs/`, and the table in `results/cost_comparison.csv`
(printed at the end of `hpc-logs/esb-aggregate-<jobid>.out`).

## 6. Run the benchmark

```bash
cd $WORK/esb
energy-scheduling-benchmark/hpc/submit_all.sh --all                      # every .nc in resources/networks/
energy-scheduling-benchmark/hpc/submit_all.sh base_s_5_elec_2019.nc lower_saxony_2030.nc
energy-scheduling-benchmark/hpc/submit_all.sh --all -- --simulate-days 1 # flags after -- go to every scenario
```

Useful environment overrides:

```bash
ESB_SCENARIOS="esb-admm esb-fdgdm" .../submit_all.sh --all   # only some scenarios
ESB_PARTITION=mpcb.p               .../submit_all.sh --all   # big-memory nodes (1.5 TB)
ESB_MAIL=first.last@uol.de         .../submit_all.sh --all   # mail on END/FAIL
ESB_DRY_RUN=1                      .../submit_all.sh --all   # only print the sbatch commands
```

What gets submitted: one job array per scenario (array size = number of networks),
each with its own memory/time request from the `RESOURCES` table at the top of
`submit_all.sh`, plus an `aggregate` job that waits for all arrays (`afterany`, so it
also runs when single tasks fail — exactly like `run_all_scenarios.sh`, which skips
failing scenarios) and then runs `compare_costs.py`. Runs with too large a
power imbalance are marked non-converged there, as locally.

Output:

| What | Where |
|------|-------|
| scenario results | `$WORK/esb/results/<network>/<scenario>-df.csv`, `-*.pdf` |
| cost comparison | `$WORK/esb/results/cost_comparison.csv` |
| logs (one per task) | `$WORK/esb/hpc-logs/<scenario>-<jobid>_<task>.out` |
| network list of a submission | `$WORK/esb/hpc-logs/networks-<timestamp>.txt` (task *N* = line *N*) |

A new submission overwrites the files of the same network/scenario in `results/`.
Move or rename `results/` between experiments you want to keep apart.

Alternative for a quick one-off: `run_all_scenarios.sh --all` also works on the cluster
(it uses the same `.venv`), but it runs everything serially inside one job — wrap it
in an `sbatch` with enough `--time`/`--mem` if you want that.

## 7. Monitor, re-run failures, tune resources

```bash
squeue --me                                            # pending / running jobs
sacct -j <jobid> -o JobID,JobName,State,Elapsed,MaxRSS,ExitCode
seff <jobid>_<task>                                    # CPU and memory efficiency of one task
scancel <jobid>                                        # cancel a whole array
tail -f hpc-logs/esb-admm-<jobid>_1.out
```

States to look out for in `sacct`: `OUT_OF_MEMORY` → raise `mem` for that scenario;
`TIMEOUT` → raise `time`. Edit the `RESOURCES` table in `submit_all.sh`.

`slurmstepd: error: execve(): esb-...: No such file or directory` in a log means the
`srun` step did not get the venv on its `PATH`. `run_scenario.sbatch` therefore calls
`srun --export=ALL`; keep that flag if you write your own job scripts with
`--export=NONE`.

**Re-run only the failed tasks** of one scenario (task numbers = lines in the
network list of that submission), from `$WORK/esb`:

```bash
sbatch --job-name=esb-admm --partition=rosa.p --array=2,5 --mem=192G --time=48:00:00 \
  --output=hpc-logs/%x-%A_%a.out \
  energy-scheduling-benchmark/hpc/run_scenario.sbatch esb-admm hpc-logs/networks-<timestamp>.txt
```

Then refresh the comparison (cheap, fine on the login node):

```bash
source energy-scheduling-benchmark/hpc/env.sh && python compare_costs.py
```

### About the resource numbers

The defaults in `submit_all.sh` are deliberately generous first guesses — tighten them
with `seff` after the first sweep; smaller requests start sooner.

* Every run is **one single-threaded process**, so each task asks for 1 CPU. Never
  request a whole node for one run.
* `esb-central-dispatch` needs < 1 GB and seconds to minutes.
* **`esb-admm` is memory-hungry**: mango keeps the full content of every delivered
  message, and ADMM messages carry an `(n, m)` matrix, so memory grows ~35 MB per ADMM
  iteration — about 70 GB at the default `--max-iter 2000` on `lower_saxony_2030`
  (measured locally; details and a proposed fix in `docs/hpc-oldenburg-plan.md` §3.1).
  Hence the 96 GB default. `--max-iter` lowers memory proportionally.
* The other iterative scenarios have not been profiled yet; 32 GB / 24 h is a guess.
* Default limits on ROSA if you do not set them: **2 h walltime** and **6000 MB per CPU**.
  Maximum walltime on `rosa.p`/`mpcs.p`/`mpcb.p` is 7 days; `rosa_express.p` is 2 h.

## 8. Get the results back

On the workstation:

```bash
rsync -av rosa:/dss/work/abcd1234/esb/results/ ./results-rosa/
```

To archive on the cluster (`$WORK` has no backup; `$DATA` is read-only on compute
nodes, so do this on the login node):

```bash
cp -r $WORK/esb/results $DATA/esb-results-$(date +%F)
```

## Things to know

* **No internet on compute nodes.** Use the `.nc` files in `resources/networks/` or
  `toy`. PyPSA example networks (`--network ac-dc-meshed`, …) are downloaded on first
  use; to use them in jobs, load each once on the login node
  (`python -c "import pypsa; pypsa.examples.ac_dc_meshed()"`), which caches it under
  `~/.cache/pypsa`, visible to the compute nodes.
* **Reproducibility.** With `--loss-percent 0` (always passed by these scripts) and the
  constant `--delay-s`, runs are deterministic, so cluster results should match the
  workstation. Comparing `cost_comparison.csv` for a network you already ran locally is
  a good acceptance test.
* **`--export=NONE`** is set in the job scripts on purpose (recommended by the HPC team):
  jobs do not inherit your login shell's modules, which avoids `Illegal instruction`
  errors from architecture-specific builds. Everything a job needs is set up by
  `env.sh`. Submit from `$WORK/esb`; the scripts find the monorepo via
  `$SLURM_SUBMIT_DIR`.
* **Threads are pinned** (`OMP_NUM_THREADS` etc. = `--cpus-per-task`). Without that,
  BLAS/HiGHS detect all 128 cores and many tasks sharing a node slow each other down.
* **PyPSA-Eur** (building new networks) is a separate toolchain and not covered here;
  see `docs/hpc-oldenburg-plan.md` §8.

## References

* HPC wiki: <https://hpcwiki.uni-oldenburg.de> — Quickstart, *Submitting Jobs with Slurm*,
  *Partitions and Resources*, *Virtual Environments and Conda*, *Available File Systems*
* ROSA overview: <https://uol.de/en/school5/sc/high-perfomance-computing/hpc-facilities/rosa>
* Support: `hpcsupport@uol.de`
