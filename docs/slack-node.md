# Slack node (`--slack`)

An optional slack node that closes the power balance for every scenario. It
imports any shortfall and exports any surplus at fixed prices, standing in for the
point of common coupling (PCC) of a grid-connected microgrid. It is **off by
default**; without `--slack` every output is identical to a run without this
feature.

## Why

The PyPSA-Eur extracts in `resources/networks/` are islanded: nothing outside the
network can absorb a surplus or cover a shortfall, so every scenario has to match
demand exactly with its own units.

That is not the setting of every algorithm's source. Bai et al. 2022 (FDGDM) model a
grid-connected microgrid whose balance is

```
P_re(t) + Σ P_c,i(t) + P_PCC(t) = P_l(t)        (Eq. 3)
```

Wind and PV run in MPPT mode (always at maximum output), and the PCC "meets the
instantaneous fluctuation of the MG" (§4.1). When load drops out, "the PCC is
absorbing energy" (§4.2). The paper never curtails renewables.

`esb-fdgdm` follows that design and pre-schedules renewables at full availability.
On islanded networks this over-generates whenever renewables exceed demand: on
ROSA, fdgdm was excluded on `base_s_1_elec_2019` (8 % imbalance),
`base_s_5_elec_2019` (10 %) and `lower_saxony_2030` (276 %). The slack gives such
runs the PCC the algorithm assumes, for all seven scenarios alike, instead of
patching FDGDM with a curtailment step that is not in the paper.

## What it does

The slack is **passive**: the distributed algorithms do not know it exists and
dispatch exactly as before. After a run, the leader (`PowerLoadAggregator`, which
every scenario uses) compares demand with the schedules it applied:

```
residual(t) = demand(t) − Σ schedule_i(t)
import(t)   = max(0,  residual(t))     # slack supplies the shortfall
export(t)   = max(0, −residual(t))     # slack absorbs the surplus
```

The slack is computed from the applied schedules rather than from the `P:`
recordings, because FDGDM clips its recording at 0 and would hide storage
charging.

For each run with `--slack`:

* `cost:total` includes `import × import price + export × export price`.
* The CSV gets two extra columns, `slack:import` and `slack:export` (MW per hour).
* The log reports the slack energy and its share of demand, instead of flagging a
  balance gap as "did not converge".
* The plot annotations show the slack energy.

**Central dispatch is the exception**, because an LP has to know the slack to stay
the optimum of the same problem. `solve_multiperiod_dispatch(..., slack=SlackSpec(...))`
adds `s_imp, s_exp ≥ 0` to every timestep's balance,

```
Σ p_i(t) + Σ (p_d − p_c)(t) + s_imp(t) − s_exp(t) = demand(t)
```

priced at the same import/export prices. Export additionally carries a tie-break of
0.001 €/MWh in the objective only (not in the reported cost), so the LP curtails
rather than exports when both cost the same. With the slack, the LP is also always
feasible, even when capacity falls short.

## Settings

| Flag | Default | Why |
|---|---|---|
| `--slack` | off | Keeps existing results comparable; turn it on per sweep. |
| `--slack-import-price` | 3000 €/MWh | Must exceed every generator's marginal cost (the dearest on these networks is oil at 84–133 €/MWh), otherwise central dispatch imports instead of generating. It works like a value-of-lost-load penalty: a distributed run that under-supplies pays for it. |
| `--slack-export-price` | 0 €/MWh | Exporting costs the same as curtailing. A positive price (being paid for exports) would make central dispatch generate extra just to sell it. |

`--slack` cannot be combined with `--strict`: the slack closes every gap, so there
is nothing left for `--strict` to enforce.

## Reading the results

* **Runs with a slack are never excluded** by `compare_costs.py`: their balance is
  closed by construction. Instead, `compare_costs.py` writes
  `results/slack_usage.csv` with each run's import/export energy and share of demand.
* **Import** is demand the algorithm left unserved; it is expensive by design, so it
  shows up directly in the cost.
* **Export** is over-generation; at the default price 0 it is free, so the cost only
  contains what the exporting units spent generating it (e.g. biomass at 20 €/MWh).
  For FDGDM this is typically renewable surplus it could not curtail.
* Central dispatch should use no slack at all on networks with enough capacity;
  compare its import/export to 0 as a sanity check.

## Usage

```bash
esb-fdgdm --network resources/networks/lower_saxony_2030.nc --name-base fdgdm --slack

# different prices
esb-consensus --network ... --slack --slack-import-price 10000 --slack-export-price 0

# on the ROSA cluster: everything after -- goes to every scenario
energy-scheduling-benchmark/hpc/submit_all.sh --all -- --slack
```

`run_all_scenarios.sh` (monorepo root) does not forward extra flags yet.

## Limitations

* **Passive:** no algorithm reacts to the slack, and its price signal is not fed back
  into the distributed iteration. In Bai et al. the PCC output is reported to the
  controllable agents; here it only closes the balance afterwards.
* **Import price dominates:** a run that under-supplies by 1 % of demand pays roughly
  3000 × that energy, which can outweigh all generation cost. That is intended (it
  ranks unconverged runs last) but means costs with import are not generation costs.
* **FDGDM's `P:` columns** clip storage charging at 0, so for FDGDM
  `Σ P + import − export` does not reproduce demand exactly in charging hours.
  The slack columns themselves are exact.

## Implementation

| File | Change |
|---|---|
| `dispatch/pyomo_dispatch.py` | `SlackSpec(import_price, export_price)`; slack variables, tie-break and `slack_import`/`slack_export` in `solve_multiperiod_dispatch` |
| `scenarios/_common.py` | CLI flags; `compute_slack`; `PowerLoadAggregator(slack=...)` with `schedule_residual()` and `slack_by_bucket()`; `write_scenario_outputs(slack=...)` adds cost, CSV columns and annotation |
| `scenarios/central_dispatch.py` | Passes the slack to the LP and logs the LP's slack |
| the other six `scenarios/*.py` | `execute_test_case(..., slack=None)`, passed to the aggregator and outputs |
| `compare_costs.py` (monorepo root) | Skips the balance exclusion for slack runs; writes `results/slack_usage.csv` |
| `tests/test_dispatch.py`, `tests/test_common.py`, `tests/test_slack_scenarios.py` | LP, helper and end-to-end tests |
