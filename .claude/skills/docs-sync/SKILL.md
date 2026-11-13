---
name: docs-sync
description: Check and update all documentation of the ESB workspace (~/development/esb) — the three READMEs and CLAUDE.md files, TODO.md, docs/*.md (plans, deviations, runbook, restart plan), DRO Sphinx pages, code docstrings/decision comments and the auto-memory — so it matches the code. Use after merging a feature into `restructuring`, or when asked to check, sync or update the docs / README / TODO.
when_to_use: after a merge ("docs-sync", "check the docs", "update README and TODO", "are the docs up to date?")
argument-hint: "[base ref or feature name, optional]"
---

# docs-sync — keep every doc true after a merge

Goal: after a feature lands on `restructuring`, every document states what the code now does,
the TODO says exactly what is next, and every decision taken during the feature is logged.
Change docs only; code changes found necessary are **reported, not made** (ask Jonas).

## 0. Rules (from the workspace CLAUDE.md — they apply here too)

- Smallest edit that makes a statement true; keep each file's style, language (English) and
  structure. No rewrites, no new sections unless a fact has no home.
- Decision log rows are history: never edit old rows' content; mark them superseded in a new
  row (or with "(superseded …)" as the TODO already does). Dates absolute (YYYY-MM-DD).
- Facts only from the code, a test run or git — never from memory or an old doc. Numbers
  (test counts, hashes, sizes, gaps) are re-measured, not copied.
- Repo split (CLAUDE.md "Architecture"): a doc in one repo must not describe another repo's
  internals; MEE/DRO docs never mention the benchmark specifics.
- Open decisions you find are **asked, not decided** (`❓ ASK FIRST` in TODO.md).
- Commit doc changes locally per repo (`Docs: …`); the workspace root (TODO.md, docs/,
  CLAUDE.md) is its own git repo (`master`, since 2026-11-12) — commit there too. **Never `git push`.** In a background session, edit repo files in a
  worktree on `restructuring` and fast-forward merge it back.

## 1. Find what changed

Per repo (`mango-energy-environments`, `distributed-resource-optimization`,
`energy-scheduling-benchmark`), with `BASE` = the argument, else the hashes last recorded in
TODO.md "Status", else `ORIG_HEAD` after the merge:

```bash
git -C <repo> log --oneline BASE..restructuring
git -C <repo> diff --stat BASE..restructuring
git -C <repo> diff BASE..restructuring -- '*.py' '*.toml' '*.sh'
```

From the diff, list the **changed facts**: new/renamed/removed CLI flags and entry points,
function signatures and defaults, constants with values (costs, bands, tolerances), meta.json /
schedule.csv fields, stop reasons, file and folder names, network list and their properties,
test counts, new dependencies, new plan documents. This list drives every check below.

## 2. Check each document against the changed facts

Grep every doc for each old name/value (`grep -rn "<old>" --include='*.md' --include='*.rst'`)
and read each section a fact touches. Per file:

| file | must state correctly |
|---|---|
| `~/development/esb/CLAUDE.md` | layout block (folders, root scripts), the "currently:" pointer to TODO "Start of next session", rules unchanged unless Jonas changed one |
| `~/development/esb/TODO.md` | **Start of next session**: the real next steps in order, nothing already done, open ❓ visible; **Status** section: date, per-repo hashes on `restructuring`, pushed or not, test counts (re-run), open worktrees/branches; feature section (e.g. follow-ups S1…) marked done with commits; **Next** items done/open; **Known limits** still true; **Decision log**: one row per decision taken in the feature (compare with the plan doc's decision table and the session) |
| `docs/<feature>-plan.md` | status line (draft → planned → implemented, with commits), every question answered or explicitly open, design matches what was built (note deviations from the plan) |
| `docs/deviations.md` | every paper deviation the feature added/removed/changed (P*, C*), with reason and decision reference; superseded entries marked |
| `docs/restart-plan.md` | §4 Progress / "Next" pointer |
| `docs/hpc-runbook.md`, root `hpc_sweeps.sh`, `run_all.sh`, `hpc-requirements.txt` | commands, flags, defaults, dates and file names still valid; header comments match |
| `docs/mango-2.4-features.md` | only when mango usage changed |
| other `docs/*.md` (research, reviews, agendas) | leave as dated records; fix only a pointer that is now wrong |
| each repo `README.md` | Scope section (repo split), install, usage/CLI examples (run one if cheap), outputs and meta fields, network list + properties, algorithm list, package layout tree |
| each repo `CLAUDE.md` | scope + "must not contain" still true |
| DRO `docs/source/**/*.rst` | algorithm parameters, API, examples match the code |
| code docstrings / module headers / decision comments | changed functions describe their new behaviour; every decision has its reason at the code (memory: decisions-in-code-comments) |
| auto-memory (`~/.claude/projects/*esb*/memory/*.md`, `MEMORY.md` index) | no memory contradicts the new state; update or delete wrong ones |

Also grep all docs for stale markers: `not merged`, `branch`, `worktree`, `draft`, `open`,
`TODO`, `❓`, `on hold`, `placeholder`, `next:` — each must still be true.

## 3. Verify the numbers

```bash
cd ~/development/esb && source .venv/bin/activate
for r in mango-energy-environments distributed-resource-optimization energy-scheduling-benchmark; do
  (cd $r && pytest -q | tail -1 && ruff check . -q)
done
for r in mango-energy-environments distributed-resource-optimization energy-scheduling-benchmark; do
  git -C $r log --oneline -1; git -C $r status --short; git -C $r worktree list; git -C $r status -sb | head -1
done
```

Use these outputs for test counts, hashes, "pushed" (ahead/behind of origin) and open worktrees.

## 4. Edit, commit, report

1. Edit the files (smallest change). Workspace-root files directly; repo files in their repo.
2. Per repo with changes: `git commit -m "Docs: <what> (docs-sync after <feature>)"` — local only.
3. Report to Jonas: per file what was wrong and what it says now; facts you could not verify;
   code issues found (not fixed); new ❓ questions. End with the next step from TODO "Start".
