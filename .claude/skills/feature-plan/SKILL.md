---
name: feature-plan
description: Start a feature of the ESB workspace (~/development/esb) the agreed way — write the plan skeleton docs/<feature>-plan.md (paper reference, decided items, ❓ questions, design, deviations, steps), ask Jonas how before anything is built, then record the answers as one-line decision-log rows. Use when a new feature, algorithm or follow-up is about to be planned ("plan S6", "new feature", "write the plan for …", "ask how").
when_to_use: before implementing any feature ("plan …", "feature-plan", "how should we build …", a TODO "Then" item starts)
argument-hint: "<feature id and name, e.g. S6 ADMM>"
---

# feature-plan — plan first, ask how, log once

The rules this codifies (workspace `CLAUDE.md` → How we work): *ask how before each feature*,
*ask first at every open decision*, *check mango first*, *paper fidelity*, *log every decision
once*. Nothing is implemented here; implementation follows later (ponytail, worktree branch).

## 1. Gather (read, don't write yet)

- `TODO.md` → Start and the feature's section; the decision log (`grep -n "<topic>" TODO.md`) for
  decisions already taken (cite them by `D-nnn`, never re-ask).
- `docs/deviations.md` for deviations that already touch the feature.
- The papers in `docs/references/` the feature implements — note equations and sections.
- If anything mango-related: the full online docs sweep (mango 2.4.0) + `docs/mango-2.4-features.md`.
- The archive `~/development/esb-archive-2026-10-30/` only as a reference for what existed.

## 2. Write the skeleton `docs/<feature>-plan.md`

```markdown
# Plan <id> — <feature> (<YYYY-MM-DD>)

Status: **draft** — questions Q1–Qn open (❓ ASK FIRST).

Source: <who/what asked for it: TODO item, meeting, audit question>; earlier decisions D-…

## 1. Paper / reference
<what each paper does, with eq./§; what it leaves open>

## 2. Already decided (re-used, not re-asked)
<D-nnn rows and deviation ids that apply>

## 3. Questions for Jonas (❓ ASK FIRST)
| # | question | options (recommended first) | answer |
|---|---|---|---|
| Q1 | … | A … / B … | |

## 4. Design (the how — written after the answers; lives here, not in the decision log)

## 5. Deviations from the papers (→ rows for docs/deviations.md)

## 6. Steps + tests (one worktree branch per repo, same name)
```

Add the plan's row to `docs/README.md` (type `plan <id>`, status `draft`).

## 3. Ask how

One message to Jonas with every question of §3, each with its options and a recommendation
with the reason. Then stop — never pick an option yourself, never start implementing.

## 4. After the answers

1. Fill the `answer` column and §4 Design; `Status:` → **planned** (date).
2. Decision log: one **one-line** row per answer (date, next `D-nnn`, what + why,
   `where` = `docs/<feature>-plan.md §4` or the deviation id). The how stays in the plan.
3. `docs/deviations.md`: a row per new deviation, with the D-id. `docs/README.md`: status.
4. `TODO.md` → Start → Then: the feature's next step, nothing else.
5. Implementation later: ponytail, worktree branch, one-line decision comments with the D-ids at
   the code; after Jonas's merge `docs-sync` flips `Status:` → **implemented and merged**.
