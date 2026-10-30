# CLAUDE.md — energy-scheduling-benchmark (ESB)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.

## Scope (IMPORTANT — check every change against it)

- Owns: the benchmark — economic-dispatch problem formulation (costs incl. quadratic, power
  balance, bounds, storage model), central-dispatch reference optimum, mapping MEE networks onto
  DRO's generic problem interface, benchmark roles (leader, loads), concrete datasets
  (network selection, lower_saxony_2030), runs, CSV + meta outputs, comparison.
- Must not contain: network representation/simulation (→ MEE) or generic algorithm logic (→ DRO).

Repo split (all three): see `README.md` § Scope of this package.
