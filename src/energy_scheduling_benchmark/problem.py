"""Rules of the benchmark problem shared by central dispatch and every algorithm.

Own module (audit 2026-11-12 Q-A12b): they used to live in ``world.py`` (the mango world), so
central dispatch and the datasets imported the world for a problem rule.
"""

from __future__ import annotations

from mango_energy_environments import RENEWABLE, THERMAL

#: Slack import price (€/MWh), export is free (Q3: SDAC harmonised maximum clearing price).
SLACK_IMPORT_PRICE = 4000.0


def predispatched(kind: str, c1: float) -> bool:
    """Runs at its available power p_max before any algorithm: renewables and free thermal units
    (marginal cost 0 → always first; KH12 eq. 27 / HKW15 §II must-take). Decided 2026-11-10
    (supervisor meeting, `docs/renewables-predispatch-plan.md` D1): no curtailment — a surplus is
    exported by the slack or reported."""
    return kind == RENEWABLE or (kind == THERMAL and c1 <= 0)
