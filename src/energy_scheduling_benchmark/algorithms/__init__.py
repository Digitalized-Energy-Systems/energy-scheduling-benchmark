"""The algorithms of the benchmark: one module each, listed in :data:`ALGORITHMS`.

Algorithm interface (audit 2026-11-12 Q-A4, decided by Jonas: module convention): every
distributed algorithm is one module here with
``setup(params, network, full, args) -> (behavior, agents, routes) -> outcome`` that adds its mango
roles to the world's agents and returns ``outcome() -> outputs.Outcome``; its parameters are the
``[<algorithm>]`` section of ``resources/tuned_parameters.toml`` (``--tuning``, ``--set``); if it
can be tuned it also has ``sweep_grid``, ``SWEEP_RUN``, ``VARIANT`` and ``RANK_BY_ITERATIONS``
(``esb-tune``). Control messages (start, stop, reports) are marked with DRO's ``ControlMessage``
and never lost. Central dispatch (:mod:`.central`, the reference optimum) is the one case without
a world.
"""

# 2026-11-17 D-145: algorithms in their own subpackage, the registry moved here from run.py
from . import admm, consensus, dummy

#: name → algorithm module (None: central dispatch, no world). Add a new algorithm here.
ALGORITHMS = {"central": None, "consensus": consensus, "admm": admm, "dummy": dummy}
