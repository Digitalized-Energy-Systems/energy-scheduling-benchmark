"""energy-scheduling-benchmark: distributed economic-dispatch algorithms (mango agents) vs. the
central optimum on copper-plate networks.

* :mod:`.problem` — the dispatch problem: ``problem.rules`` (shared by central dispatch and every
  algorithm: pre-dispatched units, grid import price, the no-dispatch window), ``problem.storage``
  (shared storage model + price response), ``problem.qp`` (Clarabel QP canonicalised once and
  re-solved per price: storage response, ADMM bus).
* :mod:`.datasets` — benchmark networks: raw PyPSA / pandapower input + preparation recipe →
  prepared ``.nc`` (``esb-prepare``).
* :mod:`.parameters` — algorithm parameters per tuning route (``resources/tuned_parameters.toml``).
* :mod:`.algorithms` — ``algorithms.central`` (the reference optimum), ``consensus``, ``admm``,
  ``dummy``: one module each, registry ``algorithms.ALGORITHMS``.
* :mod:`.world` — mango world: topology, counted lossy links, bus agents (the first one also the leader).
* :mod:`.run` — one run end-to-end (``esb-run``); :mod:`.outputs` — ``schedule.csv``,
  ``meta.json``, ``trace.csv``.
* :mod:`.tune` — ``esb-tune`` parameter sweep; :mod:`.compare` — ``esb-compare`` cost gaps.
"""
