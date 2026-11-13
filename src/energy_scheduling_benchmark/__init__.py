"""energy-scheduling-benchmark: distributed economic-dispatch algorithms (mango agents) vs. the
central optimum on copper-plate networks.

* :mod:`.problem` — rules shared by central dispatch and every algorithm (pre-dispatched units,
  grid import price, the no-dispatch window).
* :mod:`.datasets` — benchmark networks: raw PyPSA / pandapower input + preparation recipe →
  prepared ``.nc`` (``esb-prepare``).
* :mod:`.central` — central dispatch (the reference optimum); :mod:`.storage` — shared storage
  model + price response.
* :mod:`.parameters` — algorithm parameters per tuning route (``resources/tuned_parameters.toml``).
* :mod:`.consensus`, :mod:`.dummy` — the algorithms (one module each, ``run.ALGORITHMS``).
* :mod:`.world` — mango world: topology, counted lossy links, bus agents (the first one also the leader).
* :mod:`.run` — one run end-to-end (``esb-run``); :mod:`.outputs` — ``schedule.csv``,
  ``meta.json``, ``trace.csv``.
* :mod:`.tune` — ``esb-tune`` parameter sweep; :mod:`.compare` — ``esb-compare`` cost gaps.
"""
