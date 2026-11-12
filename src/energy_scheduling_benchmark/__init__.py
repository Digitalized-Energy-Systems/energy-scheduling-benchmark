"""energy-scheduling-benchmark: distributed economic-dispatch algorithms (mango agents) vs. the
central optimum on copper-plate networks.

* :mod:`.datasets` — benchmark networks: raw PyPSA input + preparation recipe → prepared ``.nc``
  (``esb-prepare``).
* :mod:`.central` — central dispatch (the reference optimum); :mod:`.storage` — shared storage model.
* :mod:`.world` — mango world: topology, lossy links, bus + leader agents.
* :mod:`.run` — one run end-to-end (``esb-run``); :mod:`.outputs` — ``schedule.csv`` + ``meta.json``.
"""
