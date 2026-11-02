"""energy-scheduling-benchmark: distributed economic-dispatch algorithms (mango agents) vs. the
central optimum on copper-plate networks.

* :mod:`.datasets` — benchmark networks (raw PyPSA input + preparation recipe).
* :mod:`.world` — mango world: topology, lossy links, leader/slack agents.
* :mod:`.run` — one run end-to-end (``esb-run``); :mod:`.outputs` — ``schedule.csv`` + ``meta.json``.
* :func:`stacked_area`, :func:`visualize_results` — matplotlib plotting utilities.
"""

from .plotting import stacked_area, visualize_results

__all__ = ["stacked_area", "visualize_results"]
