"""Quick inspection helper: print the components of a PyPSA-Eur extract.

Usage: python print_network.py [network.nc]

How to create new networks (from the pypsa-eur checkout):
  enter pixi shell, then
  snakemake resources/networks/base_s_5_elec_2020.nc --configfile config/config.elec_2020.yaml -j 10
"""

import sys

import pypsa

network = pypsa.Network(sys.argv[1] if len(sys.argv) > 1 else "base_s_1_elec_2020.nc")

print(network.components)
print(network.generators)
