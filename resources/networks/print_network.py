import pypsa
import matplotlib.pyplot as plt
import pandas as pd

network = pypsa.Network("base_s_1_elec_2020.nc")

print(network.components)
#print(network.generators)
with pd.option_context("display.max_rows", 100, "display.max_columns", 100):
    print(network.storage_units)

## How to create new networks
## enter pixi shell
## snakemake resources/networks/base_s_5_elec_2020.nc --configfile config/config.elec_2020.yaml -j 10

