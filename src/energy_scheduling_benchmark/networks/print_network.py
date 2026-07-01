import pypsa
import matplotlib.pyplot as plt
import pandas as pd

network = pypsa.Network("base_s_1_elec_2020.nc")

print(network.components)
#print(network.generators)
with pd.option_context("display.max_rows", 100, "display.max_columns", 100):
    print(network.storage_units)

