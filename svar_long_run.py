import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import macro_df, climate_df, fred_macro_variables


yl = pd.concat([
    climate_df[["co2", "phdi", "us_tavg_anom"]],
    macro_df[["indpro", "IPMANSICS".lower(), "MANEMP".lower(), "AWHMAN".lower(), "CPIAUCSL".lower(), "gs10"]]
], axis=1)

yl = yl.loc[yl.dropna().index.min():][yl.loc[yl.dropna().index.min():].isna().any(axis=1).cumsum() == 0]

raw_cols = [
    "co2", "phdi", "us_tavg_anom", "indpro", "ipmansics",
    "manemp", "awhman", "cpiaucsl", "gs10"
]

# Create a container DataFrame for the transformed Long-Run System
yl_delta = pd.DataFrame(index=yl.index)

