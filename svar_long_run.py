import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import macro_df, climate_df


yl = pd.concat([
    climate_df[["co2", "phdi", "us_tavg_anom"]],
    macro_df[["indpro", "IPMANSICS".lower(), "MANEMP".lower(), "AWHMAN".lower(), "CPIAUCSL".lower(), "gs10"]]
], axis=1)

yl = yl.loc[yl.dropna().index.min():][yl.loc[yl.dropna().index.min():].isna().any(axis=1).cumsum() == 0]