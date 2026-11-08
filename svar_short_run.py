import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import macro_df, climate_df


ys = pd.concat([
    climate_df[["us_tavg_anom", "us_tmax_anom", "zndx"]],
    macro_df[["indpro", "iputil", "DPCERA3M086SBEA".lower(), "cpiaucsl", "CUSR0000SA0L2".lower(), "unrate", "fedfunds", "credit_spread"]]
], axis = 1)

ys.rename(columns={"dpcera3m086sbea": "rpce", "cusr0000sa0l2": "cpi_urban"}, inplace=True)

ys = ys.loc[ys.dropna().index.min():][ys.loc[ys.dropna().index.min():].isna().any(axis=1).cumsum() == 0]