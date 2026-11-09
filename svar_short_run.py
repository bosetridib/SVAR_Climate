import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import macro_df, climate_df, fred_macro_variables


ys = pd.concat([
    climate_df[["us_tavg_anom", "us_tmax_anom", "zndx"]],
    macro_df[["indpro", "iputil", "DPCERA3M086SBEA".lower(), "cpiaucsl", "CUSR0000SA0L2".lower(), "unrate", "fedfunds", "credit_spread"]]
], axis = 1)

ys.rename(columns={"dpcera3m086sbea": "rpce", "cusr0000sa0l2": "cpi_urban"}, inplace=True)

ys = ys.loc[ys.dropna().index.min():][ys.loc[ys.dropna().index.min():].isna().any(axis=1).cumsum() == 0]

# Necessary transformations
ys_delta = pd.DataFrame(index = ys.index)
# Climate
ys_delta[['us_tavg_anom', 'us_tmax_anom']] = ys[['us_tavg_anom', 'us_tmax_anom']]

ys_delta['d_us_tavg_anom'] = ys['us_tavg_anom'].diff()
ys_delta['d_us_tmax_anom'] = ys['us_tmax_anom'].diff()
ys_delta['zndx'] = ys['zndx']
# Macroeconomic
ys_delta['dln_indpro'] = np.log(ys['indpro']).diff() * 100.0
ys_delta['dln_iputil'] = np.log(ys['iputil']).diff() * 100.0
ys_delta['dln_rpce'] = np.log(ys['rpce']).diff() * 100.0

ys_delta['infl_cpiaucsl'] = np.log(ys['cpiaucsl']).diff() * 100.0
ys_delta['infl_cpi_urban'] = np.log(ys['cpi_urban']).diff() * 100.0

ys_delta['d_unrate'] = ys['unrate'].diff()
ys_delta['d_fedfunds'] = ys['fedfunds'].diff()
ys_delta['credit_spread'] = ys['credit_spread']

ys_delta = ys_delta.dropna()

# DEFINE VARIABLE BLOCKS FOR DOWNSTREAM PIPELINE STAGES
# Endogenous stationary macroeconomic vector (y_t)
macro_variables = [
    'dln_indpro', 'dln_iputil', 'dln_rpce', 'infl_cpiaucsl',
    'infl_cpi_urban', 'd_unrate', 'd_fedfunds', 'credit_spread'
]
# Physical climate context / surrogate vector (u_t / C_t) for CDANs & SVAR
climate_variables = ['us_tavg_anom', 'us_tmax_anom', 'zndx']
# Purely stationary climate increments for Stage 1 J_LZ complexity screening
climate_variables_stationary = ['d_us_tavg_anom', 'd_us_tmax_anom', 'zndx']


# ========================================================================
# ===========================stage 1======================================
# ========================================================================






#ys_delta.plot(subplots=True, layout=(4, 4)); plt.show()
