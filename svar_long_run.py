import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import macro_df, climate_df


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

# -------------------------------------------------------------------------
# 1. CLIMATE THRESHOLD VARIABLE & KINKED REGIME COMPONENTS (y_t, y_t^+, y_t^-)
# -------------------------------------------------------------------------
# Demean us_tavg_anom by calendar month over the pre-1980 baseline window (if present in df)
# or over the full sample to purge any residual seasonal cycle
baseline_slice = yl.loc[yl.index.year <= 1980, "us_tavg_anom"]
if len(baseline_slice) >= 24:
    monthly_means = baseline_slice.groupby(baseline_slice.index.month).mean()
    yl_delta["y_tavg"] = yl["us_tavg_anom"] - yl.index.month.map(monthly_means)
else:
    yl_delta["y_tavg"] = (
        yl["us_tavg_anom"] - yl.groupby(yl.index.month)["us_tavg_anom"].transform("mean")
    )

# Define the physical warming threshold tau (0.0 corresponds to the 1901-1980 baseline mean)
tau_threshold = 0.0
y_centered = yl_delta["y_tavg"] - tau_threshold

# Construct censored/kinked regime components for CKSVAR state vector z_t^*
yl_delta["y_tavg_pos"] = np.maximum(y_centered, 0.0)
yl_delta["y_tavg_neg"] = np.minimum(y_centered, 0.0)

# -------------------------------------------------------------------------
# 2. NONSTATIONARY I(1) / I*(1) LEVEL VARIABLES (x_t)
# -------------------------------------------------------------------------
# Log atmospheric CO2 trend
yl_delta["ln_co2"] = np.log(yl["co2"])

# Palmer Hydrological Drought Index (kept in levels since it spans negative/positive values)
yl_delta["phdi_level"] = yl["phdi"]

# Log Headline and Manufacturing Industrial Production
yl_delta["ln_indpro"] = np.log(yl["indpro"])
yl_delta["ln_ipman"] = np.log(yl["ipmansics"])

# Log Manufacturing Labor Productivity: ln(Output / (Employment * Weekly Hours))
total_mfg_hours = yl["manemp"] * yl["awhman"]
yl_delta["ln_labor_prod"] = np.log(yl["ipmansics"] / total_mfg_hours)

# Log Consumer Price Index level and 10-Year Treasury Yield level
yl_delta["ln_cpi"] = np.log(yl["cpiaucsl"])
yl_delta["gs10_level"] = yl["gs10"]

# -------------------------------------------------------------------------
# 3. STATIONARY FIRST DIFFERENCES (\Delta z_t) FOR VECM LAGS & J_LZ SCREENING
# -------------------------------------------------------------------------
yl_delta["d_y_tavg"] = yl_delta["y_tavg"].diff()
yl_delta["d_y_tavg_pos"] = yl_delta["y_tavg_pos"].diff()
yl_delta["d_y_tavg_neg"] = yl_delta["y_tavg_neg"].diff()
yl_delta["d_ln_co2"] = yl_delta["ln_co2"].diff()
yl_delta["d_phdi"] = yl_delta["phdi_level"].diff()
yl_delta["d_ln_indpro"] = yl_delta["ln_indpro"].diff()
yl_delta["d_ln_ipman"] = yl_delta["ln_ipman"].diff()
yl_delta["d_ln_labor_prod"] = yl_delta["ln_labor_prod"].diff()
yl_delta["d_ln_cpi"] = yl_delta["ln_cpi"].diff()
yl_delta["d_gs10"] = yl_delta["gs10_level"].diff()

# Drop initial NaN row created by first differencing
yl_delta = yl_delta.dropna()

# Separate views for convenience in downstream modeling:
# z_levels: Standard I(1) levels z_t = [y_t, x_t]
z_cols = [
    "y_tavg", "ln_co2", "phdi_level", "ln_indpro",
    "ln_ipman", "ln_labor_prod", "ln_cpi", "gs10_level"
]
yl_z_levels = yl_delta[z_cols]

# z_star_levels: Kinked state-space vector z_t^* = [y_t^+, y_t^-, x_t] for CKSVAR cointegration
z_star_cols = [
    "y_tavg_pos", "y_tavg_neg", "ln_co2", "phdi_level",
    "ln_indpro", "ln_ipman", "ln_labor_prod", "ln_cpi", "gs10_level"
]
yl_z_star = yl_delta[z_star_cols]

# dz_diffs: Stationary increments \Delta z_t for Stage-1 J_LZ complexity screening and VECM short-run lags
dz_cols = [
    "d_y_tavg", "d_ln_co2", "d_phdi", "d_ln_indpro",
    "d_ln_ipman", "d_ln_labor_prod", "d_ln_cpi", "d_gs10"
]
yl_dz_diffs = yl_delta[dz_cols]

print("Transformed dataset shape:", yl_delta.shape)
print(yl_z_star.tail())