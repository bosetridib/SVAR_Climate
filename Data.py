import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FETCH ALL MACROECONOMIC SERIES FROM FRED VIA WEB.DATAREADER
# ==============================================================================
start_date = datetime.datetime(1959, 1, 1)
end_date = datetime.datetime(2025, 9, 30)

# Combined list of FRED tickers for Idea 1 and Idea 2
# Note: W875RX1 is FRED's ticker for Real Personal Income Less Transfers (RPI);
# WTISPLC is Spot Crude Oil WTI (OILPRICE).
fred_tickers = [
    "INDPRO",    # Industrial Production Index (Idea 1 & 2)
    "W875RX1",   # Real Personal Income Less Transfers / RPI (Idea 1)
    "PAYEMS",    # Total Nonfarm Payrolls (Idea 1)
    "AWHMAN",    # Average Weekly Hours: Manufacturing (Idea 1)
    "CPIAUCSL",  # Headline CPI (Idea 1 & 2)
    "CPILFESL",  # Core CPI (Idea 1)
    "PPIENG",    # PPI: Fuels and Related Products and Power (Idea 1)
    "WTISPLC",   # Spot Crude Oil Price: WTI (Idea 1)
    "FEDFUNDS",  # Effective Federal Funds Rate (Idea 1)
    "UNRATE",    # Civilian Unemployment Rate (Idea 2)
    "PCEPI",     # Personal Consumption Expenditures Chain-Type Price Index (Idea 2)
    "WPSID61",   # PPI: Intermediate Energy Goods (Idea 2)
    "GS1",       # 1-Year Treasury Constant Maturity Rate (Idea 2)
]

macro_df = web.DataReader(fred_tickers, "fred", start_date, end_date)
macro_df.index = macro_df.index.to_period("M").to_timestamp()

# Construct log-level transformations for Idea 1 (Nonstationary Levels)
macro_df["y_t"] = np.log(macro_df["INDPRO"])
macro_df["rpi_t"] = np.log(macro_df["W875RX1"])
macro_df["a_t"] = np.log(macro_df["INDPRO"] / macro_df["PAYEMS"])
macro_df["p_t"] = np.log(macro_df["CPIAUCSL"])
macro_df["p_core_t"] = np.log(macro_df["CPILFESL"])
macro_df["p_e_t"] = np.log(macro_df["PPIENG"])
macro_df["oil_t"] = np.log(macro_df["WTISPLC"])
macro_df["i_t"] = macro_df["FEDFUNDS"]

# ==============================================================================
# 2. FETCH NOAA NCEI & CPC CLIMATE SERIES (IDEA 1 & IDEA 2)
# ==============================================================================
# 2A. CONUS Mean (tavg) & Max (tmax) Temperature Anomalies, PDSI, CDD, HDD (nClimGrid)
ncei_base = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/"
ncei_params = "/1/0/1960-2025.csv?base_prd=true&begbaseyear=1951&endbaseyear=1980"

tavg_raw = pd.read_csv(f"{ncei_base}tavg{ncei_params}", comment="#")
tmax_raw = pd.read_csv(f"{ncei_base}tmax{ncei_params}", comment="#")
pdsi_raw = pd.read_csv(f"{ncei_base}pdsi{ncei_params}", comment="#")
cdd_raw  = pd.read_csv(f"{ncei_base}cdd{ncei_params}", comment="#")
hdd_raw  = pd.read_csv(f"{ncei_base}hdd{ncei_params}", comment="#")

climate_df = pd.DataFrame({
    "T_mean_anom": tavg_raw.set_index("Date")["Departure from Average"],
    "T_max_anom":  tmax_raw.set_index("Date")["Departure from Average"],
    "PDSI":        pdsi_raw.set_index("Date")["Value"],
    "CDD":         cdd_raw.set_index("Date")["Value"],
    "HDD":         hdd_raw.set_index("Date")["Value"],
})
climate_df.index = pd.to_datetime(climate_df.index.astype(str), format="%Y%m")
climate_df["T_cum"] = climate_df["T_mean_anom"].cumsum()