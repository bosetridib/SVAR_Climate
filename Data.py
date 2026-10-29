import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FETCH ALL MACROECONOMIC SERIES FROM FRED VIA WEB.DATAREADER
# ==============================================================================
start_date = datetime.datetime(1950, 1, 1)
end_date = datetime.datetime(2025, 9, 30)

# Combined list of FRED tickers for Idea 1 and Idea 2
# Note: W875RX1 is FRED's ticker for Real Personal Income Less Transfers (RPI);
# WTISPLC is Spot Crude Oil WTI (OILPRICE).
fred_tickers = [
    "INDPRO",    # Industrial Production Index (Idea 1 & 2)
    "TCU",
    "IPG2211S",
    "WPSFD4111",
    "MICH",
    "AHETPI",
    "GS10",
    "BAA10YM",
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
    "WPSFD49207",
    "RPI",
    "CE16OV"
]

# 1. Fetch Seasonally Adjusted Monthly U.S. Macroeconomic Series from FRED
macro = web.DataReader(fred_tickers, "fred", start=start_date, end=end_date)
macro.index = macro.index.to_period("M").to_timestamp()

macro.dropna(axis=1).shape[0]
macro.plot(subplots=True); plt.show()

macro_df = pd.DataFrame()
# Construct log-level transformations for Idea 1 (Nonstationary Levels)
macro_df["y_t"] = np.log(macro["INDPRO"])
macro_df["rpi_t"] = np.log(macro["W875RX1"])
macro_df["a_t"] = np.log(macro["INDPRO"] / macro["PAYEMS"])
macro_df["p_t"] = np.log(macro["CPIAUCSL"])
macro_df["p_core_t"] = np.log(macro["CPILFESL"])
macro_df["p_e_t"] = np.log(macro["PPIENG"])
macro_df["oil_t"] = np.log(macro["WTISPLC"])
macro_df["i_t"] = macro["FEDFUNDS"]





















# ==============================================================================
# 2. FETCH CLIMATE SERIES (IDEA 1 & IDEA 2)
# ==============================================================================
# 2A. CONUS Mean (tavg) & Max (tmax) Temperature Anomalies, PDSI, CDD, HDD (nClimGrid)
ncei_base = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/"
ncei_params = "/1/0/1960-2025.csv?base_prd=true&begbaseyear=1951&endbaseyear=1980"

tavg_raw = pd.read_csv(f"{ncei_base}tavg{ncei_params}", comment="#")
tmax_raw = pd.read_csv(f"{ncei_base}tmax{ncei_params}", comment="#")
pdsi_raw = pd.read_csv(f"{ncei_base}pdsi{ncei_params}", comment="#")

climate_df = pd.DataFrame({
    "T_mean_anom": tavg_raw.set_index("Date")["Departure from Average"],
    "T_max_anom":  tmax_raw.set_index("Date")["Departure from Average"],
    "PDSI":        pdsi_raw.set_index("Date")["Value"],
})
climate_df.index = pd.to_datetime(climate_df.index.astype(str), format="%Y%m")
climate_df["T_cum"] = climate_df["T_mean_anom"].cumsum()



# 2. Fetch Non-Seasonal Standardized Monthly U.S. Climate Indices from NOAA NCEI
# Palmer Drought Severity Index (PDSI) and Palmer Z-Index (ZNDX) for Contiguous U.S. (Region 110)
url_pdsi = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi/1/0/1960-2026.csv"
url_zndx = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/zndx/1/0/1960-2026.csv"

urllib.request.urlretrieve(url_pdsi, "noaa_us_pdsi.csv")
urllib.request.urlretrieve(url_zndx, "noaa_us_zndx.csv")

df_pdsi = pd.read_csv("noaa_us_pdsi.csv", comment="#")
df_pdsi["DATE"] = pd.to_datetime(df_pdsi["Date"].astype(str), format="%Y%m")
df_pdsi = df_pdsi.set_index("DATE")[["Value"]].rename(columns={"Value": "US_PDSI"})

df_zndx = pd.read_csv("noaa_us_zndx.csv", comment="#")
df_zndx["DATE"] = pd.to_datetime(df_zndx["Date"].astype(str), format="%Y%m")
df_zndx = df_zndx.set_index("DATE")[["Value"]].rename(columns={"Value": "US_ZNDX"})

# 3. Merge into a Single Aligned Monthly Short-Run Macro-Climate DataFrame


import urllib.request
import pandas as pd


# 2. Fetch NASA GISS Global Land-Ocean Monthly Temperature Anomaly (Deseasonalized)
url_giss = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
urllib.request.urlretrieve(url_giss, "nasa_giss_global_temp.csv")

df_giss_raw = pd.read_csv("nasa_giss_global_temp.csv", skiprows=1)
df_giss_months = df_giss_raw[["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
df_giss_melted = df_giss_months.melt(id_vars=["Year"], var_name="Month", value_name="GLOBAL_TEMP_ANOM")
df_giss_melted["GLOBAL_TEMP_ANOM"] = pd.to_numeric(df_giss_melted["GLOBAL_TEMP_ANOM"], errors="coerce")
df_giss_melted["DATE"] = pd.to_datetime(df_giss_melted["Year"].astype(str) + "-" + df_giss_melted["Month"], format="%Y-%b")
df_global_temp = df_giss_melted.set_index("DATE")[["GLOBAL_TEMP_ANOM"]].sort_index()

# 3. Fetch NOAA NCEI Contiguous U.S. Monthly Temperature Departure (Deseasonalized via 1901-2000 Base)
url_us_tavg = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/1/0/1965-2026.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
urllib.request.urlretrieve(url_us_tavg, "noaa_us_tavg_anom.csv")

df_us_tavg = pd.read_csv("noaa_us_tavg_anom.csv", comment="#")
df_us_tavg["Date"] = pd.to_datetime(df_us_tavg["Date"].astype(str), format="%Y%m")
df_us_tavg = df_us_tavg.set_index("Date")[["Departure from Average"]].rename(columns={"Departure from Average": "US_TAVG_ANOM"})

# 4. Fetch NOAA GML Mauna Loa Deseasonalized Monthly Atmospheric CO2 Trend
url_co2 = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
urllib.request.urlretrieve(url_co2, "noaa_mlo_co2.csv")

df_co2 = pd.read_csv("noaa_mlo_co2.csv", comment="#")
df_co2["DATE"] = pd.to_datetime(df_co2["year"].astype(str) + "-" + df_co2["month"].astype(str) + "-01")
df_co2_trend = df_co2.set_index("DATE")[["deseasonalized"]].rename(columns={"deseasonalized": "CO2_DESEASON_TREND"})

# long_run_data.plot(subplots=True); plt.show()