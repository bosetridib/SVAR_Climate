import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FETCH ALL MACROECONOMIC SERIES FROM FRED VIA WEB.DATAREADER
# ==============================================================================
start_date = datetime.datetime(1950, 1, 1)
end_date = datetime.datetime(2026, 10, 31)

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
# macro.apply(pd.Series.first_valid_index)
# macro.notna().idxmin()

# 4. Obtain Kilian (2009) Monthly Global Real Economic Activity Index (1968-2026)
greai = pd.read_csv("https://kvasilopoulos.github.io/econdata/data/k2009.csv").set_index("date")
greai.index = pd.to_datetime(greai.index).to_period("M").to_timestamp()
greai.name = "greai"
greai = greai.loc[start_date:end_date]

# 5. Obtain Baker, Bloom, & Davis (2016) Monthly U.S. Economic Policy Uncertainty Index (1900-2026)
epu = pd.read_csv("https://kvasilopoulos.github.io/econdata/data/bbd2016.csv").set_index("date").iloc[:,0]
epu.name = "epu"
epu.index = pd.to_datetime(epu.index).to_period("M").to_timestamp()
epu = epu.loc[start_date:end_date]

# 2. Obtain Gilchrist & Zakrajsek (2012) Monthly Excess Bond Premium & Credit Spread (1973-2026)
gz_url = "https://kvasilopoulos.github.io/econdata/data/gz2012.csv"
gz = pd.read_csv(gz_url).set_index('date')
gz.index = pd.to_datetime(gz.index).to_period("M").to_timestamp()
gz = gz.loc[start_date:end_date]

# 3. Obtain Jurado, Ludvigson, & Ng (2015) Monthly Macro, Financial, and Real Uncertainty (1960-2026)
jln_url = "https://kvasilopoulos.github.io/econdata/data/jln2015.csv"
jln = pd.read_csv(jln_url).set_index('date')
jln.index = pd.to_datetime(jln.index).to_period("M").to_timestamp()
jln = jln.loc[start_date:end_date]

macro_df = pd.concat([macro, greai, epu, gz[['gz_spread','ebp']], jln], axis=1)

# Construct log-level transformations for Idea 1 (Nonstationary Levels)
#macro_df["ip"] = np.log(macro["INDPRO"])
#macro_df["rpi_t"] = np.log(macro["W875RX1"])
macro_df["a_t"] = macro["INDPRO"] / macro["PAYEMS"]
#macro_df["p_t"] = np.log(macro["CPIAUCSL"])
#macro_df["p_core_t"] = np.log(macro["CPILFESL"])
#macro_df["p_e_t"] = np.log(macro["PPIENG"])
#macro_df["oil_t"] = np.log(macro["WTISPLC"])
#macro_df["i_t"] = macro["FEDFUNDS"]



# ==============================================================================
# 2. FETCH CLIMATE SERIES (IDEA 1 & IDEA 2)
# ==============================================================================
# 2A. CONUS Mean (tavg) & Max (tmax) Temperature Anomalies, PDSI, CDD, HDD (nClimGrid)
ncei_base = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/"
ncei_params = "/1/0/1900-2026/data.csv"

pdsi_raw = pd.read_csv(f"{ncei_base}pmdi{ncei_params}", comment="#")
zndx_raw = pd.read_csv(f"{ncei_base}zndx{ncei_params}", comment="#")

# 1. Obtain NOAA GML Monthly Deseasonalized Atmospheric CO2 (Mauna Loa, 1958-2026)
co2 = pd.read_csv("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", comment="#")
co2 = pd.DataFrame(co2["deseasonalized"].values, index=pd.to_datetime(co2[["year", "month"]].assign(day=1)))
co2.index = co2.index.to_period("M").to_timestamp()
co2.columns = ["co2"]

us_tavg = pd.read_csv("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/1/0/1900-2026/data.csv?base_prd=true&begbaseyear=1901&endbaseyear=1980", comment="#")
us_tavg["date"] = pd.to_datetime(us_tavg["Date"].astype(str), format="%Y%m")
us_tavg = us_tavg.set_index("date")[["Departure from Average"]].rename(columns={"Departure from Average": "US_TAVG_ANOM"})

climate_df = pd.DataFrame({
    "PMDI":        pdsi_raw.set_index("Date")["Value"],
    "ZNDX":        zndx_raw.set_index("Date")["Value"]
})
climate_df.index = pd.to_datetime(climate_df.index.astype(str), format="%Y%m").to_period("M").to_timestamp()
climate_df = pd.concat([climate_df, us_tavg, co2], axis=1)

#.to_period("M").to_timestamp()
#df_tavg_anomaly.plot(subplots=True); plt.show()