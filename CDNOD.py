import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import svar_dataset, climate_dataset
from sklearn.preprocessing import StandardScaler, RobustScaler

import pandas as pd
from pandas_datareader.fred import FredReader

# -------------------------------------------------------------------------
# 1. Define Observation Window and Monthly FRED Series Identifiers
# -------------------------------------------------------------------------
start_date = "1902-01-01"
end_date = "2026-10-01"

# Avenue 1: Electric Power Generation, Utilities Output, and Energy PPIs
fred_avenue_1 = [
    "IPG2211S",           # Industrial Production: Utilities: Electric Power (SA)
    "IPG22111N",          # Industrial Production: Electric Power Generation (NSA)
    "PCU221110221110",    # PPI: Electric Power Generation
    "PCU2211122111",      # PPI: Electric Power Generation (Historical/Industry)
    "PCU22112222112241",  # PPI: Residential Electric Power Distribution
    "PCU22112222112243",  # PPI: Industrial Electric Power Distribution
    "PCU221122221122413", # PPI: Residential Electric Power (East North Central)
    "M0128AUSM247NNBR"    # NBER Historical Electric Power Production
]

# Avenue 2: Agricultural Commodities, Dairy/Livestock Prices, and Core PPI/CPI
fred_avenue_2 = [
    "PWHEAMTUSDM",        # Global Price of Wheat (USD per Metric Ton)
    "PCOTTINDUSDM",       # Global Price of Cotton (US Cents per Pound)
    "APU0000703112",      # Average Price: Ground Chuck, 100% Beef (per lb)
    "PCU31151131151153",  # PPI: Fluid Milk (Temperature and Flavored)
    "PCU3115143115145",   # PPI: Dry, Condensed, and Evaporated Dairy Products
    "PCU311511311511P",   # PPI: Fluid Milk Manufacturing Primary Products
    "PCU311511311511A",   # PPI: Fluid Milk Manufacturing Other Receipts
    "WPUFD49103",         # PPI: Final Demand Less Foods and Energy
    "PPIACO",             # PPI: All Commodities
    "CPIAUCSL"            # CPI for All Urban Consumers: All Items (SA)
]

# Avenue 3: Labor Market, Consumption, Retail, Population, and Business Climate
fred_avenue_3 = [
    "UNRATE",             # Civilian Unemployment Rate
    "PCE",                # Personal Consumption Expenditures
    "MRTSSM44112USN",     # Retail Sales: Used Car Dealers
    "POPTHM",             # Population (Monthly, Thousands)
    "POP",                # Total Population: All Ages including Armed Forces Overseas
    "BCCINA156MNFRBNY",   # Current Business Climate: Better than Normal (FRB NY)
    "BCCNNA156MNFRBNY",   # Current Business Climate: Normal (FRB NY)
    "BCFDNA156MNFRBNY",   # Future Business Climate: Expecting to Worsen (FRB NY)
    "BCFINA156MNFRBNY",   # Future Business Climate: Expecting to Improve (FRB NY)
    "BCFNNA156MNFRBNY",   # Future Business Climate: Expecting Not to Change (FRB NY)
    "NASDAQNQUSM50101025" # Nasdaq Mid Cap Building: Climate Control Index
]

# Avenue 4: Monetary Policy, Sovereign Yields, Inflation Expectations, Saving
fred_avenue_4 = [
    "FEDFUNDS",           # Effective Federal Funds Rate
    "INTDSRUSM193N",      # Interest Rates, Discount Rate for United States
    "WALCL",              # Federal Reserve Total Assets (Eliminations from Consolidation)
    "T10YIEM",            # 10-Year Breakeven Inflation Rate
    "DFII10",             # 10-Year Treasury Inflation-Indexed Security, Constant Maturity
    "DGS20",              # 20-Year Treasury Constant Maturity Rate
    "PSAVE",              # Personal Saving
    "USREC"               # NBER based Recession Indicators for the United States
]

all_fred_series = fred_avenue_1 + fred_avenue_2 + fred_avenue_3 + fred_avenue_4

# -------------------------------------------------------------------------
# 2. Retrieve Macroeconomic & Financial Series via pandas-datareader
# -------------------------------------------------------------------------
fred_reader = FredReader(symbols=all_fred_series, start=start_date, end=end_date)
df_fred_raw = fred_reader.read()
fred_reader.close()

# Resample any higher-frequency financial/balance-sheet series to Monthly Start (MS) means
df_fred_monthly = df_fred_raw.resample("MS").mean()
df_fred_monthly.index.name = "Date"

# -------------------------------------------------------------------------
# 3. Retrieve Monthly U.S. Physical Climate Series from NOAA NCEI
#    (Contiguous U.S. Location Code 110, All Months 1895-2026)
# -------------------------------------------------------------------------
noaa_base_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110"

noaa_parameters = [
    "tavg",  # Contiguous U.S. Average Temperature (Fahrenheit)
    "tmax",  # Contiguous U.S. Maximum Temperature (Fahrenheit)
    "tmin",  # Contiguous U.S. Minimum Temperature (Fahrenheit)
    "pcp",   # Contiguous U.S. Precipitation (Inches)
    "cdd",   # Contiguous U.S. Cooling Degree Days (Base 65F)
    "hdd",   # Contiguous U.S. Heating Degree Days (Base 65F)
    "pdsi",  # Palmer Drought Severity Index
    "phdi",  # Palmer Hydrological Drought Index
    "pmdi",  # Palmer Modified Drought Index
    "zndx"   # Palmer Z-Index
]

df_climate_list = []

for param in noaa_parameters:
    # NOAA NCEI Climate at a Glance National Time Series CSV endpoint for all months
    csv_url = f"{noaa_base_url}/{param}/all/1/1895-2026.csv"
    df_temp = pd.read_csv(csv_url, comment="#")
    
    # Standardize column names (first column is YYYYMM Date, second column is Value)
    date_col = df_temp.columns[0]
    val_col = df_temp.columns[1]
    
    df_temp = df_temp[[date_col, val_col]].copy()
    df_temp.columns = ["YYYYMM", f"NOAA_CONUS_{param.upper()}"]
    
    # Convert YYYYMM integer/string to Monthly Start Timestamp
    df_temp["Date"] = pd.to_datetime(df_temp["YYYYMM"].astype(str), format="%Y%m")
    df_temp = df_temp.set_index("Date").drop(columns=["YYYYMM"])
    
    # Coerce NOAA missing-value sentinel (-99.99 or -9999) to NaN
    series_numeric = pd.to_numeric(df_temp[f"NOAA_CONUS_{param.upper()}"], errors="coerce")
    df_temp[f"NOAA_CONUS_{param.upper()}"] = series_numeric.mask(series_numeric <= -99.0)
    
    df_climate_list.append(df_temp)

# Retrieve Global Land and Ocean Temperature Departure as External Climate Instrument
global_temp_url = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "global/time-series/globe/land_ocean/all/1/1850-2026.csv"
)
df_global_temp = pd.read_csv(global_temp_url, comment="#")
g_date_col = df_global_temp.columns[0]
g_val_col = df_global_temp.columns[1]
df_global_temp = df_global_temp[[g_date_col, g_val_col]].copy()
df_global_temp.columns = ["YYYYMM", "NOAA_GLOBAL_TEMP_ANOMALY_C"]
df_global_temp["Date"] = pd.to_datetime(df_global_temp["YYYYMM"].astype(str), format="%Y%m")
df_global_temp = df_global_temp.set_index("Date").drop(columns=["YYYYMM"])
df_global_temp["NOAA_GLOBAL_TEMP_ANOMALY_C"] = pd.to_numeric(
    df_global_temp["NOAA_GLOBAL_TEMP_ANOMALY_C"], errors="coerce"
).mask(pd.to_numeric(df_global_temp["NOAA_GLOBAL_TEMP_ANOMALY_C"], errors="coerce") <= -99.0)

df_climate_list.append(df_global_temp)

# -------------------------------------------------------------------------
# 4. Merge All Climate and Macroeconomic Series into a Unified Monthly Panel
# -------------------------------------------------------------------------
df_climate_monthly = pd.concat(df_climate_list, axis=1)
df_climate_monthly = df_climate_monthly.loc[start_date:end_date]

df_climate_macro_panel = pd.concat([df_climate_monthly, df_fred_monthly], axis=1)
df_climate_macro_panel = df_climate_macro_panel.sort_index()

# Construct normalized Energy Degree Days (EDD = CDD + HDD) as a reference series
df_climate_macro_panel["NOAA_CONUS_EDD"] = (
    df_climate_macro_panel["NOAA_CONUS_CDD"] + df_climate_macro_panel["NOAA_CONUS_HDD"]
)

# Export unified panel for structural causal SVAR estimation
df_climate_macro_panel.to_csv("us_monthly_climate_macro_svar_panel.csv")
print(df_climate_macro_panel.info())