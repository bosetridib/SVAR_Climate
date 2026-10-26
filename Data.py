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
    "T_mean_anom": tavg_raw.set_index("Date")["Anomaly"],
    "T_max_anom":  tmax_raw.set_index("Date")["Anomaly"],
    "PDSI":        pdsi_raw.set_index("Date")["Value"],
    "CDD":         cdd_raw.set_index("Date")["Value"],
    "HDD":         hdd_raw.set_index("Date")["Value"],
})
climate_df.index = pd.to_datetime(climate_df.index.astype(str), format="%Y%m")
climate_df["T_cum"] = climate_df["T_mean_anom"].cumsum()

# 2B. NOAA Residential Energy Demand Temperature Index (REDTI)
redti_url = "https://www.ncei.noaa.gov/access/monitoring/redti/time-series/all/1960-2025.csv"
redti_raw = pd.read_csv(redti_url, comment="#")
redti_raw["Date"] = pd.to_datetime(redti_raw["Date"].astype(str), format="%Y%m")
climate_df["REDTI"] = redti_raw.set_index("Date")["REDTI"]

# 2C. NOAA CPC Oceanic Niño Index (ONI) for Score Intervention (e_t)
oni_url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
oni_raw = pd.read_fwf(oni_url)
season_to_month = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
}
oni_raw["Month"] = oni_raw["SEAS"].map(season_to_month)
oni_raw["Date"] = pd.to_datetime(dict(year=oni_raw["YR"], month=oni_raw["Month"], day=1))
climate_df["ONI"] = oni_raw.set_index("Date")["ANOM"]

# 2D. NOAA U.S. Climate Extremes Index (USCEI Steps 1-5) for Idea 2 Tail Shares
# Step 1: Max Temp Upper Tail, Step 2: Min Temp Upper Tail, Step 3: PDSI Tails,
# Step 4: Extreme 1-Day Precip, Step 5: Days With/Without Precip
cei_base = "https://www.ncei.noaa.gov/access/monitoring/cei/time-series/US/monthly/"
cei_step1 = pd.read_csv(f"{cei_base}step1.csv", comment="#").set_index("Date")
cei_step2 = pd.read_csv(f"{cei_base}step2.csv", comment="#").set_index("Date")
cei_step3 = pd.read_csv(f"{cei_base}step3.csv", comment="#").set_index("Date")
cei_step4 = pd.read_csv(f"{cei_base}step4.csv", comment="#").set_index("Date")

for df_step, name in zip([cei_step1, cei_step2, cei_step3, cei_step4],
                         ["CEI_heat_max", "CEI_heat_min", "CEI_drought", "CEI_flood"]):
    df_step.index = pd.to_datetime(df_step.index.astype(str), format="%Y%m")
    climate_df[name] = df_step.iloc[:, -1] / 100.0  # Convert percentage area to [0, 1] share

# ==============================================================================
# 3. FETCH FEDERAL RESERVE EXCESS BOND PREMIUM & SHOCK PROXIES (IDEA 2)
# ==============================================================================
# Gilchrist-Zakrajsek Excess Bond Premium (Federal Reserve Board public CSV)
ebp_url = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
ebp_raw = pd.read_csv(ebp_url)
ebp_raw["date"] = pd.to_datetime(ebp_raw["date"]).dt.to_period("M").dt.to_timestamp()
ebp_series = ebp_raw.set_index("date")["ebp"]

# External Shock Proxies Z_t (Gertler-Karadi FOMC, Känzig Oil, Ramey Fiscal)
# Load from local replication CSV or public replication repository URL
# Format expected: columns ['date', 'gk_mps', 'kanzig_oil', 'ramey_news']
proxies_url = "https://raw.githubusercontent.com/macro-replication/shock-proxies/main/us_monthly_proxies.csv"
# If offline or using local replication files, replace proxies_url with local path: "us_monthly_proxies.csv"

# ==============================================================================
# 4. ASSEMBLE FINAL DATASETS FOR IDEA 1 AND IDEA 2
# ==============================================================================
# Idea 1 Dataset: 1960M1 - 2025M12 (Raw Nonstationary Levels + Climate + ONI)
df_idea1 = pd.concat([
    macro_df[["y_t", "rpi_t", "a_t", "p_t", "p_core_t", "p_e_t", "oil_t", "i_t"]],
    climate_df[["T_cum", "T_mean_anom", "T_max_anom", "REDTI", "CDD", "HDD", "ONI"]]
], axis=1).loc["1960-01-01":"2025-12-01"]

# Idea 2 Dataset: 1970M1 - 2025M12 (Macro Vector + EBP + USCEI Tail Shares)
df_idea2 = pd.concat([
    macro_df[["INDPRO", "UNRATE", "CPIAUCSL", "PCEPI", "WPSID61", "GS1"]],
    ebp_series.rename("EBP"),
    climate_df[["CEI_heat_max", "CEI_heat_min", "CEI_drought", "CEI_flood", "PDSI", "REDTI"]]
], axis=1).loc["1970-01-01":"2025-12-01"]

df_idea1.to_csv("idea1_neural_cointegration_svar_data.csv")
df_idea2.to_csv("idea2_tail_risk_riesz_svar_data.csv")
print("Idea 1 shape:", df_idea1.shape, "| Idea 2 shape:", df_idea2.shape)


# Load the monthly sheet (skipping default headers to map the structure manually)
df_raw = pd.read_excel("ACI.xlsx", sheet_name='ACI Combined Monthly', header=None).iloc[6:,1:].T
date_index = pd.to_datetime({'year': df_raw.iloc[1:,0], 'month': df_raw.iloc[1:,1], 'day': 1})
ACI = pd.DataFrame(df_raw.iloc[1:, 17].values, index=date_index, columns=['ACI'])
ACI.index.name = 'Date'
