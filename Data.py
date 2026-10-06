import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURE TIMEFRAME
# ==========================================
start_date = datetime.datetime(1900, 1, 1)
end_date = datetime.datetime(2025, 1, 1)

# ==========================================
# 2. DEFINE FRED MACRO & CLIMATE TICKERS
# ==========================================
# PDR can fetch a list of FRED tickers in a single call and auto-merge them.
fred_tickers = [
    "INDPRO",               # Industrial Production
    "CFNAI",                # Chicago Fed National Activity Index
    "CPIAUCSL",             # Headline CPI
    "CORESTICKM159SFRBATL", # Sticky Price CPI (Atlanta Fed)
    "PAYEMS",               # Nonfarm Payrolls
    "UNRATE",               # Unemployment Rate
    "FEDFUNDS",             # Federal Funds Rate
    #"T10Y3M",               # 10Y-3M Term Spread, daily
    #"NFCI",                 # National Financial Conditions Index, weekly
    "WTISPLC",              # WTI Crude Oil Prices
    "MHHNGSP",              # Henry Hub Natural Gas
    "TTLCONS",              # Total Construction Spending
    "ANDENO",               # Core Capital Goods Orders
    #"JTSHIR",               # JOLTS Hires
    #"JTSQUR"               # JOLTS Quits
]

svar_dataset = web.DataReader(fred_tickers, "fred", start_date, end_date)
svar_dataset.index.name = "Date"
svar_dataset.apply(pd.Series.first_valid_index)
svar_dataset.describe()

# ==========================================
# 1. CONFIGURE TIMEFRAME
# ==========================================
start_year = 1900
end_year = 2025

# List to hold all our individual monthly series before joining
climate_series = []

# ==========================================
# 2. NOAA U.S. CLIMATE AT A GLANCE (MONTHLY)
# ==========================================
# NOAA Area 110 = Contiguous U.S. | Time Scale 1 = Monthly
noaa_base_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110"

# Dictionary of NOAA Parameter : Column Name
noaa_variables = {
    'tavg': 'US_Avg_Temp_F',
    'pcp':  'US_Precipitation_Inches',
    'pdsi': 'US_Palmer_Drought_Index',
    'cdd':  'US_Cooling_Degree_Days',
    'hdd':  'US_Heating_Degree_Days'
}

for param, col_name in noaa_variables.items():
    # Construct the stable CSV URL
    url = f"{noaa_base_url}/{param}/1/0/{start_year}-{end_year}/data.csv"
    
    # NOAA's CSVs have 4 lines of metadata at the top before the headers
    df = pd.read_csv(url, comment ='#')
    
    # NOAA formats monthly dates as YYYYMM (e.g., "199001" for Jan 1990)
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m')
    df.set_index('Date', inplace=True)
    
    # Keep only the value column and rename it
    df = df[['Value']].rename(columns={'Value': col_name})
    climate_series.append(df)

# ==========================================
# 3. CLIMATE POLICY UNCERTAINTY (CPU) INDEX
# ==========================================
cpu_url = "https://www.policyuncertainty.com/media/cpu_pu.xlsx"
cpu_df = pd.read_excel(cpu_url, sheet_name='data', usecols=['date', 'cpu_index_narrow'])
cpu_df.rename(columns={'date': 'Date', 'cpu_index_narrow': 'CPU_Index'}, inplace=True)

# Clean date column and set as index
cpu_df['Date'] = pd.to_datetime(cpu_df['Date'], format="%YM%m")
cpu_df.set_index('Date', inplace=True)

# Ensure the index aligns exactly to the 1st of every month
cpu_df.index = cpu_df.index.to_period('M').to_timestamp()
climate_series.append(cpu_df)

# ==========================================
# 4. OCEANIC NINO INDEX (ONI)
# ==========================================
oni_url = "https://psl.noaa.gov/data/correlation/oni.data"

# Read the raw text block. Skip header, ignore trailing descriptive text rows.
oni_raw = pd.read_csv(oni_url, sep=r"\s+", skiprows=1, header=None, on_bad_lines='skip')

# Filter out footer text: Keep only rows where the first column is a valid year
oni_raw = oni_raw[pd.to_numeric(oni_raw[0], errors='coerce').notnull()]
oni_raw = oni_raw.astype(float)
oni_raw = oni_raw[oni_raw[0] >= start_year]
oni_raw = oni_raw[oni_raw[0] <= end_year]

# The data is a matrix: Col 0 is Year, Cols 1-12 are Jan-Dec. We need to melt it.
oni_raw.rename(columns={0: 'Year'}, inplace=True)
oni_melted = oni_raw.melt(id_vars=['Year'], var_name='Month', value_name='ONI')

# Construct proper DateTime objects from the Year and Month columns
oni_melted['Date'] = pd.to_datetime(
    oni_melted['Year'].astype(int).astype(str) + '-' + 
    oni_melted['Month'].astype(int).astype(str) + '-01'
)
oni_melted.set_index('Date', inplace=True)

# Clean up missing values (NOAA typically uses -99.9 for missing data)
oni_melted = oni_melted[['ONI']].sort_index()
climate_series.append(oni_melted)

# ==========================================
# 5. MERGE INTO SINGLE SVAR MATRIX
# ==========================================
print("Merging all series into a single monthly DateTime matrix...")

# Concatenate all dataframes along the columns
climate_dataset = pd.concat(climate_series, axis=1)

# Trim to ensure we only have data within our exact requested window
climate_dataset = climate_dataset.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

print("\nData acquisition complete. Head of the dataset:")
print(climate_dataset.head())

# Export the final aligned time-series to CSV
climate_dataset.to_csv("us_monthly_climate_dataset.csv")
print("\nDataset successfully saved to 'us_monthly_climate_dataset.csv'")