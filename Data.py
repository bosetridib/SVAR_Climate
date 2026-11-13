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
fred_macro_variables = {
    # =========================================================================
    # 1. INDUSTRIAL PRODUCTION & CAPACITY (Physical Output by Stage & Sector)
    # =========================================================================
    'INDPRO': (
        "Industrial Production: Total Index | "
        "Intuition: Broad headline measure of real physical output across all U.S. factories, mines, and electric/gas utilities (excludes services, retail, and construction). | "
        "Examples: Cars, semiconductors, refined gasoline, mined coal, residential electricity."
    ),
    'TCU': (
        "Capacity Utilization: Total Index | "
        "Intuition: Percentage of total sustainable maximum production capacity currently in use across manufacturing, mining, and utilities; high values signal supply bottlenecks and inflation pressure. | "
        "Examples: Assembly line run-rates at auto plants, refinery operating rates, steel mill utilization."
    ),
    'IPMANSICS': (
        "Industrial Production: Manufacturing (SIC) | "
        "Intuition: Physical output strictly from manufacturing plants (using the historical Standard Industrial Classification), stripping out mining and utilities from INDPRO. | "
        "Examples: Machinery, chemicals, processed food, motor vehicles, electronics, apparel."
    ),
    'IPFINAL': (
        "Industrial Production: Final Products (Market Group) | "
        "Intuition: Finished physical goods ready for final use by households or businesses, rather than inputs to be processed further (splits into IPCONGD + IPBUSEQ). | "
        "Examples: Passenger cars, laptops, commercial aircraft, household appliances, packaged groceries."
    ),
    'IPCONGD': (
        "Industrial Production: Consumer Goods | "
        "Intuition: Total physical output of goods destined for household consumption (combines durable and nondurable consumer goods). | "
        "Examples: Pickup trucks, refrigerators, prescription drugs, gasoline, breakfast cereal, clothing."
    ),
    'IPDCONGD': (
        "Industrial Production: Durable Consumer Goods | "
        "Intuition: Output of long-lasting household goods (lifespan of 3+ years); highly sensitive to interest rates and consumer credit cycles. | "
        "Examples: Automobiles, washing machines, riding lawnmowers, home furniture, TVs, carpeting."
    ),
    'IPBUSEQ': (
        "Industrial Production: Equipment: Business Equipment | "
        "Intuition: Physical output of capital goods bought by firms to produce other goods and services; a direct monthly proxy for business fixed investment (CapEx). | "
        "Examples: Commercial jets, factory robotics, servers, farm tractors, medical imaging scanners, delivery trucks."
    ),
    'IPMAT': (
        "Industrial Production: Materials | "
        "Intuition: Upstream industrial inputs, raw materials, and intermediate components used by factories to make final products. | "
        "Examples: Unfinished steel coils, industrial chemicals, microchips, paper pulp, crude oil, auto transmissions."
    ),
    'IPUTIL': (
        "Industrial Production: Utilities: Electric and Gas Utilities (NAICS = 2211,2) | "
        "Intuition: Physical volume of electricity and natural gas distributed to residential, commercial, and industrial users. | "
        "Examples: Kilowatt-hours of grid electricity, cubic feet of piped natural gas for home heating."
    ),
    'IPG2211S': (
        "Industrial Production: Utilities: Electric Power Generation, Transmission, and Distribution (NAICS = 2211) | "
        "Intuition: Pure electricity output sub-index of IPUTIL (excludes natural gas utilities); reflects both industrial power demand and weather/cooling loads. | "
        "Examples: Coal, natural gas, nuclear, hydro, wind, and solar power fed into the electrical grid."
    ),

    # =========================================================================
    # 2. PRICES, INFLATION & COMMODITIES (Supply Chain to Consumer Checkout)
    # =========================================================================
    'WTISPLC': (
        "Spot Crude Oil Price: West Texas Intermediate (WTI) | "
        "Intuition: Benchmark dollar price per barrel of U.S. light sweet crude oil delivered at Cushing, Oklahoma; the primary upstream energy shock variable. | "
        "Examples: Unrefined light sweet crude oil (feedstock for gasoline, diesel, and jet fuel)."
    ),
    'PPIENG': (
        "Producer Price Index by Commodity: Fuels and Related Products and Power | "
        "Intuition: Wholesale price index for energy inputs paid or received by domestic producers across the economy (not seasonally adjusted). | "
        "Examples: Wholesale gasoline, diesel fuel, crude petroleum, industrial electric power, utility natural gas, coal."
    ),
    'WPSID61': (
        "Producer Price Index by Commodity: Intermediate Demand: Processed Goods for Intermediate Demand | "
        "Intuition: B2B pipeline inflation—prices charged by manufacturers for partially finished goods sold to other businesses as production inputs. | "
        "Examples:Structural steel, plastic resins, lumber, industrial chemicals, diesel fuel for trucking, flour sold to bakeries."
    ),
    'WPSFD49207': (
        "Producer Price Index by Commodity: Final Demand: Finished Goods | "
        "Intuition: Factory-gate/wholesale prices of completed goods ready for sale to retail distributors or end-users (excludes services). | "
        "Examples: Wholesale price of new cars shipped to dealerships, packaged food sold to supermarkets, factory machinery."
    ),
    'WPSFD4111': (
        "Producer Price Index by Commodity: Final Demand: Finished Consumer Foods | "
        "Intuition: Wholesale/farm-to-retail gate prices for foods ready for household consumption, before supermarket markups. | "
        "Examples: Boxed beef, pasteurized milk, fresh fruits and vegetables, bakery products, roasted coffee."
    ),
    'CPIAUCSL': (
        "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average | "
        "Intuition: Headline retail inflation measuring out-of-pocket prices paid by urban households for a fixed basket of goods and services. | "
        "Examples: Apartment rent, owners' equivalent rent, groceries, pump gasoline, used cars, doctor visits, airline tickets."
    ),
    'CPILFESL': (
        "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy | "
        "Intuition: 'Core CPI'—strips out volatile food and energy prices from CPIAUCSL to reveal underlying, sticky retail inflation trends. | "
        "Examples: Housing shelter/rent, auto insurance, medical care, apparel, new/used vehicles, restaurant meals, tuition."
    ),
    'CUSR0000SA0L2': (
        "Consumer Price Index for All Urban Consumers: All Items Less Shelter | "
        "Intuition: Headline CPI excluding housing shelter costs; useful for tracking non-housing inflation since CPI rent measures lag market rents. | "
        "Examples: Groceries, gasoline, utilities, vehicles, clothing, medical services, streaming subscriptions."
    ),
    'PCEPI': (
        "Personal Consumption Expenditures: Chain-Type Price Index | "
        "Intuition: The Federal Reserve's preferred headline inflation gauge; unlike fixed-weight CPI, it accounts for consumer substitution and includes expenses paid on behalf of households. | "
        "Examples: Household retail goods/services plus employer-paid health insurance, Medicare/Medicaid payments, and financial services."
    ),
    'MICH': (
        "University of Michigan: Inflation Expectation | "
        "Intuition: Survey-based median expected price change (%) over the next 12 months reported by U.S. households; gauges whether consumer inflation expectations are anchored. | "
        "Examples: Household expectations for next year's grocery, gas, and general living costs."
    ),

    # =========================================================================
    # 3. LABOR MARKET, HOURS & WAGES (Household & Establishment Surveys)
    # =========================================================================
    'PAYEMS': (
        "All Employees, Total Nonfarm | "
        "Intuition: Headline 'Nonfarm Payrolls' from the establishment survey; counts total paid U.S. workers excluding farm, private household, and nonprofit/self-employed workers. | "
        "Examples: Corporate office workers, factory staff, retail cashiers, nurses, teachers, construction crews."
    ),
    'CE16OV': (
        "Employment Level | "
        "Intuition: Total number of employed civilians (age 16+) from the household survey; broader than PAYEMS because it includes farm workers, the self-employed, and gig workers. | "
        "Examples: Salaried employees plus freelance contractors, Uber drivers, small business owners, and agricultural workers."
    ),
    'UNRATE': (
        "Unemployment Rate | "
        "Intuition: Headline U-3 unemployment rate—percentage of the civilian labor force (age 16+) who are jobless and actively looked for work in the past 4 weeks. | "
        "Examples: Laid-off workers, recent graduates, and re-entrants actively submitting job applications."
    ),
    'MANEMP': (
        "All Employees, Manufacturing | "
        "Intuition: Total payroll employment across all U.S. durable and nondurable manufacturing plants; highly cyclical indicator of industrial labor demand. | "
        "Examples: Auto assembly workers, machinists, chemical plant operators, food processing workers."
    ),
    'DMANEMP': (
        "All Employees, Durable Goods | "
        "Intuition: Payroll employment in factories producing long-lasting goods (subset of MANEMP); the most cyclically sensitive segment of manufacturing jobs. | "
        "Examples: Workers manufacturing aerospace parts, motor vehicles, primary metals, industrial machinery, and furniture."
    ),
    'USCONS': (
        "All Employees, Construction | "
        "Intuition: Total payroll employment in residential, commercial, and heavy civil construction; tightly linked to housing starts and interest rates. | "
        "Examples: Homebuilders, commercial electricians, plumbers, highway paving crews, roofers."
    ),
    'AWHMAN': (
        "Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing | "
        "Intuition: Leading business-cycle indicator—factories adjust existing workers' weekly hours and overtime before hiring or laying off staff (MANEMP). | "
        "Examples: Weekly shift hours and overtime clocked by floor machinists and assembly line workers."
    ),
    'AHETPI': (
        "Average Hourly Earnings of Production and Nonsupervisory Employees, Total Private | "
        "Intuition: Wage inflation gauge tracking average hourly pay for non-management private-sector workers (~80% of private payrolls). | "
        "Examples: Hourly wages of retail associates, warehouse workers, nurses, construction tradespeople, and factory staff."
    ),

    # =========================================================================
    # 4. PERSONAL INCOME & REAL CONSUMPTION (Household Demand)
    # =========================================================================
    'RPI': (
        "Real Personal Income | "
        "Intuition: Total inflation-adjusted income received by all U.S. individuals from all sources before taxes. | "
        "Examples: Wages and salaries, proprietor income, rental income, dividends, interest, plus government transfer payments (Social Security, unemployment)."
    ),
    'W875RX1': (
        "Real Personal Income Excluding Current Transfer Receipts | "
        "Intuition: Core organic household income—strips government transfers out of RPI to measure inflation-adjusted income earned strictly from market economic activity (key NBER recession indicator). | "
        "Examples: Paychecks, employer benefits, small business profits, dividends, and interest (excludes Social Security, stimulus checks, and Medicare)."
    ),
    'DPCERA3M086SBEA': (
        "Real Personal Consumption Expenditures (Quantity Index) | "
        "Intuition: Monthly chain-type quantity index measuring the real physical volume of goods and services bought by U.S. households (~68% of U.S. GDP). | "
        "Examples: Inflation-adjusted household spending on cars, groceries, rent, healthcare, dining out, and travel."
    ),

    # =========================================================================
    # 5. INTEREST RATES, YIELDS & CREDIT SPREADS (Monetary Policy & Financial Conditions)
    # =========================================================================
    'FEDFUNDS': (
        "Federal Funds Effective Rate | "
        "Intuition: The central monetary policy rate—volume-weighted median overnight interest rate at which depository institutions lend reserve balances to each other. | "
        "Examples: Overnight interbank reserve loans that anchor prime rates, SOFR, and short-term borrowing costs."
    ),
    'GS1': (
        "Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity | "
        "Intuition: Risk-free short-term government bond yield; heavily reflects bond market expectations of the Fed's policy path over the next 12 months. | "
        "Examples: Secondary-market yield on 1-year U.S. Treasury bills/notes."
    ),
    'GS10': (
        "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity | "
        "Intuition: Benchmark long-term risk-free interest rate; reflects long-run growth and inflation expectations plus term premium, directly driving mortgage and corporate borrowing rates. | "
        "Examples: 10-Year U.S. Treasury note yield (benchmark for 30-year fixed mortgages and corporate bonds)."
    ),
    'AAA': (
        "Moody's Seasoned Aaa Corporate Bond Yield | "
        "Intuition: Average yield on long-term corporate bonds issued by the highest-credit-quality ('prime' investment-grade) U.S. corporations with minimal default risk. | "
        "Examples: Long-term bonds (20+ year maturity) of top-rated blue-chip firms like Microsoft or Johnson & Johnson."
    ),
    'BAA': (
        "Moody's Seasoned Baa Corporate Bond Yield | "
        "Intuition: Average yield on long-term corporate bonds at the lowest tier of investment grade (just above junk/high-yield); sensitive to corporate default risk. | "
        "Examples: Long-term bonds of medium-grade industrial, utility, and BBB/Baa-rated corporations."
    ),
    'BAA10YM': (
        "Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity | "
        "Intuition: Classic corporate credit spread (BAA minus GS10); measures the extra yield investors demand for bearing corporate default and liquidity risk—spikes during financial stress and recessions. | "
        "Examples: Yield spread between lower-tier investment-grade corporate debt (BAA) and risk-free 10-Year Treasuries (GS10)."
    ),
}

# 1. Fetch Seasonally Adjusted Monthly U.S. Macroeconomic Series from FRED
macro = web.DataReader(list(fred_macro_variables), "fred", start=start_date, end=end_date)
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
# macro_df["a_t"] = macro["INDPRO"] / macro["PAYEMS"]
macro_df["labor_productivity"] = macro_df["IPMANSICS"] / (macro_df["MANEMP"] * macro_df["AWHMAN"])
macro_df["credit_spread"] = macro_df["BAA"] - macro_df["AAA"]
#macro_df["p_t"] = np.log(macro["CPIAUCSL"])
#macro_df["p_core_t"] = np.log(macro["CPILFESL"])
#macro_df["p_e_t"] = np.log(macro["PPIENG"])
#macro_df["oil_t"] = np.log(macro["WTISPLC"])
#macro_df["i_t"] = macro["FEDFUNDS"]

macro_df.columns = [_.lower() for _ in macro_df.columns]

# ==============================================================================
# 2. FETCH CLIMATE SERIES (IDEA 1 & IDEA 2)
# ==============================================================================
# 2A. CONUS Mean (tavg) & Max (tmax) Temperature Anomalies, PDSI, CDD, HDD (nClimGrid)
ncei_base = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/"
ncei_params = "/1/0/1900-2026/data.csv"

pmdi_raw = pd.read_csv(f"{ncei_base}pmdi{ncei_params}", comment="#")
zndx_raw = pd.read_csv(f"{ncei_base}zndx{ncei_params}", comment="#")
phdi_raw = pd.read_csv(f"{ncei_base}phdi{ncei_params}", comment="#")

# 1. Obtain NOAA GML Monthly Deseasonalized Atmospheric CO2 (Mauna Loa, 1958-2026)
co2 = pd.read_csv("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", comment="#")
co2 = pd.DataFrame(co2["deseasonalized"].values, index=pd.to_datetime(co2[["year", "month"]].assign(day=1)))
co2.index = co2.index.to_period("M").to_timestamp()
co2.columns = ["co2"]

us_tavg = pd.read_csv("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/1/0/1900-2026/data.csv?base_prd=true&begbaseyear=1901&endbaseyear=1980", comment="#")
us_tavg["date"] = pd.to_datetime(us_tavg["Date"].astype(str), format="%Y%m")
us_tavg = us_tavg.set_index("date")[["Departure from Average"]].rename(columns={"Departure from Average": "US_TAVG_ANOM"})

us_tmax = pd.read_csv("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tmax/1/0/1900-2026/data.csv?base_prd=true&begbaseyear=1901&endbaseyear=1980", comment="#")
us_tmax["date"] = pd.to_datetime(us_tmax["Date"].astype(str), format="%Y%m")
us_tmax = us_tmax.set_index("date")[["Departure from Average"]].rename(columns={"Departure from Average": "US_TMAX_ANOM"})

climate_df = pd.DataFrame({
    "PMDI":        pmdi_raw.set_index("Date")["Value"],
    "PHDI":        phdi_raw.set_index("Date")["Value"],
    "ZNDX":        zndx_raw.set_index("Date")["Value"]
})
climate_df.index = pd.to_datetime(climate_df.index.astype(str), format="%Y%m").to_period("M").to_timestamp()
climate_df = pd.concat([climate_df, us_tavg, us_tmax, co2], axis=1)

climate_df.columns = [_.lower() for _ in climate_df.columns]

#.to_period("M").to_timestamp()
#df_tavg_anomaly.plot(subplots=True); plt.show()