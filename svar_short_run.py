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

import pickle
import zlib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Ensure clean alignment across all 11 series
ys_stage1 = ys_delta[macro_variables + climate_variables_stationary].copy()
N = len(ys_stage1)

# -------------------------------------------------------------------------
# STAGE 1A & 1B: Stationarity (ADF), LZ Complexity (float32 vs int8), and dim_H
# -------------------------------------------------------------------------
records = []

for col in ys_stage1.columns:
    x = ys_stage1[col].values.astype(np.float64)
    
    # 1. Stage 1A: Augmented Dickey-Fuller (ADF) Stationarity Test
    adf_x = adfuller(x, autolag="AIC", result_object=True)
    adf_stat = adf_x.statistic; adf_pval = adf_x.pvalue
    
    # 2. Method A: Hamzi et al. (2026) Baseline pickle + float32
    s_f32 = pickle.dumps(x.astype(np.float32))
    c_lz_f32 = len(zlib.compress(s_f32)) / len(s_f32)
    
    # 3. Method B: Naive Min-Max int8 (vulnerable to 2020 COVID-19 outliers)
    xmin, xmax = np.min(x), np.max(x)
    q_minmax = np.round((x - xmin) / (xmax - xmin + 1e-12) * 255.0 - 128.0).astype(np.int8)
    s_i8_mm = q_minmax.tobytes()
    c_lz_i8_minmax = len(zlib.compress(s_i8_mm)) / len(s_i8_mm)
    
    # 4. Method C (RECOMMENDED): Robust 1%-99% Winsorized int8 Quantization
    p01, p99 = np.percentile(x, [1.0, 99.0])
    x_rob = np.clip(x, p01, p99)
    q_rob = np.round((x_rob - p01) / (p99 - p01 + 1e-12) * 255.0 - 128.0).astype(np.int8)
    s_i8_rob = q_rob.tobytes()
    c_lz_i8_robust = len(zlib.compress(s_i8_rob)) / len(s_i8_rob)
    
    # 5. Method D: Normalized Spectral Entropy (Hamzi et al. Section 3.2 Second Proxy)
    x_centered = x - np.mean(x)
    psd = np.abs(np.fft.rfft(x_centered)) ** 2
    psd_prob = psd[1:] / (np.sum(psd[1:]) + 1e-12)
    psd_prob = psd_prob[psd_prob > 0]
    spec_entropy = -np.sum(psd_prob * np.log2(psd_prob)) / np.log2(len(psd_prob))
    
    # 6. Effective Hausdorff Dimension Proxy (dim_H = 2 - H via variogram scaling)
    var_lag1 = np.mean(np.diff(x) ** 2)
    var_lag2 = np.mean((x[2:] - x[:-2]) ** 2)
    hurst_est = np.clip(0.5 * np.log2(var_lag2 / (var_lag1 + 1e-12)), 0.01, 0.99)
    dim_hausdorff = 2.0 - hurst_est
    
    records.append({
        "Variable": col,
        "ADF_Stat": round(adf_stat, 3),
        "ADF_p": round(adf_pval, 4),
        "Stationary_I0": adf_pval < 0.05,
        "C_LZ_float32": round(c_lz_f32, 3),
        "C_LZ_int8_minmax": round(c_lz_i8_minmax, 3),
        "C_LZ_int8_robust": round(c_lz_i8_robust, 3),
        "Spectral_Entropy": round(spec_entropy, 3),
        "Hurst_H": round(hurst_est, 3),
        "Hausdorff_dim_H": round(dim_hausdorff, 3)
    })

df_diagnostics = pd.DataFrame(records).set_index("Variable")
print("=== STAGE 1A & 1B: UNIVARIATE STATIONARITY & COMPLEXITY DIAGNOSTICS ===")
print(df_diagnostics.to_string())

# -------------------------------------------------------------------------
# STAGE 1C: Bivariate Joint Complexity (J_LZ) & Spurious Correlation Screen
# -------------------------------------------------------------------------
theta_floor = 0.30   # Hamzi et al. (2026) complexity threshold
rho_cutoff = 0.30    # Correlation magnitude trigger for screening

c_vec = df_diagnostics["C_LZ_int8_robust"]
corr_matrix = ys_stage1.corr()

# Compute J_LZ = min(C_LZ(x), C_LZ(y)) and Geometric J_LZ = sqrt(C_LZ(x)*C_LZ(y))
j_lz_min = pd.DataFrame(
    np.minimum.outer(c_vec.values, c_vec.values),
    index=ys_stage1.columns, columns=ys_stage1.columns
)
j_lz_geom = pd.DataFrame(
    np.sqrt(np.outer(c_vec.values, c_vec.values)),
    index=ys_stage1.columns, columns=ys_stage1.columns
)

# Flag any pair with high correlation but low Joint Complexity (J_LZ < theta)
flagged_pairs = []
for i in range(len(ys_stage1.columns)):
    for j in range(i + 1, len(ys_stage1.columns)):
        v1, v2 = ys_stage1.columns[i], ys_stage1.columns[j]
        rho_val = corr_matrix.loc[v1, v2]
        jlz_val = j_lz_min.loc[v1, v2]
        jlz_g = j_lz_geom.loc[v1, v2]
        if abs(rho_val) >= rho_cutoff and jlz_val < theta_floor:
            flagged_pairs.append((v1, v2, round(rho_val, 3), round(jlz_val, 3), round(jlz_g, 3)))

print("\n=== STAGE 1C: PAIRWISE J_LZ (ROBUST INT8) MINIMUM MATRIX ===")
print(j_lz_min.round(3).to_string())
print(f"\nFlagged Spurious Low-Complexity Pairs (|rho| >= {rho_cutoff} & J_LZ < {theta_floor}):", flagged_pairs)


# ========================================================================
# ===========================stage 2======================================
# ========================================================================

# High-level CDANs class & visualization methods
from cdans import CDANs, plot_process_graph, plot_time_series_graph

# Low-level step functions exposed by the cdans library
from cdans.steps import (
    discover_lagged_adjacencies,  # Step 1 (§3.1): MCI lagged parent discovery
    build_partial_graph,          # Step 2 (§3.2): Partial graph with surrogate C
    refine_skeleton,              # Step 3 (§3.3): Lagged-parent conditioning & changing modules
    orient_edges                  # Step 4 (§3.4): Surrogate C -> X, v-structures, Meek's rules & independent change
)

# -------------------------------------------------------------------------
# 1. PREPARE STANDARDIZED DATA ARRAY & PHYSICAL CLIMATE SURROGATE (1D ARRAY)
# (Assumes df_short_run with your 11 variables is loaded in memory)
# -------------------------------------------------------------------------
var_names = macro_variables + climate_variables

df_clean = ys_delta[var_names].copy()
data_matrix = ((df_clean - df_clean.mean()) / df_clean.std()).values
n_samples, n_vars = data_matrix.shape

# Construct a 1D physical climate surrogate array C_t of shape (n_samples,)
# Reconstructs the level warming anomaly from d_us_tavg_anom + extreme heat anomaly
climate_surrogate_1d = np.cumsum(df_clean["us_tavg_anom"].values)
climate_surrogate_1d = (climate_surrogate_1d - climate_surrogate_1d.mean()) / climate_surrogate_1d.std()

# -------------------------------------------------------------------------
# 2. FIT HIGH-LEVEL CDANs MODEL
# Use ci_test="kci" for nonparametric Kernel CI (or "fisherz" for fast baseline)
# -------------------------------------------------------------------------
model_cdans = CDANs(
    tau_max=6,                        # 6 monthly lags to absorb credit_spread & CPI persistence
    alpha=0.05,                       # Significance level for CI tests
    pc_alpha=0.20,                    # Looser alpha threshold for initial PC step inside Step 1
    ci_test="kci",                    # Nonparametric Kernel Conditional Independence test
    surrogate=climate_surrogate_1d,   # Custom 1D climate surrogate array of shape (n_samples,)
    max_extra_conds=2,                # Max contemporaneous neighbors added to lagged conditioning sets
    use_independent_change=True,      # Orient remaining lag-0 edges via independent change principle
    independent_change_width="auto",  # Automatic kernel width for independent change direction inference
    verbose=True
)

result = model_cdans.fit(data_matrix)

# with open('result.pkl', 'wb') as f:
#     pickle.dump(result, f)
import pickle
with open('result.pkl', 'rb') as f:
    result = pickle.load(f)

# -------------------------------------------------------------------------
# 3. INSPECT RECOVERED STRUCTURAL GRAPH, CHANGING MODULES & B0 MATRIX
# -------------------------------------------------------------------------
print("\n=== CDANs FULL SUMMARY REPORT ===")
print(result.summary())

print("\n=== WALL-CLOCK TIMINGS PER STEP ===")
print(result.timings)

# Map numeric variable indices back to your 11 variable names
changing_vars = [var_names[idx] for idx in result.graph.changing_modules]
print("\n=== DETECTED NON-STATIONARY (CLIMATE-CHANGING) MODULES ===")
print(changing_vars)

print("\n=== RECOVERED LAGGED DIRECTED EDGES (X_i[t-lag] -> X_j[t]) ===")
for edge in result.graph.lagged_edges:
    print(edge)

# Convert contemporaneous adjacency matrix into a labeled DataFrame for Stage 3 SVAR
df_contemp_adj = pd.DataFrame(
    result.graph.contemp_adj,
    index=var_names,
    columns=var_names
)
print("\n=== RECOVERED CONTEMPORANEOUS ADJACENCY MATRIX (contemp_adj) ===")
print(df_contemp_adj.to_string())

# -------------------------------------------------------------------------
# 4. SAVE AGGREGATE PROCESS GRAPH & TIME-UNROLLED DAG VISUALIZATIONS
# -------------------------------------------------------------------------
plot_process_graph(result.graph, save_path="cdans_macro_climate_process_graph.png")
plot_time_series_graph(result.graph, save_path="cdans_macro_climate_unrolled_graph.png")