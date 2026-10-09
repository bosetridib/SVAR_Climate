import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import svar_dataset, climate_dataset
from sklearn.preprocessing import StandardScaler, RobustScaler

import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.rpcmci import RPCMCI

# Tests
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmiknn_mixed import CMIknnMixed
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite import plotting as tp

# Data Preparation
econ = svar_dataset[['INDPRO', 'CFNAI', 'CPIAUCSL', 'PAYEMS', 'UNRATE', 'FEDFUNDS']]
econ[['INDPRO', 'CPIAUCSL', 'PAYEMS']] = np.log(econ[['INDPRO', 'CPIAUCSL', 'PAYEMS']]).diff()
econ = econ.dropna()
clim = climate_dataset[['US_Precipitation_Inches', 'US_Palmer_Drought_Index', 'ONI']].dropna()

clim = clim.loc[econ.index]

y = pd.concat([econ, clim], axis=1)

# y.plot(subplots=True); plt.show()

T, N = y.shape
raw_data = y.copy()
var_names = y.columns.tolist()

# CRITICAL: Standardize data (Z-score) before using non-linear distance-based metrics
scaler = StandardScaler()
# scaler = RobustScaler()
scaled_data = scaler.fit_transform(raw_data)

# Initialize Tigramite DataFrame
dataframe = pp.DataFrame(data=scaled_data, var_names=var_names, data_type=np.zeros(scaled_data.shape, dtype=int))

# Define the maximum lag to test (analogous to SVAR lag order 'p')
tau_max = 4
pc_alpha = 0.05 # Significance level for the causal links

test_ci = [
    GPDC(significance='analytic'),
    CMIknn(significance='shuffle_test', sig_samples=100, knn=50, shuffle_neighbors=8, transform='ranks'),
    #ParCorr(significance='analytic'),
    #ParCorrMult(significance='analytic'),
    RobustParCorr(significance='analytic'),
    RegressionCI(significance='analytic'),
    CMIknnMixed(significance='shuffle_test', sig_samples=100, knn=50, shuffle_neighbors=8, transform='ranks')
]

results = {}
for test in test_ci:
    print(test.__class__.__name__)
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=test,
        verbosity=2
    )
    results[test.__class__.__name__] = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.05)
# End of the loop

results_r = {}
for test in test_ci:
    print(test.__class__.__name__)
    pcmci = RPCMCI(
        dataframe=dataframe, 
        cond_ind_test=test,
        verbosity=2
    )
    results_r[test.__class__.__name__] = pcmci.run_rpcmci(tau_max=tau_max, num_regimes=2, max_transitions=10, pc_alpha=0.05)
# End of the loop

results_l = {}
for test in test_ci:
    print(test.__class__.__name__)
    pcmci = LPCMCI(
        dataframe=dataframe, 
        cond_ind_test=test,
        verbosity=2
    )
    results_l[test.__class__.__name__] = pcmci.run_lpcmci(tau_max=tau_max, pc_alpha=0.05)
# End of the loop

# ---------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------
# Plotting the GPDC Time Series Graph
for _ in results.keys():
    tp.plot_graph(
        val_matrix=results[_]['val_matrix'],
        graph=results[_]['graph'],
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI'
    )
    plt.show()
#

import pickle
# Save the results to a file
with open('pcmci_results.pkl', 'wb') as f:
    pickle.dump([results, results_r], f)

import pickle
# Load the results from a file
with open('pcmci_results.pkl', 'rb') as f:
    results, results_r = pickle.load(f)
#####