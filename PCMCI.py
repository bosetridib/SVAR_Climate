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
econ['FEDFUNDS'] = econ['FEDFUNDS'].diff()
econ = econ.dropna()
clim = climate_dataset[['US_Precipitation_Inches', 'US_Palmer_Drought_Index', 'ONI']].dropna()

clim = clim.loc[econ.index]

y = pd.concat([econ, clim], axis=1)
y = y.drop(columns=['INDPRO', 'PAYEMS', 'UNRATE', 'ONI'])
# y.plot(subplots=True); plt.show()

from statsmodels.tsa.stattools import adfuller
def check_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Augmented Dickey-Fuller (ADF) tests on all preprocessed variables."""
    results = []
    for col in df.columns:
        stat, pval, lags, nobs, _, _ = adfuller(df[col].dropna(), autolag="AIC")
        results.append({
            "Variable": col,
            "ADF_Stat": round(stat, 3),
            "p_value": round(pval, 4),
            "Lags": lags,
            "Stationary_5pct": pval < 0.05
        })
    return pd.DataFrame(results).set_index("Variable")
check_stationarity(y)

from statsmodels.tsa.statespace.tools import companion_matrix
from statsmodels.tsa.tsatools import lagmat
p = 4

z = lagmat(y, maxlag=p-1, trim='both', original='in', use_pandas=True)
z_t1 = lagmat(y, maxlag=p, trim='both', original='ex', use_pandas=True)

companion_matrix(y.values)


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
    #ParCorr(significance='analytic'),
    #ParCorrMult(significance='analytic'),
    #RobustParCorr(significance='analytic'),
    #RegressionCI(significance='analytic'),
    GPDC(significance='analytic'),
    #GPDC(significance='shuffle_test'),
    # CMIknn(significance='shuffle_test', transform='ranks'),
    # CMIknnMixed(significance='shuffle_test', sig_samples=100, knn=50, shuffle_neighbors=8, transform='ranks')
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
    results_r[test.__class__.__name__] = pcmci.run_rpcmci(tau_max=tau_max, num_regimes=2, max_transitions=1, pc_alpha=0.05)
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
        node_colorbar_label='auto-MCI',
        #figsize=(10, 8),
        #save_name='pcmci_results_{}.png'.format(_)
    )
    plt.show()
#

plt.plot(results_r['GPDC']['regimes'][0]); plt.show()

for _ in results_r.keys():
    for reg in range(2):
        tp.plot_graph(
            val_matrix=results_r[_]['causal_results'][reg]['val_matrix'],
            graph=results_r[_]['causal_results'][reg]['graph'],
            var_names=var_names,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            #figsize=(10, 8),
            #save_name='rpcmci_results_{}.png'.format(_+str(reg))
        )
        plt.show()
#

for _ in results_l.keys():
    tp.plot_graph(
        val_matrix=results_l[_]['val_matrix'],
        graph=results_l[_]['graph'],
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        #figsize=(10, 8),
        #save_name='pcmci_results_{}.png'.format(_)
    )
    plt.show()
#


import pickle
# Save the results to a file
with open('pcmci_results.pkl', 'wb') as f:
    pickle.dump((results, results_r, results_l), f)

import pickle
# Load the results from a file
with open('pcmci_results.pkl', 'rb') as f:
    results, results_r, results_l = pickle.load(f)
#####



from LIN.models.LIN_GaussianNet import LIN_GaussianNet
import torch
import pandas as pd

# 1. Format the time-series dataset
# The fit method expects a tensor of shape (T, d)
if isinstance(y, pd.DataFrame):
    y_tens = torch.tensor(y.to_numpy(), dtype=torch.float32)
elif not isinstance(y, torch.Tensor):
    y_tens = torch.tensor(y, dtype=torch.float32)

y_tens = torch.tensor(scaled_data, dtype=torch.float32)

# 2. Define exact architectural parameters
d = y.shape[1]  # Number of endogenous variables in the system
P = 4           # SVAR time lag order (e.g., P=1 for an SVAR(1) process)
E = 2           # Estimated number of latent interventions/regimes
    
# 3. Define arguments for the underlying Interventions neural network
intv_args = {
    'hidden_dim': 64,   # Dimension of the hidden layers
    'n_hidden_lyr': 4,  # Number of hidden layers
}

# 4. Initialize the LIN_GaussianNet model
model = LIN_GaussianNet(
    d=d, 
    P=P, 
    E=E, 
    intv_args=intv_args,
    device='cpu'
)

# 5. Execute the built-in training loop[cite: 2]
# CRITICAL FIX: struct_pnt_coeff must be a list of 3 elements. 
# The repository's default is a float (1e-8), which will cause a TypeError 
# when the source code attempts to call `struct_pnt_coeff[2]`.
best_model = model.fit(
    data=y_tens, 
    epoch=500,
    batch_size=256,
    struct_pnt_coeff=[0.01, 0.01, 0.01] 
)

import networkx as nx
import matplotlib.pyplot as plt

# 1. Extract the contemporaneous adjacency matrix (lag 0)
# Applying the sigmoid thresholding as defined in LIN_GaussianNet.py
adj_matrix = (torch.sigmoid(best_model.G[0]) >= 0.5).int().cpu().numpy()

# 2. Create the Directed Graph
G_nx = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

# Optional: Map your DataFrame column names to the nodes if you have them
variable_names = {i: col_name for i, col_name in enumerate(y.columns)}
nx.relabel_nodes(G_nx, variable_names, copy=False)

# 3. Plot the DAG
plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G_nx) # circular or spring_layout work well
nx.draw_networkx(
    G_nx, 
    pos, 
    with_labels=True, 
    node_color='lightblue', 
    node_size=2000, 
    font_size=12, 
    font_weight='bold', 
    arrows=True, 
    arrowsize=20
)
plt.title("Learned Contemporaneous Causal Graph (DAG)")
plt.show()


# 1. Extract the intervention probabilities per time step
# best_model.intv_p shape is (T, E), where T is time and E is the number of clusters
regime_probs = best_model.intv_p.detach().cpu().numpy()

# 2. Assign each time step to its most likely regime
assigned_regimes = regime_probs.argmax(axis=1)

# 3. Plot the regime timeline
plt.figure(figsize=(12, 4))
plt.plot(assigned_regimes, drawstyle='steps-mid', linewidth=2, color='darkorange')

plt.title("Latent Structural Regimes over Time")
plt.xlabel("Time Step (Index)")
plt.ylabel("Regime Assignment (0 to E-1)")
plt.yticks(range(best_model.E)) # best_model.E is the total number of interventions
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()