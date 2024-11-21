# Standard library imports
import os

# Third-party library imports
import torch
import autograd.numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Local imports
from Four_neuron_model.Script_analysis.Three_area.Parameters_3 import setup_parameters
from Four_neuron_model.Script_analysis.Three_area.Communication_3 import calculate_pred_performance_dim_3
from Four_neuron_model.Script_analysis.Three_area.Three_area_model import RingModel

# Initialize model parameters
params = setup_parameters(tau=1e-3, kernel=None, N=36, M=None, tauPlus=1e-3)
Ring_Model = RingModel(params, simulate_firing_rates=True)
initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01

# Set up indices based on simulation type
N = Ring_Model.params['N']
if Ring_Model.simulate_firing_rates:
    N1_y = np.arange(N, 2 * N)
    N4_y = np.arange(3 * N, 4 * N)
    N5_y = np.arange(5 * N, 6 * N)
    bw_y1_y4 = False  # Between firing rates
else:
    N1_y = np.arange(0 * N, 1 * N)
    N4_y = np.arange(1 * N, 2 * N)
    N5_y = np.arange(2 * N, 3 * N)
    bw_y1_y4 = True   # Between LFPs

# Analysis parameters
noise = 0.01
delta_tau = 3 * params['tau']
com_params = {
    'num_trials': 250,
    'tol': 5e-2,
    'V1_s': 18, 'V1_t': 18, 'V4_t': 18, 'V5_t': 18,
    'N1_y_idx': N1_y, 'N4_y_idx': N4_y, 'N5_y_idx': N5_y,
    'bw_y1_y4': bw_y1_y4
}

pairs = [(0.50,1.0,0.50), (0.50,0.50,1.0)] #(g, gamma14, gamma15)
contrast = 1.0
method = 'RK45'
t_span = [0, 6]
Ring_Model.params['beta1'] = 0.5
Pred_perf_vs_dim_data = calculate_pred_performance_dim_3(Ring_Model, pairs, contrast, method, com_params, delta_tau, noise, t_span)




