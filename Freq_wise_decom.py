# Essential imports
import torch
import numpy as np
from tqdm.notebook import tqdm
from Create_weight_matrices import setup_parameters
from Model import RingModel
from Communication import calculate_pred_performance_freq, calculate_dim_vs_freq

# Initialize model parameters
params = setup_parameters(tau=1e-3, kernel=None, N=36, M=None, tauPlus=1e-3)
Ring_Model = RingModel(params, simulate_firing_rates=True)
initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01

# Set up indices based on simulation type
N = params['N']
if Ring_Model.simulate_firing_rates:
    N1_y = np.arange(N, 2*N)    # index for y1Plus
    N4_y = np.arange(3*N, 4*N)  # index for y4Plus   
    bw_y1_y4 = False
else:
    N1_y = np.arange(0*N, N)    # index for y1
    N4_y = np.arange(1*N, 2*N)  # index for y4
    bw_y1_y4 = True

# Analysis parameters
noise = 0.01
delta_tau = 3*params['tau']
com_params = {
    'num_trials': 250,
    'tol': 5e-2,
    'V1_s': 20,
    'V1_t': 15,
    'V4_t': 15,
    'N1_y_idx': N1_y,
    'N4_y_idx': N4_y,
    'bw_y1_y4': bw_y1_y4
}

# ---------------------- Frequency Analysis ----------------------
# Simulation parameters
freq = torch.logspace(np.log10(1), np.log10(500), 100)
gamma_vals = [0.3, 0.6]
c_vals = [0.032, 0.064, 0.125, 0.25, 0.5, 1.0]
method = 'RK45'
t_span = [0, 6]

# Run frequency analysis for each contrast value
for contrast in c_vals:
    prediction_data = calculate_pred_performance_freq(
        Ring_Model, 
        gamma_vals,
        contrast,
        fb_gain=True,
        input_gain_beta1=False,
        input_gain_beta4=False,
        method=method,
        com_params=com_params, 
        delta_tau=delta_tau, 
        noise=noise, 
        freq=freq, 
        bw_y1_y4=bw_y1_y4, 
        t_span=t_span
    )
    np.save(f'Data/Freq_wise_decom_CS/fb_gain_contrast_{contrast}.npy', prediction_data)

# ---------------------- Dimensionality Analysis ----------------------
# Update parameters for dimensionality analysis
gamma_vals = [1.0]
c_vals = [1.0]
threshold = 0.90

# Run dimensionality analysis for each contrast value
for contrast in c_vals:
    dimension_data = calculate_dim_vs_freq(
        Ring_Model, 
        gamma_vals, 
        contrast,
        fb_gain=True,
        input_gain_beta1=False,
        input_gain_beta4=False, 
        method=method,
        com_params=com_params,  
        delta_tau=delta_tau, 
        noise=noise, 
        freq=freq,
        threshold=threshold, 
        bw_y1_y4=bw_y1_y4, 
        t_span=t_span
    )
    np.save(f'Data/Dim_vs_freq_data/fb_gain_contrast_{contrast}_gamma_1.0.npy', dimension_data)



