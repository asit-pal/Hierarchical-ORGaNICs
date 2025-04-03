# Standard library imports
import os
import sys
import argparse

# Third-party library imports       
import autograd.numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Local imports
from Three_area.Parameters_3 import setup_parameters
from Three_area.Communication_3 import calculate_pred_performance_dim_3
from Three_area.Three_area_model import RingModel

# Initialize model parameters
params = setup_parameters(tau=1e-3, kernel=None, N=36*2, M=None, tauPlus=1e-3)
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

# noise parameters
noise_potential = 0.000
noise_firing_rate = 0.001
GR_noise = True
delta_tau = 3 * params['tau']

com_params = {
    'num_trials': 300,
    'tol': 5e-2,
    'V1_s': 30, 'V1_t': 30, 'V4_t': 30, 'V5_t': 30,
    'N1_y_idx': N1_y, 'N4_y_idx': N4_y, 'N5_y_idx': N5_y,
    'bw_y1_y4': bw_y1_y4
}

# pairs = [(0.50,0.50,1.0)] #(g, gamma14, gamma15)
pairs = [(1.0,0.50), (0.50,1.0)]

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Three-Area Communication Analysis for a specific configuration.')
parser.add_argument('config_num', type=int, help='Configuration number (e.g., 74)')
args = parser.parse_args()
config_num = args.config_num

contrast = 1.0
method = 'RK45'
t_span = [0, 6]
Pred_perf_vs_dim_data = calculate_pred_performance_dim_3(Ring_Model, pairs, contrast, method, com_params, delta_tau, noise_potential, noise_firing_rate, GR_noise, t_span)

# Create data directory if it doesn't exist
results_dir = os.path.join(project_root, 'Results_4', f'config_{config_num}')
data_dir = os.path.join(results_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)

# Save data to file using savez
data_file = os.path.join(data_dir, 'Pred_perf_vs_dim_data.npz') # Change extension back to .npz
np.savez(data_file, 
         performance_data=Pred_perf_vs_dim_data, 
         pairs=pairs, 
         contrast=contrast)
print(f"Saved prediction performance data to: {data_file}")




