# Essential imports
import autograd.numpy as np
import tqdm
from Create_weight_matrices import *
from Model import *
from Plotting import *

# Initialize model parameters
params = setup_parameters(
    tau=1e-3,
    kernel=None, 
    N=36, 
    M=None,
    tauPlus=1e-3
)
Ring_Model = RingModel(params, simulate_firing_rates=False)
initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01

# ---------------------- Feedback Gain Analysis (γ1) ----------------------
# Simulation parameters
t_span = [0, 5]
gamma_vals = [0.25, 0.50, 0.75, 1.0]  # Feedback gain values
c_vals = np.logspace(np.log10(1e-2), np.log10(1), 40)  # Contrast values
method = 'RK45'

# Run feedback gain simulation
steady_states_gamma1 = Contrast_response_curve(
    Ring_Model,
    params,
    c_vals,
    gamma_vals,
    method, 
    t_span,
    initial_conditions,
    fb_gain=True,
    input_gain_beta1=False,
    input_gain_beta4=False
)

# Save feedback gain results
np.save('Data/Gain_modulation/Feedback_Gain_gamma_c.npy', steady_states_gamma1)

# ---------------------- Input Gain Analysis (β1) ----------------------
# Update simulation parameters
t_span = [0, 6]
beta1_vals = [1.0, 1.25, 1.5, 1.75]  # Input gain values
c_vals = np.logspace(np.log10(1e-3), np.log10(1), 40)  # Contrast values
method = 'LSODA'

# Run input gain simulation
steady_states_beta1 = Contrast_response_curve(
    Ring_Model,
    params,
    c_vals,
    beta1_vals,
    method, 
    t_span,
    initial_conditions,
    fb_gain=False,
    input_gain_beta1=True,
    input_gain_beta4=False
)

# Save input gain results
np.save('Data/Gain_modulation/Input_Gain_beta1_c.npy', steady_states_beta1)



