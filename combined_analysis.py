# Essential imports
import os
import json
import numpy as np
from Create_weight_matrices import *
from Model import *
from Coherence import *
from Communication import *

def create_directories():
    """Create necessary directories for data storage"""
    base_dir = 'Data'
    subdirs = ['Parameters', 'Gain_modulation', 'Power_spectra', 
               'Coherence', 'Communication']
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def save_parameters(params_dict, filename='Data/Parameters/analysis_parameters.json'):
    """Save parameters to JSON file"""
    with open(filename, 'w') as f:
        json.dump(params_dict, f, indent=4)

def main():
    # Create directories
    create_directories()
    
    # Initialize common parameters
    params = setup_parameters(tau=1e-3, kernel=None, N=36, M=None, tauPlus=1e-3)
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01
    
    # Common analysis parameters
    common_params = {
        'noise': 0.01,
        'delta_tau': 3 * params['tau'],
        'method': 'RK45',
        't_span': [0, 6],
        'contrast_values': [0.032, 0.064, 0.125, 0.25, 0.5, 1.0],
        'gamma_values': [0.0, 0.25, 0.50, 0.75, 1.0],
        'beta1_values': [0.5, 0.75, 1.0, 1.25, 1.50],
    }
    
    # Save all parameters
    save_parameters({
        'model_params': params,
        'analysis_params': common_params
    })

    # ---------------------- 1. Gain Modulation Analysis ----------------------
    print("Running Gain Modulation Analysis...")
    for contrast in common_params['contrast_values']:
        # Feedback gain analysis
        steady_states_gamma1 = Contrast_response_curve(
            Ring_Model, params, 
            common_params['contrast_values'],
            common_params['gamma_values'],
            common_params['method'],
            common_params['t_span'],
            initial_conditions,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False
        )
        np.save(f'Data/Gain_modulation/fb_gain_contrast_{contrast}.npy', 
                steady_states_gamma1)
        
        # Input gain analysis
        steady_states_beta1 = Contrast_response_curve(
            Ring_Model, params,
            common_params['contrast_values'],
            common_params['beta1_values'],
            common_params['method'],
            common_params['t_span'],
            initial_conditions,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False
        )
        np.save(f'Data/Gain_modulation/input_gain_contrast_{contrast}.npy', 
                steady_states_beta1)

    # ---------------------- 2. Power Spectra Analysis ----------------------
    print("Running Power Spectra Analysis...")
    N = params['N']
    i = int(N / 2)  # y1 LFP power index
    
    for contrast in common_params['contrast_values']:
        # Feedback gain power analysis
        Power_data_fb = Calculate_power_spectra(
            Ring_Model, i,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            contrast=contrast,
            method=common_params['method'],
            param_vals=common_params['gamma_values']
        )
        np.save(f'Data/Power_spectra/fb_gain_contrast_{contrast}.npy', 
                Power_data_fb)
        
        # Input gain power analysis
        Power_data_input = Calculate_power_spectra(
            Ring_Model, i,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            contrast=contrast,
            method=common_params['method'],
            param_vals=common_params['beta1_values']
        )
        np.save(f'Data/Power_spectra/input_gain_contrast_{contrast}.npy', 
                Power_data_input)

    # ---------------------- 3. Coherence Analysis ----------------------
    print("Running Coherence Analysis...")
    j = int(N / 2) + 2*int(N) if Ring_Model.simulate_firing_rates else int(N / 2) + int(N)
    
    for contrast in common_params['contrast_values']:
        # Feedback gain coherence
        coherence_data_fb = calculate_coherence(
            Ring_Model, i, j,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            contrast=contrast,
            method=common_params['method'],
            param_vals=common_params['gamma_values']
        )
        np.save(f'Data/Coherence/fb_gain_contrast_{contrast}.npy', 
                coherence_data_fb)
        
        # Input gain coherence
        coherence_data_input = calculate_coherence(
            Ring_Model, i, j,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            contrast=contrast,
            method=common_params['method'],
            param_vals=common_params['beta1_values']
        )
        np.save(f'Data/Coherence/input_gain_contrast_{contrast}.npy', 
                coherence_data_input)

    # ---------------------- 4. Communication Analysis ----------------------
    print("Running Communication Analysis...")
    # Set up indices for communication analysis
    N1_y = np.arange(N, 2*N)
    N4_y = np.arange(3*N, 4*N)
    
    com_params = {
        'num_trials': 250,
        'tol': 5e-2,
        'V1_s': 18,
        'V1_t': 18,
        'V4_t': 18,
        'N1_y_idx': N1_y,
        'N4_y_idx': N4_y,
        'bw_y1_y4': False
    }
    
    for contrast in common_params['contrast_values']:
        # Feedback gain communication
        comm_data_fb = Calculate_Pred_perf_Dim(
            Ring_Model,
            common_params['gamma_values'],
            contrast,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            method=common_params['method'],
            com_params=com_params,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            t_span=common_params['t_span']
        )
        np.save(f'Data/Communication/fb_gain_contrast_{contrast}.npy', 
                comm_data_fb)
        
        # Input gain communication
        comm_data_input = Calculate_Pred_perf_Dim(
            Ring_Model,
            common_params['beta1_values'],
            contrast,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            method=common_params['method'],
            com_params=com_params,
            delta_tau=common_params['delta_tau'],
            noise=common_params['noise'],
            t_span=common_params['t_span']
        )
        np.save(f'Data/Communication/input_gain_contrast_{contrast}.npy', 
                comm_data_input)

if __name__ == "__main__":
    main() 