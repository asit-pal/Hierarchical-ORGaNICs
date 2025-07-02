# Essential imports
import os
import sys
import yaml
import numpy as np
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Communication import calculate_pred_performance_freq, calculate_dim_vs_freq


def main(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config from: {config_file}")
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Load config from yaml file
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Get results directory from config file path
    results_dir = os.path.dirname(os.path.abspath(config_file))
    data_dir = os.path.join(results_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize model parameters
    # Here we use defaults similar to Communication_analysis.py (e.g. N=36*2)
    params = setup_parameters(
        config=config,
        tau=1e-3,
        tauPlus=1e-3,
        N=36*2,
        M=None
    )
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Set up indices based on simulation type
    N = params['N']
    if Ring_Model.simulate_firing_rates:
        N1_y = np.arange(N, 2 * N)    # index for y1Plus
        N4_y = np.arange(3 * N, 4 * N)  # index for y4Plus   
        bw_y1_y4 = False
    else:
        N1_y = np.arange(0 * N, N)
        N4_y = np.arange(1 * N, 2 * N)
        bw_y1_y4 = True

    # Communication subspace parameters (com_params)
    com_params = {
        'num_trials':  300,
        'tol': 5e-2,
        'V1_s': 30,
        'V1_t': 30,
        'V4_t': 30,
        'N1_y_idx': N1_y,
        'N4_y_idx': N4_y,
        'bw_y1_y4': bw_y1_y4
    }
    min_freq = 1
    max_freq = 100
    n_freq_mat = 100
    freq = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    
    # Run Frequency Analysis based on config settings
    if config['Communication']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        contrast_vals = config['Communication']['Feedback_gain']['c_vals']
        gamma_vals = config['Communication']['Feedback_gain']['gamma_vals']
        t_span = config['Communication']['Feedback_gain']['t_span']
        
        # Note: calculate_pred_performance_freq is assumed to handle a list of contrasts
        frequency_data = calculate_pred_performance_freq(
            Ring_Model,
            gamma_vals,
            contrast_vals,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise_potential=config['noise_params']['noise_potential'],
            noise_firing_rate=config['noise_params']['noise_firing_rate'],
            GR_noise=config['noise_params']['GR_noise'],
            method='RK45',
            com_params=com_params,
            # Here, our function must accept lists for contrasts or process them internally.
            freq=freq,  # Assume the function uses config-defined frequency parameters if needed
            t_span=t_span
        )
        save_frequency_data(frequency_data, data_dir, gain_type, freq)


def save_frequency_data(frequency_data, data_dir, gain_type, freq):
    """
    Save frequency analysis data to a single file.
    
    Args:
        frequency_data (dict): Dictionary containing frequency analysis data.
        data_dir (str): Directory to save the data.
        gain_type (str): Type of gain ('fb_gain' or 'input_gain_beta1').
        freq (torch.Tensor): Frequency tensor.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f'Frequency_data_{gain_type}.npy'
    # Combine performance data and frequency array into one dictionary
    data_to_save = {
        'performance_data': frequency_data,
        'freq': freq.numpy() # Convert torch tensor to numpy array for saving
    }
    np.save(os.path.join(data_dir, filename), data_to_save)
    print(f"Saved frequency data and frequencies to: {os.path.join(data_dir, filename)}")
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Freq_wise_decom.py path/to/config.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)



