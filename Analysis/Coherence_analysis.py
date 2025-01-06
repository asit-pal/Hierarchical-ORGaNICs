# Essential imports
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import numpy as np
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Coherence import Calculate_coherence 

def main(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config from: {config_file}")
    
    # Load config from yaml file
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
        print(f"Absolute path: {os.path.abspath(config_file)}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    
    # Get results directory from config file path
    results_dir = os.path.dirname(config_file)
    data_dir = os.path.join(results_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    # Initialize model parameters
    params = setup_parameters(
        config=config,
        tau=1e-3,
        tauPlus=1e-3,
        N=36
    )
    
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Analysis parameters
    N = params['N']
    i = int(N / 2)  # y1 index
    j = int(N / 2) + 2*int(N) if Ring_Model.simulate_firing_rates else int(N / 2) + int(N)  # y4 index

    # Run analyses based on config
    if config['Coherence']['Feedback_gain']['enabled']:
        fb_config = config['Coherence']['Feedback_gain']
        coherence_data = Calculate_coherence(
            Ring_Model,
            i, j,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            poisson=config['noise_params']['poisson'],
            get_simulated_taus=config['noise_params']['get_simulated_taus'],
            low_pass_add=config['noise_params']['low_pass_add'],
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            contrast_vals=fb_config['c_vals'],
            method='RK45',
            gamma_vals=fb_config['gamma_vals'],
            min_freq=0.1,
            max_freq=200,
            n_freq_mat=500,
            t_span=fb_config['t_span']
        )
        if fb_config['save_data']:
            save_coherence(coherence_data, data_dir, 'fb_gain')
    
    if config['Coherence']['Input_gain_beta1']['enabled']:
        beta1_config = config['Coherence']['Input_gain_beta1']
        coherence_data = Calculate_coherence(
            Ring_Model,
            i, j,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            poisson=config['noise_params']['poisson'],
            get_simulated_taus=config['noise_params']['get_simulated_taus'],
            low_pass_add=config['noise_params']['low_pass_add'],
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            contrast_vals=beta1_config['c_vals'],
            method='RK45',
            beta1_vals=beta1_config['beta1_vals'],
            min_freq=0.1,
            max_freq=200,
            n_freq_mat=500,
            t_span=beta1_config['t_span']
        )
        if beta1_config['save_data']:
            save_coherence(coherence_data, data_dir, 'input_beta1_gain')

def save_coherence(coherence_data, data_dir, gain_type):
    """
    Save coherence data to a single file.
    
    Args:
        coherence_data (dict): Dictionary containing coherence data
        data_dir (str): Directory to save the data
        gain_type (str): Type of gain ('fb_gain' or 'input_beta1_gain')
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save all data to a single file
    filename = f'coherence_{gain_type}_all.npy'
    filepath = os.path.join(data_dir, filename)
    np.save(filepath, coherence_data)
    print(f"Saved coherence data to: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Coherence_analysis.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)


