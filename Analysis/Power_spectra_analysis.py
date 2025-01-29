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
from Utils.Coherence import Calculate_power_spectra

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
    i = int(N / 2)  # y1 LFP power index
    
    # Run analyses based on config
    if config['Power_spectra']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        contrast_vals = config['Power_spectra']['Feedback_gain']['c_vals']
        gamma_vals = config['Power_spectra']['Feedback_gain']['gamma_vals']
        Power_data = Calculate_power_spectra(
            Ring_Model,
            i,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise_potential=config['noise_params']['noise_potential'],
            noise_firing_rate=config['noise_params']['noise_firing_rate'],
            GR_noise=config['noise_params']['GR_noise'],
            low_pass_add=config['noise_params']['low_pass_add'],
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            contrast_vals=contrast_vals,
            method='RK45',
            gamma_vals=gamma_vals,
            min_freq=1,
            max_freq=200,
            n_freq_mat=500,
            t_span=[0, 6],
        )
        # Save all data to a single file
        save_power_spectra(Power_data, data_dir, gain_type)
    
    if config['Power_spectra']['Input_gain_beta1']['enabled']:
        gain_type = 'input_beta1_gain'
        contrast_vals = config['Power_spectra']['Input_gain_beta1']['c_vals']
        beta1_vals = config['Power_spectra']['Input_gain_beta1']['beta1_vals']
        Power_data = Calculate_power_spectra(
            Ring_Model,
            i,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise_potential=config['noise_params']['noise_potential'],
            noise_firing_rate=config['noise_params']['noise_firing_rate'],
            GR_noise=config['noise_params']['GR_noise'],
            low_pass=config['noise_params']['low_pass'],
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            contrast_vals=contrast_vals,
            method='RK45',
            beta1_vals=beta1_vals,
            min_freq=1,
            max_freq=200,
            n_freq_mat=500,
            t_span=[0, 6],
        )
        save_power_spectra(Power_data, data_dir, gain_type)
    
def save_power_spectra(power_data, data_dir, gain_type):
    """
    Save power spectra data to a single file.
    
    Args:
        power_data (dict): Dictionary containing power spectra data
        data_dir (str): Directory to save the data
        gain_type (str): Type of gain ('fb', 'beta1', or 'beta4')
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save all data to a single file
    filename = f'power_spectra_{gain_type}_all.npy'
    filepath = os.path.join(data_dir, filename)
    np.save(filepath, power_data)
    print(f"Saved all power spectra data to: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Power_spectra.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)  # Pass the config_file to main


