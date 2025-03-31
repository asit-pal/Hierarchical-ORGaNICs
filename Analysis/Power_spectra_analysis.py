# Essential imports
import os
import sys
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import numpy as np
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Coherence import Calculate_power_spectra, create_S_matrix, create_L_matrix

def main(config_file, area='V1'):
    """
    Main function to analyze power spectra.
    Args:
        config_file (str): Path to config file
        area (str): Brain area to analyze ('V1' or 'V2')
    """
    print(f"Analyzing {area} power spectra")
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
    
    # Select index based on area
    if area.upper() == 'V1':
        i = int(N / 2)  # y1 LFP power index
    elif area.upper() == 'V2':
        i = int(2*N) + int(N/2)  # y2 LFP power index
    else:
        raise ValueError(f"Invalid area: {area}. Must be 'V1' or 'V2'")
    
    # Run analyses based on config
    if config['Power_spectra']['Feedback_gain']['enabled']:
        gain_type = f'fb_gain_{area.lower()}'  # Add area to filename
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
            tau_f = config['noise_params']['tau_f'],
            sigma_f = config['noise_params']['sigma_f'],
            min_freq=1,
            max_freq=1000,
            n_freq_mat=500,
            t_span=[0, 6],
        )
        # Save all data to a single file
        save_power_spectra(Power_data, data_dir, gain_type)
    
    if config['Power_spectra']['Input_gain_beta1']['enabled']:
        gain_type = f'input_gain_beta1_{area.lower()}'  # Add area to filename
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
            low_pass_add=config['noise_params']['low_pass_add'],
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            contrast_vals=contrast_vals,
            method='RK45',
            gamma_vals=beta1_vals,
            tau_f = config['noise_params']['tau_f'],
            sigma_f = config['noise_params']['sigma_f'],
            min_freq=1,
            max_freq=1000,
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
    filename = f'power_spectra_{gain_type}.npy'
    filepath = os.path.join(data_dir, filename)
    np.save(filepath, power_data)
    print(f"Saved power spectra data to: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze power spectra for V1 or V2')
    parser.add_argument('config_file', help='Path to config file')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to analyze (V1 or V2)')
    args = parser.parse_args()
    
    main(args.config_file, args.area)


