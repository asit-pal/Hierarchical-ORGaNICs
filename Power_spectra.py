# Essential imports
import yaml
import numpy as np
import os
import sys
from Create_weight_matrices import setup_parameters
from Model import RingModel
from Coherence import Calculate_power_spectra

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
        run_feedback_gain_analysis(Ring_Model, i, config, data_dir)
    
    if config['Power_spectra']['Input_gain_beta1']['enabled']:
        run_input_gain_beta1_analysis(Ring_Model, i, config, data_dir)

def run_feedback_gain_analysis(Ring_Model, i, config, data_dir):
    fb_config = config['Power_spectra']['Feedback_gain']
    for contrast in fb_config['c_vals']:
        Power_data = Calculate_power_spectra(
            Ring_Model,
            i,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            baseline=0.000,
            poisson=False,
            contrast=contrast,
            method='RK45',
            gamma_vals=fb_config['gamma_vals']
        )
        # Save Power_data
        filename = f'power_fb_gain_contrast_{contrast}.npy'
        np.save(os.path.join(data_dir, filename), Power_data)
    # return Power_data

def run_input_gain_beta1_analysis(Ring_Model, i, config, data_dir):
    beta1_config = config['Power_spectra']['Input_gain_beta1']
    for contrast in beta1_config['c_vals']:
        Power_data = Calculate_power_spectra(
            Ring_Model,
            i,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            contrast=contrast,
            method='RK45',
            beta1_vals=beta1_config['beta1_vals']
        )
        # Save Power_data
        filename = f'power_beta1_contrast_{contrast}.npy'
        np.save(os.path.join(data_dir, filename), Power_data)
    # return Power_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Power_spectra.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)  # Pass the config_file to main


