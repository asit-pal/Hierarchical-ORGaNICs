# Essential imports
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import numpy as np
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel, Contrast_response_curve

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
    
    # Run analyses based on config
    if config['Gain_modulation']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        gamma_vals = config['Gain_modulation']['Feedback_gain']['gamma_vals']
        steady_states = Contrast_response_curve(
            model=Ring_Model,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            c_vals=np.logspace(np.log10(1e-2), np.log10(1), 40),
            gamma_vals=gamma_vals,
            method='RK45',
            t_span=config['Gain_modulation']['Feedback_gain']['t_span']
        )
        save_steady_states(steady_states, data_dir, gain_type)
    
    if config['Gain_modulation']['Input_gain_beta1']['enabled']:
        gain_type = 'input_gain_beta1'
        beta1_vals = config['Gain_modulation']['Input_gain_beta1']['beta1_vals']
        steady_states = Contrast_response_curve(
            model=Ring_Model,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            c_vals=np.logspace(np.log10(1e-2), np.log10(1), 40),
            gamma_vals=beta1_vals,
            method='RK45',
            t_span=config['Gain_modulation']['Input_gain_beta1']['t_span']
        )
        save_steady_states(steady_states, data_dir, gain_type)

def save_steady_states(steady_states, data_dir, gain_type):
    """
    Save steady states data to a single file.
    
    Args:
        steady_states (dict): Dictionary containing steady states data
        data_dir (str): Directory to save the data
        gain_type (str): Type of gain ('fb_gain' or 'input_beta1_gain')
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save all data to a single file
    filename = f'steady_states_{gain_type}_all.npy'
    filepath = os.path.join(data_dir, filename)
    np.save(filepath, steady_states)
    print(f"Saved steady states data to: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Gain_modulation.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)  # Pass the config_file to main



