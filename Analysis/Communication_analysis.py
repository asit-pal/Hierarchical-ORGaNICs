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
from Utils.Communication import Calculate_Pred_perf_Dim

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
    
    # Initialize model parameters - using default tau and N since they're not in config
    params = setup_parameters(
        config=config,
        tau=1e-3,  # default value
        tauPlus=1e-3,
        N=72       # default value
    )
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Initialize initial conditions
    # initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01

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
    # Analysis parameters using exact config structure
    # Communication subspace parameters
    com_params = {'num_trials':250,
                'tol':5e-2,
                'V1_s':36,
                'V1_t':36,
                'V4_t':36,
                'N1_y_idx':N1_y,
                'N4_y_idx':N4_y,
                'bw_y1_y4':bw_y1_y4} # is it between potential or firing rates?

    # Run analyses based on config
    if config['Communication']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        contrast_vals = config['Communication']['Feedback_gain']['c_vals']
        gamma_vals = config['Communication']['Feedback_gain']['gamma_vals']
        Communication_data, covariance_data = Calculate_Pred_perf_Dim(
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
            t_span=config['Communication']['Feedback_gain']['t_span']
        )
        save_communication_data(Communication_data, data_dir, gain_type)
        save_covariance_data(covariance_data, data_dir, gain_type)
    if config['Communication']['Input_gain_beta1']['enabled']:
        gain_type = 'input_beta1_gain'
        contrast_vals = config['Communication']['Input_gain_beta1']['c_vals']
        beta1_vals = config['Communication']['Input_gain_beta1']['beta1_vals']
        Communication_data = Calculate_Pred_perf_Dim(
            Ring_Model,
            beta1_vals,
            contrast_vals,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise_potential=config['noise_params']['noise_potential'],
            noise_firing_rate=config['noise_params']['noise_firing_rate'],
            GR_noise=config['noise_params']['GR_noise'],
            method='RK45',
            com_params=com_params,
            t_span=config['Communication']['Input_gain_beta1']['t_span']
        )

        save_communication_data(Communication_data, data_dir, gain_type)

def save_communication_data(Communication_data, data_dir, gain_type):
    """
    Save communication data to a single file.
    
    Args:
        Communication_data (dict): Dictionary containing communication data
        data_dir (str): Directory to save the data
        gain_type (str): Type of gain ('fb', 'beta1', or 'beta4')
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save all data to a single file
    filename = f'Communication_data_{gain_type}_all.npy'
    np.save(os.path.join(data_dir, filename), Communication_data)
    print(f"Saved all communication data to: {os.path.join(data_dir, filename)}")
    
def save_covariance_data(covariance_data, data_dir, gain_type):
    """
    Save covariance data to a single file.
    
    Args:
        covariance_data (dict): Dictionary containing covariance data
        data_dir (str): Directory to save the data
        gain_type (str): Type of gain ('fb', 'beta1', or 'beta4')
    """
    filename = f'Covariance_data_{gain_type}_all.npy'
    np.save(os.path.join(data_dir, filename), covariance_data)
    print(f"Saved all covariance data to: {os.path.join(data_dir, filename)}")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Communication_analysis.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)  # Pass the config_file to main



