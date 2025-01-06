# Essential imports
import yaml
import numpy as np
import os
import sys
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Communication import Calculate_Pred_perf_Dim, Calculate_Fano_Factor

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
    data_dir = os.path.join(results_dir, 'Data', 'Fano_factor_data')
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
    
    # Communication subspace parameters
    com_params = {
        'num_trials': 250,
        'tol': 5e-2,
        'V1_s': 18,
        'V1_t': 18,
        'V4_t': 18,
        'N_idx': N + int(N/2),  # y1Plus center neuron index
        'bw_y1_y4': False
    }
    
    # Run analyses based on config
    if config['Communication']['Feedback_gain']['enabled']:
        run_fano_factor_analysis(Ring_Model, config, com_params, data_dir)

def run_fano_factor_analysis(Ring_Model, config, com_params, data_dir):
    """Run Fano factor analysis for different feedback gains and contrasts"""
    fb_config = config['Communication']['Feedback_gain']
    
    for contrast in fb_config['c_vals']:
        fano_data = Calculate_Fano_Factor(
            Ring_Model,
            fb_config['gamma_vals'],
            contrast,
            g=config['model_params']['g1'],
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            method='RK45',
            com_params=com_params,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            baseline=None,
            poisson=True,
            t_span=fb_config['t_span']
        )
        
        if fb_config['save_data']:
            filename = f'fano_factor_neuron_{com_params["N_idx"]}_contrast_{contrast}.npy'
            np.save(os.path.join(data_dir, filename), fano_data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Fano_factor.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)



