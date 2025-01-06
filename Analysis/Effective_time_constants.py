# Essential imports
import yaml
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Plotting import plot_decay_trajectories
from Utils.Coherence import get_effective_timescales

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
    plot_dir = os.path.join(results_dir, 'Plots')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
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
    initial_conditions = np.ones((Ring_Model.num_var * N)) * 0.01
    
    # Calculate effective time constants for different conditions
    timescale_data = {}
    
    if config['Power_spectra']['Feedback_gain']['enabled']:
        contrast_vals = config['Power_spectra']['Feedback_gain']['c_vals']
        gamma_vals = config['Power_spectra']['Feedback_gain']['gamma_vals']
        
        for gamma in gamma_vals:
            # Update model parameters for this gamma value
            updated_params = params.copy()
            updated_params['gamma1'] = gamma
            Ring_Model.params = updated_params
            
            for contrast in contrast_vals:
                print(f"Calculating effective timescales for gamma={gamma}, contrast={contrast}")
                
                # Calculate effective timescales and get decay data
                tau_eff, decay_data = get_effective_timescales(
                    model=Ring_Model,
                    c=contrast,
                    initial_conditions=initial_conditions,
                    method='RK45',
                    t_span=[0, 6],
                    decay_time=0.25,
                    plot=True
                )
                
                # Store results
                key = (gamma, contrast)
                timescale_data[key] = {
                    'tau_eff': tau_eff,
                    'decay_data': decay_data
                }
                
                # Create plots
                fig, axs = plot_decay_trajectories(decay_data, center_idx=Ring_Model.N//2, plot_analytical=False)
                plt.suptitle(f'Decay Trajectories (γ={gamma}, c={contrast})')
                
                # Save plot
                plot_filename = f'decay_trajectories_gamma{gamma}_contrast{contrast}.png'
                plot_path = os.path.join(plot_dir, plot_filename)
                plt.savefig(plot_path)
                plt.close()
    
    # Save all data
    save_timescale_data(timescale_data, data_dir)

def save_timescale_data(timescale_data, data_dir):
    """
    Save effective timescale data to a file.
    
    Args:
        timescale_data (dict): Dictionary containing timescale data
        data_dir (str): Directory to save the data
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save all data to a single file
    filename = 'effective_timescales_all.npy'
    filepath = os.path.join(data_dir, filename)
    np.save(filepath, timescale_data)
    print(f"Saved all timescale data to: {filepath}")
    
    # Also save a summary of just the time constants
    summary_data = {key: data['tau_eff'] for key, data in timescale_data.items()}
    summary_filename = 'effective_timescales_summary.npy'
    summary_filepath = os.path.join(data_dir, summary_filename)
    np.save(summary_filepath, summary_data)
    print(f"Saved timescale summary to: {summary_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Effective_time_constants.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    main(config_file)