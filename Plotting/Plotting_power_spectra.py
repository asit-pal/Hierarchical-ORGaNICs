import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_power_spectra 

def plot_feedback_gain_results(results_dir):
    """
    Plot power spectra for feedback gain analysis.
    
    Args:
        results_dir (str): Path to the results directory containing config and data
        gain_type (str): Type of gain ('fb_gain' or 'input_gain_beta1' or 'input_gain_beta4')
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Load config file from results directory
    config_files = [f for f in os.listdir(results_dir) if f.endswith('.yaml')]
    if not config_files:
        print(f"Error: No yaml config file found in {results_dir}")
        sys.exit(1)
        
    config_path = os.path.join(results_dir, config_files[0])
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Get parameters from config
    fb_config = config['Power_spectra']['Feedback_gain']
    gamma_vals = fb_config['gamma_vals']
    c_vals = fb_config['c_vals']
    if fb_config['enabled']:
        gain_type = 'fb_gain'
    else:
        gain_type = 'input_gain_beta1'
    
    print(f"Processing data for contrasts: {c_vals}")
    print(f"With gamma values: {gamma_vals}")
    
    # Load power spectra data from the single file
    power_file = os.path.join(data_dir, f'power_spectra_{gain_type}_all.npy')
    if not os.path.exists(power_file):
        print(f"Error: Power spectra data file not found at {power_file}")
        sys.exit(1)
        
    print(f"Loading power spectra data from: {power_file}")
    power_data = np.load(power_file, allow_pickle=True).item()
    
    # Find normalization factor (maximum power at contrast=1)
    norm_factor = max(np.max(power_data[key]['power']) for key in power_data.keys())
    print(f"Global normalization factor (max power at c=1): {norm_factor}")
    
    # Normalize all power data by this single factor
    normalized_power_data = {}
    for key, data in power_data.items():
        normalized_power_data[key] = {
            'freq': data['freq'],
            'power': data['power'] / norm_factor
        }
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots','Power_spectra')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create plots for each contrast value
    for contrast in c_vals:
        print(f"Creating plot for contrast = {contrast}")
        
        # Create plots
        fig, axs = plot_power_spectra(
            normalized_power_data,
            gamma_vals,
            contrast,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False
        )
        
        # Save the figure
        Save_power_plots(plots_dir,fig,contrast,gain_type)
        
def Save_power_plots(plots_dir,fig,contrast,gain_type):
    save_path = os.path.join(plots_dir, f'power_spectra_{gain_type}_contrast_{contrast}.pdf')
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_power_spectra.py /path/to/results_dir_config_n")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir)