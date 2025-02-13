import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from Plotting import plot_power_spectra_fixed_gamma

def plot_fixed_gamma_power_spectra(results_dir):
    """
    Create power spectra plots for each gamma value, showing different contrasts.
    All power spectra are normalized by the maximum power at contrast=1 for each gamma.
    
    Args:
        results_dir (str): Path to the results directory containing config and data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Load config file
    config_files = [f for f in os.listdir(results_dir) if f.endswith('.yaml')]
    if not config_files:
        print(f"Error: No yaml config file found in {results_dir}")
        sys.exit(1)
        
    config_path = os.path.join(results_dir, config_files[0])
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    fb_config = config['Power_spectra']['Feedback_gain']
    gamma_vals = fb_config['gamma_vals']
    c_vals = fb_config['c_vals']
    
    # Load power spectra data from the single file
    power_file = os.path.join(data_dir, 'power_spectra_fb_gain_V2.npy') #power_spectra_fb_gain_all.npy
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
    plots_dir = os.path.join(results_dir, 'Plots', 'Power_spectra')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create a plot for each gamma value
    for gamma in gamma_vals:
        print(f"Creating plot for gamma = {gamma}")
        
        # Create the plot
        fig, ax = plot_power_spectra_fixed_gamma(
            normalized_power_data,
            c_vals,
            gamma,
            line_width=5,
            line_labelsize=42,
            legendsize=42
        )
        
        # Save the plot
        save_path = os.path.join(plots_dir, f'power_spectra_gamma_{gamma}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_power_spectra_fixed_gamma.py /path/to/results_dir_config_n")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_fixed_gamma_power_spectra(results_dir) 