import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from Plotting import plot_power_spectra_fixed_gamma

def power_spectra_fixed_gamma(results_dir):
    """
    Create power spectra plots for each gamma value, showing different contrasts.
    
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
    
    # Initialize power_data dictionary
    power_data = {}
    
    # For each contrast value
    for contrast in c_vals:
        data_file = f'power_fb_gain_contrast_{contrast}.npy'
        data_path = os.path.join(data_dir, data_file)
        
        if not os.path.exists(data_path):
            print(f"Warning: No data file found for contrast = {contrast}")
            continue
            
        # Load data for this contrast
        contrast_data = np.load(data_path, allow_pickle=True).item()
        
        # For each gamma value
        for gamma in gamma_vals:
            # Get the data for this (gamma, contrast) pair
            key = (gamma, contrast)
            if key in contrast_data:
                power_data[key] = contrast_data[key]
    
    if not power_data:
        print("Error: No data found to plot")
        sys.exit(1)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create a plot for each gamma value
    for gamma in gamma_vals:
        print(f"Creating plot for gamma = {gamma}")
        
        # Create the plot
        fig, ax = plot_power_spectra_fixed_gamma(
            power_data,
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
        print("Usage: python Plot_power_spectra_fixed_gamma.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    power_spectra_fixed_gamma(results_dir) 