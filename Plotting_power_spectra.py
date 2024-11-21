import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_power_spectra 

def plot_feedback_gain_results(results_dir):
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
        
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
    
    print(f"Processing data for contrasts: {c_vals}")
    print(f"With gamma values: {gamma_vals}")
    
    # Load and plot data for each contrast value
    for contrast in c_vals:
        data_file = f'power_fb_gain_contrast_{contrast}.npy'
        data_path = os.path.join(data_dir, data_file)
        
        print(f"Looking for data file: {data_path}")
        if os.path.exists(data_path):
            print(f"Processing contrast = {contrast}")
            Power_data = np.load(data_path, allow_pickle=True).item()
            
            # Create plots
            fig, axs = plot_power_spectra(
                Power_data,
                gamma_vals,
                contrast,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False
            )
            
            # Save the figure
            plots_dir = os.path.join(results_dir, 'Plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            save_path = os.path.join(plots_dir, f'power_spectra_fb_gain_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
        else:
            print(f"Warning: No data file found for contrast = {contrast}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_power_spectra.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir)
