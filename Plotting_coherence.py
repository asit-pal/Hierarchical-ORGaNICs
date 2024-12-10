import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_coherence_data

def plot_feedback_gain_results(results_dir,gain_type):
    """
    Plot coherence for feedback gain analysis.
    
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
    fb_config = config['Coherence']['Feedback_gain']
    gamma_vals = fb_config['gamma_vals']
    c_vals = fb_config['c_vals']
    
    print(f"Processing data for contrasts: {c_vals}")
    print(f"With gamma values: {gamma_vals}")
    
    # Load coherence data from the single file
    coherence_file = os.path.join(data_dir, f'coherence_{gain_type}_all.npy')
    if not os.path.exists(coherence_file):
        print(f"Error: Coherence data file not found at {coherence_file}")
        sys.exit(1)
        
    print(f"Loading coherence data from: {coherence_file}")
    coherence_data = np.load(coherence_file, allow_pickle=True).item()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots','Coherence')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create plots for each contrast value
    for contrast in c_vals:
        print(f"Creating plot for contrast = {contrast}")
        
        # Create plots
        fig, ax = plot_coherence_data(
            coherence_data,
            gamma_vals,
            contrast,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False
        )
        
        # Save the figure
        save_coherence_plots(plots_dir, fig, contrast, gain_type)

def save_coherence_plots(plots_dir, fig, contrast, gain_type):
    """
    Save coherence plots to file.
    """
    save_path = os.path.join(plots_dir, f'coherence_{gain_type}_contrast_{contrast}.pdf')
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_coherence.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir,'fb_gain') # change this for input gain