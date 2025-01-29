import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from Plotting import plot_coherence_data_fixed_gamma

def plot_fixed_gamma_coherence(results_dir):
    """
    Create coherence plots for each gamma value, showing different contrasts.
    
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
    fb_config = config['Coherence']['Feedback_gain']
    gamma_vals = fb_config['gamma_vals']
    c_vals = fb_config['c_vals']
    
    # Load coherence data from the single file
    coherence_file = os.path.join(data_dir, 'coherence_fb_gain_all.npy')
    if not os.path.exists(coherence_file):
        print(f"Error: Coherence data file not found at {coherence_file}")
        sys.exit(1)
        
    print(f"Loading coherence data from: {coherence_file}")
    coherence_data = np.load(coherence_file, allow_pickle=True).item()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a plot for each gamma value
    for gamma in gamma_vals:
        print(f"Creating plot for gamma = {gamma}")
        
        # Create the plot
        fig, ax = plot_coherence_data_fixed_gamma(
            coherence_data,
            c_vals,
            gamma,
            line_width=5,
            line_labelsize=42,
            legendsize=42
        )
        
        # Save the plot
        save_path = os.path.join(plots_dir, f'coherence__fixed_gamma_{gamma}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_coherence_fixed_gamma.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_fixed_gamma_coherence(results_dir) 