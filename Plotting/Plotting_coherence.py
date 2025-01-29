import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_coherence_data

def plot_feedback_gain_results(results_dir):
    """
    Plot coherence for feedback gain analysis.
    
    Args:
        results_dir (str): Path to the results directory containing config and data
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
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot for each enabled gain type
    if config['Coherence']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        gamma_vals = config['Coherence']['Feedback_gain']['gamma_vals']
        c_vals = config['Coherence']['Feedback_gain']['c_vals']
        
        print(f"Processing feedback gain data for contrasts: {c_vals}")
        print(f"With gamma values: {gamma_vals}")
        
        coherence_file = os.path.join(data_dir, f'coherence_{gain_type}_all.npy')
        if os.path.exists(coherence_file):
            print(f"Loading feedback gain coherence data from: {coherence_file}")
            coherence_data = np.load(coherence_file, allow_pickle=True).item()
            
            # Create plots for each contrast value
            for contrast in c_vals:
                print(f"Creating feedback gain plot for contrast = {contrast}")
                
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
        else:
            print(f"Warning: Coherence data file not found at {coherence_file}")
    
    if config['Coherence']['Input_gain_beta1']['enabled']:
        gain_type = 'input_gain_beta1'
        beta1_vals = config['Coherence']['Input_gain_beta1']['beta1_vals']
        c_vals = config['Coherence']['Input_gain_beta1']['c_vals']
        
        print(f"Processing beta1 gain data for contrasts: {c_vals}")
        print(f"With beta1 values: {beta1_vals}")
        
        coherence_file = os.path.join(data_dir, f'coherence_{gain_type}_all.npy')
        if os.path.exists(coherence_file):
            print(f"Loading beta1 gain coherence data from: {coherence_file}")
            coherence_data = np.load(coherence_file, allow_pickle=True).item()
            
            # Create plots for each contrast value
            for contrast in c_vals:
                print(f"Creating beta1 gain plot for contrast = {contrast}")
                
                # Create plots
                fig, ax = plot_coherence_data(
                    coherence_data,
                    beta1_vals,
                    contrast,
                    fb_gain=False,
                    input_gain_beta1=True,
                    input_gain_beta4=False
                )
                
                # Save the figure
                save_coherence_plots(plots_dir, fig, contrast, gain_type)
        else:
            print(f"Warning: Coherence data file not found at {coherence_file}")

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
    plot_feedback_gain_results(results_dir)