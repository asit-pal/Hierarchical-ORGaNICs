import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_steady_states

def plot_gain_modulation_results(results_dir):
    """
    Plot steady states for gain modulation analysis.
    
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
    plots_dir = os.path.join(results_dir, 'Plots', 'Gain_modulation')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot for each enabled gain type
    if config['Gain_modulation']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        gamma_vals = config['Gain_modulation']['Feedback_gain']['gamma_vals']
        
        data_file = os.path.join(data_dir, f'steady_states_{gain_type}_all.npy')
        if os.path.exists(data_file):
            print(f"Loading feedback gain data from: {data_file}")
            steady_states = np.load(data_file, allow_pickle=True).item()
            
            fig, axs = plot_steady_states(
                steady_states=steady_states,
                c_vals=np.logspace(np.log10(1e-2), np.log10(1), 40),
                gamma_vals=gamma_vals,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False,
                index_y1=1, # index for y1Plus
                index_y4=3   # index for y4Plus
            )
            save_steady_states_plot(plots_dir, fig, gain_type)
    
    if config['Gain_modulation']['Input_gain_beta1']['enabled']:
        gain_type = 'input_beta1_gain'
        beta1_vals = config['Gain_modulation']['Input_gain_beta1']['beta1_vals']
        
        data_file = os.path.join(data_dir, f'steady_states_{gain_type}_all.npy')
        if os.path.exists(data_file):
            print(f"Loading beta1 gain data from: {data_file}")
            steady_states = np.load(data_file, allow_pickle=True).item()
            
            fig, axs = plot_steady_states(
                steady_states=steady_states,
                c_vals=np.logspace(np.log10(1e-2), np.log10(1), 40),
                gamma_vals=beta1_vals,
                fb_gain=False,
                input_gain_beta1=True,
                input_gain_beta4=False,
                index_y1=1, # index for y1Plus
                index_y4=3   # index for y4Plus
            )
            save_steady_states_plot(plots_dir, fig, gain_type)

def save_steady_states_plot(plots_dir, fig, gain_type):
    """Save the steady states plot to file."""
    save_path = os.path.join(plots_dir, f'steady_states_{gain_type}.pdf')
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_gain_modulation.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_gain_modulation_results(results_dir)