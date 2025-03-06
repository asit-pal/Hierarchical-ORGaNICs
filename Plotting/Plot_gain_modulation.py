import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Plotting import plot_steady_states

def plot_gain_modulation_results(results_dir):
    """
    Plot steady states for gain modulation analysis.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Gain_modulation')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot for each enabled gain type
    # Load feedback gain data
    data_file = os.path.join(data_dir, 'steady_states_fb_gain_all.npy')
    if os.path.exists(data_file):
        print(f"Loading feedback gain data from: {data_file}")
        steady_states = np.load(data_file, allow_pickle=True).item()
        
        # Extract gamma and contrast values from the data
        gamma_vals = sorted(list(set([g for g, _ in steady_states.keys()])))
        c_vals = sorted(list(set([c for _, c in steady_states.keys()])))
        print(f"Found gamma values: {gamma_vals}")
        print(f"Found contrast values: {c_vals}")
        
        fig, axs = plot_steady_states(
            steady_states=steady_states,
            c_vals=c_vals,
            gamma_vals=gamma_vals,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            index_y1=1, # index for y1Plus
            index_y4=3  # index for y4Plus
        )
        save_steady_states_plot(plots_dir, fig, 'fb_gain')
    
    # Load beta1 gain data
    data_file = os.path.join(data_dir, 'steady_states_input_gain_beta1_all.npy')
    if os.path.exists(data_file):
        print(f"Loading beta1 gain data from: {data_file}")
        steady_states = np.load(data_file, allow_pickle=True).item()
        
        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(list(set([b for b, _ in steady_states.keys()])))
        c_vals = sorted(list(set([c for _, c in steady_states.keys()])))
        print(f"Found beta1 values: {beta1_vals}")
        print(f"Found contrast values: {c_vals}")
        
        fig, axs = plot_steady_states(
            steady_states=steady_states,
            c_vals=c_vals,
            gamma_vals=beta1_vals,  # Use beta1_vals instead of gamma_vals
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            index_y1=1, # index for y1Plus
            index_y4=3  # index for y4Plus
        )
        save_steady_states_plot(plots_dir, fig, 'input_gain_beta1')

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