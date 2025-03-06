import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Plotting import plot_coherence_data

def plot_feedback_gain_results(results_dir):
    """
    Plot coherence for feedback gain analysis.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load feedback gain data
    coherence_file = os.path.join(data_dir, 'coherence_fb_gain_all.npy')
    if os.path.exists(coherence_file):
        print(f"Loading feedback gain coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()
        
        # Extract gamma and contrast values from the data
        gamma_vals = sorted(set(g for g, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
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
            save_path = os.path.join(plots_dir, f'coherence_fb_gain_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    else:
        print(f"Warning: Coherence data file not found at {coherence_file}")
    
    # Load beta1 gain data
    coherence_file = os.path.join(data_dir, 'coherence_input_gain_beta1_all.npy')
    if os.path.exists(coherence_file):
        print(f"\nLoading beta1 gain coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()
        
        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(set(b for b, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"Beta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
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
            save_path = os.path.join(plots_dir, f'coherence_input_beta1_gain_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    else:
        print(f"Warning: Beta1 gain coherence data file not found at {coherence_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_coherence.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir)