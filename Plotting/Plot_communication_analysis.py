import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Plotting import plot_pred_perf_vs_dim
# %matplotlib inline
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     "font.size": 36,  # Set a consistent font size for all text in the plot
# })
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



def plot_communication_analysis(results_dir):
    """
    Plot communication analysis results for both feedback gain and input gain beta1.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check for feedback gain data
    fb_gain_file = os.path.join(data_dir, 'Communication_data_fb_gain_all.npy')
    if os.path.exists(fb_gain_file):
        print(f"Loading feedback gain data from: {fb_gain_file}")
        Communication_data = np.load(fb_gain_file, allow_pickle=True).item()
        
        # Extract gamma and contrast values from the data
        gamma_vals = sorted(set(g for g, _ in Communication_data.keys()))
        c_vals = sorted(set(c for _, c in Communication_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating feedback gain plot for contrast = {contrast}")
            
            # Create plots
            fig, axs = plot_pred_perf_vs_dim(
                Communication_data,
                gamma_vals,
                contrast,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False
            )
            
            # Save the figure
            save_path = os.path.join(plots_dir, f'Communication_analysis_fb_gain_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    
    # Check for input gain beta1 data
    input_gain_beta1_file = os.path.join(data_dir, 'Communication_data_input_gain_beta1_all.npy')
    if os.path.exists(input_gain_beta1_file):
        print(f"Loading input gain beta1 data from: {input_gain_beta1_file}")
        Communication_data = np.load(input_gain_beta1_file, allow_pickle=True).item()
        
        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(set(b for b, _ in Communication_data.keys()))
        c_vals = sorted(set(c for _, c in Communication_data.keys()))
        print(f"\nBeta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating input gain beta1 plot for contrast = {contrast}")
            
            # Create plots
            fig, axs = plot_pred_perf_vs_dim(
                Communication_data,
                beta1_vals,
                contrast,
                fb_gain=False,
                input_gain_beta1=True,
                input_gain_beta4=False
            )
            
            # Save the figure
            save_path = os.path.join(plots_dir, f'Communication_analysis_input_gain_beta1_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_communication_analysis.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_communication_analysis(results_dir)