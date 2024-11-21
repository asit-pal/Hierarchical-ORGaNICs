import numpy as np
import matplotlib.pyplot as plt
import os
from Plotting import plot_pred_perf_vs_dim, plot_dimension_vs_freq
# %matplotlib inline
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 36,  # Set a consistent font size for all text in the plot
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



def plot_feedback_gain_results(results_dir):
    data_dir = os.path.join(results_dir, 'Data', 'Communication_data')
    fb_files = [f for f in os.listdir(data_dir) if f.startswith('Communication_fb_gain')]
    
    for file in fb_files:
        # Load data
        data_path = os.path.join(data_dir, file)
        Communication_data = np.load(data_path, allow_pickle=True).item()
        
        # Extract gamma values and contrast from the data
        gamma_vals = sorted(list(set([key[0] for key in Communication_data.keys()])))
        contrast = list(set([key[1] for key in Communication_data.keys()]))[0]  # Should be same for all
        
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
        save_path = os.path.join(results_dir, 'Plots', f'pred_perf_fb_gain_contrast_{contrast}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

def plot_input_gain_beta1_results(results_dir):
    data_dir = os.path.join(results_dir, 'Data', 'Communication_data')
    beta1_files = [f for f in os.listdir(data_dir) if f.startswith('Communication_input_gain_beta1')]
    
    for file in beta1_files:
        # Load data
        data_path = os.path.join(data_dir, file)
        Communication_data = np.load(data_path, allow_pickle=True).item()
        
        # Extract beta1 values and contrast from the data
        beta1_vals = sorted(list(set([key[0] for key in Communication_data.keys()])))
        contrast = list(set([key[1] for key in Communication_data.keys()]))[0]  # Should be same for all
        
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
        save_path = os.path.join(results_dir, 'Plots', f'pred_perf_input_gain_beta1_contrast_{contrast}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

def main():
    results_dir = 'Results/config3'
    
    # Plot feedback gain results if they exist
    plot_feedback_gain_results(results_dir)
    
    # Plot input gain beta1 results if they exist
    # plot_input_gain_beta1_results(results_dir)

if __name__ == "__main__":
    main()