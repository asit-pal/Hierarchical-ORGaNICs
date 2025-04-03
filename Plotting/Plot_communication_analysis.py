import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Plotting import plot_pred_perf_vs_dim
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# %matplotlib inline
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     "font.size": 36,  # Set a consistent font size for all text in the plot
# })
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_pred_perf_vs_dim(performance_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, line_width=5, line_labelsize=42, legendsize=42):
    """
    Plot prediction performance versus dimensions with original color scheme.
    """
    colors = ['#DC143C', '#00BFFF', '#32CD32', '#000000']  # Red, Blue, Green, Black
    fig, ax = plt.subplots(figsize=(14, 10))
    
    custom_handles = []
    
    # Create marker styles for V1 and V4
    markers = {'V1': 's', 'V4': 'o'}
    
    for gamma, color in zip(gamma_vals, colors):
        data = performance_data[gamma, contrast]
        
        # Retrieve V1 and V4 data
        V1_mean = data['V1']['mean']
        V1_std = data['V1']['std']
        V1_dims = data['V1']['dims']
        V4_mean = data['V4']['mean']
        V4_std = data['V4']['std']
        V4_dims = data['V4']['dims']
        
        # Fill between for V1 data
        ax.fill_between(V1_dims, V1_mean - 0.5*V1_std, V1_mean + 0.5*V1_std, 
                       color='gray', alpha=0.1)
        ax.plot(V1_dims, V1_mean, marker=markers['V1'], 
                linestyle='None', markersize=12,
                color=color, markeredgecolor='black')
        
        # Fill between for V4 data
        ax.fill_between(V4_dims, V4_mean - 0.5*V4_std, V4_mean + 0.5*V4_std, 
                       color='gray', alpha=0.1)
        ax.plot(V4_dims, V4_mean, marker=markers['V4'], 
                markersize=12, linestyle='None',
                color=color, markeredgecolor='black')
        
        # Determine label based on which gain is being varied
        if fb_gain:
            label = rf'$\gamma_1={gamma:.2f}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma:.2f}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma:.2f}$'
        
        custom_handles.append(mpatches.Patch(color=color, label=label))
    
    # First legend for gamma/beta values
    legend1 = ax.legend(handles=custom_handles, fontsize=legendsize,
                       loc='upper left', frameon=False,
                       handletextpad=0.1, labelspacing=0.15,
                       handlelength=1.0)
    ax.add_artist(legend1)
    
    # Second legend for markers
    marker_legend = [
        mlines.Line2D([0], [0], color='black', marker='o',
                     linestyle='None', markersize=10,
                     label=r'$\mathrm{V1\text{-}V2}$'),
        mlines.Line2D([0], [0], color='black', marker='s',
                     linestyle='None', markersize=10,
                     label=r'$\mathrm{V1\text{-}V1}$')
    ]
    legend2 = ax.legend(handles=marker_legend, loc='upper right',
                       bbox_to_anchor=(0.68, 1.0), fontsize=legendsize,
                       frameon=False, handletextpad=-0.2,
                       labelspacing=0.15)
    ax.add_artist(legend2)
    
    # Set labels and ticks
    ax.set_xlabel(r'$\mathrm{Dimensions}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{Prediction\;performance}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    
    # Set x-axis ticks to integers
    # x_min, x_max = ax.get_xlim()
    x_min, x_max = 0, 8
    ax.set_xlim(x_min, x_max)
    # x_ticks = range(int(x_min), int(x_max) + 1, 2)
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([rf'$\mathrm{{{x}}}$' for x in x_ticks])
    
    # Manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

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