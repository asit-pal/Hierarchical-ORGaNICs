import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from Plotting import setup_plot_params # Import the setup function

# Set up common plot parameters
setup_plot_params()

# --- Script-specific setup ---

# Create color mapping that ensures gamma=1.0 always gets dark blue
def get_colors_for_gammas(gamma_values):
    """Get colors for gamma values, ensuring gamma=1.0 gets dark blue."""
    gamma_values = np.array(gamma_values)
    if len(gamma_values) == 1 and gamma_values[0] == 1.0:
        # If only gamma=1.0, return dark blue
        return [plt.cm.Blues(1.0)]
    else:
        # For multiple values, map them to color range
        min_val, max_val = gamma_values.min(), gamma_values.max()
        if max_val == min_val:
            positions = np.full_like(gamma_values, 0.5)
        else:
            positions = (gamma_values - min_val) / (max_val - min_val)
        positions = 0.3 + positions * 0.7  # Map to range [0.3, 1.0]
        return [plt.cm.Blues(pos) for pos in positions]

# Initial gamma values
gamma_values = np.array([0.5, 1.0])
colors = get_colors_for_gammas(gamma_values)

# Specific overrides for this script
plt.rcParams.update({
    "lines.linewidth": 10,
    "lines.markersize": 28,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
})

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
        
        gamma_vals = sorted(set(g for g, _ in Communication_data.keys()))
        c_vals = sorted(set(c for _, c in Communication_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating feedback gain plot for contrast = {contrast}")
            
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
            fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    
    # Check for input gain beta1 data
    input_gain_beta1_file = os.path.join(data_dir, 'Communication_data_input_gain_beta1_all.npy')
    if os.path.exists(input_gain_beta1_file):
        print(f"Loading input gain beta1 data from: {input_gain_beta1_file}")
        Communication_data = np.load(input_gain_beta1_file, allow_pickle=True).item()
        
        beta1_vals = sorted(set(b for b, _ in Communication_data.keys()))
        c_vals = sorted(set(c for _, c in Communication_data.keys()))
        print(f"\nBeta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating input gain beta1 plot for contrast = {contrast}")
            
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
            fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
        
def plot_pred_perf_vs_dim(performance_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Plot prediction performance versus dimensions with original color scheme.
    """
    fig, ax = plt.subplots()
    
    # Update colors based on actual gamma values being plotted
    colors = get_colors_for_gammas(gamma_vals)
    
    # Create marker styles for V1 and V4
    markers = {'V1': 's', 'V2': 'o'}
    line_styles = {'V1': (0, (5, 10)), 'V2': '-'}
    
    for gamma, color in zip(gamma_vals, colors):
        data = performance_data[gamma, contrast]
        
        # Retrieve V1 and V4 data
        V1_mean = data['V1']['mean']
        V1_std = data['V1']['std']
        V1_dims = data['V1']['dims']
        V4_mean = data['V4']['mean']
        V4_std = data['V4']['std']
        V4_dims = data['V4']['dims']
        
        # Calculate SEM
        sample_size = 200
        V1_sem = V1_std / np.sqrt(sample_size)
        V4_sem = V4_std / np.sqrt(sample_size)
        
        # Define the slice for dimensions
        dim_slice = slice(1, 6)
        
        # Fill between and plot V1 data
        ax.fill_between(V1_dims[dim_slice], V1_mean[dim_slice] - V1_sem[dim_slice], V1_mean[dim_slice] + V1_sem[dim_slice], 
                       color='gray', alpha=0.2)
        ax.plot(V1_dims[dim_slice], V1_mean[dim_slice], marker=markers['V1'], 
                linestyle='none',
                color=color, markeredgecolor='black')
        ax.plot(V1_dims[dim_slice], V1_mean[dim_slice], 
                linestyle=line_styles['V1'], linewidth=2,
                color=color)
        
        # Fill between and plot V4 data
        ax.fill_between(V4_dims[dim_slice], V4_mean[dim_slice] - V4_sem[dim_slice], V4_mean[dim_slice] + V4_sem[dim_slice], 
                       color='gray', alpha=0.2)
        ax.plot(V4_dims[dim_slice], V4_mean[dim_slice], marker=markers['V2'], 
                linestyle='none',
                color=color, markeredgecolor='black')
        ax.plot(V4_dims[dim_slice], V4_mean[dim_slice], 
                linestyle=line_styles['V2'], linewidth=2,
                color=color)
        
    # Legend for markers
    marker_legend = [
        mlines.Line2D([0], [0], color=colors[0], marker='o',
                     linestyle=line_styles['V2'], linewidth=2,
                     label='V1-V2'),
        mlines.Line2D([0], [0], color=colors[0], marker='s',
                     linestyle=line_styles['V1'], linewidth=2,
                     label='V1-V1')
    ]
    ax.legend(handles=marker_legend, loc='upper left', frameon=False)
    
    # Set labels and ticks
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Prediction Performance')
    
    # Set x-axis limits and ticks
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 3, 5])
    ax.set_xticks([2, 4], minor=True)
    
    # Set y-axis limits and ticks
    ax.set_ylim(0.04, 0.12)
    ax.set_yticks([0.04, 0.08, 0.12])
    ax.set_yticks([0.06, 0.10], minor=True)
    
    # Manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_communication_analysis.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_communication_analysis(results_dir)