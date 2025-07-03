import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params

# Define the contrast values explicitly
CONTRAST_VALS = np.array([0.032,0.064,0.125,0.25,0.5,1.0])

# Create color mapping function
def get_colors_for_contrasts(contrast_values):
    """Get colors for contrast values using Reds colormap."""
    if len(contrast_values) == 1:
        # If only one value, use a mid-range red
        return [plt.cm.Reds(0.7)]
    else:
        # Map multiple values to color range [0.3, 1.0]
        min_val, max_val = contrast_values.min(), contrast_values.max()
        if max_val == min_val: # Avoid division by zero if all values are the same
            positions = np.full_like(contrast_values, 0.5) 
        else:
            positions = (contrast_values - min_val) / (max_val - min_val)
        positions = 0.3 + positions * 0.7  # Map to range [0.3, 1.0]
        return [plt.cm.Reds(pos) for pos in positions]

# Set up plotting parameters for journal-quality figures
setup_plot_params()
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
})

def get_max_performance(performance_data, contrast_vals, gain):
    """
    Extracts the maximum prediction performance for V1-V1 and V1-V2 
    across dimensions for each contrast value.
    """
    max_perf_v1v1 = []
    max_perf_v1v2 = []

    for contrast in contrast_vals:
        if (gain, contrast) in performance_data:
            data = performance_data[(gain, contrast)]
            # Find max performance (excluding dim 0 if present)
            v1_mean = data['V1_V1']['mean']
            v2_mean = data['V1_V4']['mean'] 
            
            # Assuming dimensions start from 1 in the mean array
            max_perf_v1v1.append(np.max(v1_mean) if len(v1_mean)>0 else np.nan)
            max_perf_v1v2.append(np.max(v2_mean) if len(v2_mean)>0 else np.nan)
        else:
            max_perf_v1v1.append(np.nan)
            max_perf_v1v2.append(np.nan)
            
    return np.array(max_perf_v1v1), np.array(max_perf_v1v2)

def plot_max_perf_vs_contrast(performance_data, contrast_vals, gain):
    """
    Plots maximum prediction performance versus contrast values for a given baseline gain.
    
    Args:
        performance_data (dict): Loaded communication data.
        contrast_vals (np.array): Array of contrast values.
        gain (float): The baseline gain value to plot (e.g., 1.0).
    """
    fig, ax = plt.subplots()
    
    max_v1v1, max_v1v2 = get_max_performance(performance_data, contrast_vals, gain)
    colors = get_colors_for_contrasts(contrast_vals)
    contrast_percentages = contrast_vals * 100
    
    # Create marker styles
    markers = {'V1': 's', 'V2': 'o'} # V1-V1: square, V1-V2: circle
    
    # Plot line segments and markers with corresponding colors
    for i in range(len(contrast_vals) - 1):
        ax.plot(contrast_percentages[i:i+2], max_v1v1[i:i+2], linestyle='--', color=colors[i])
        ax.plot(contrast_percentages[i:i+2], max_v1v2[i:i+2], linestyle='--', color=colors[i])

    # Plot markers on top
    for i in range(len(contrast_vals)):
        ax.plot(contrast_percentages[i], max_v1v1[i], marker=markers['V1'], color=colors[i], 
                markeredgecolor='black', linestyle='none', markersize=30)
        ax.plot(contrast_percentages[i], max_v1v2[i], marker=markers['V2'], color=colors[i], 
                markeredgecolor='black', linestyle='none', markersize=30)

    # Legend for markers
    legend_color = colors[-1] 
    marker_legend = [
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V2'],
                     linestyle='--', linewidth=4, markersize=30,
                     label='V1-V2'),
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V1'],
                     linestyle='--', linewidth=4, markersize=30,
                     label='V1-V1')
    ]
    ax.legend(handles=marker_legend, loc='best', frameon=False)

    # Set labels and ticks
    ax.set_xlabel('Contrast (%)')
    ax.set_xlim(min(contrast_percentages) * 0.8, max(contrast_percentages) * 1.1)
    ax.set_ylabel('Max Pred. Perf.')
    ax.set_xscale('log')
    ax.set_xticks(contrast_percentages)
    ax.xaxis.set_minor_locator(plt.NullLocator()) # Turn off minor ticks on x-axis
    
    ax.set_xticklabels([f'{x:.1f}' for x in contrast_percentages])

    # Y-axis settings
    ax.set_ylim(0.04,0.8)
    ax.set_yscale('log')

    yticks = np.array([ 0.1,0.4])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])
    
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)


    return fig, ax

def main():
    if len(sys.argv) != 2:
        print("Usage: python Plot_max_perf_vs_contrast.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    data_dir = os.path.join(results_dir, 'Data')
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication_MaxPerf')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plots will be saved in: {plots_dir}")

    # --- Process Feedback Gain Data ---
    fb_gain_file = os.path.join(data_dir, 'Communication_data_fb_gain_all.npy')
    if os.path.exists(fb_gain_file):
        print(f"\nProcessing Feedback Gain data from: {fb_gain_file}")
        comm_data_fb = np.load(fb_gain_file, allow_pickle=True).item()
        
        # Use baseline feedback gain = 1.0
        baseline_gain = sorted(list(set(g for g, _ in comm_data_fb.keys())))[0]
        print(f"Creating Contrast plot for baseline feedback gain = {baseline_gain}")
        fig, ax = plot_max_perf_vs_contrast(
                comm_data_fb,
                CONTRAST_VALS,
                baseline_gain
        )
        if fig:
            save_path = os.path.join(plots_dir, f'MaxPerf_vs_Contrast_fb_gain_baseline.pdf')
            fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            fig.savefig(save_path, dpi=400, format='pdf')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    else:
        print(f"Feedback gain data file not found: {fb_gain_file}")

if __name__ == "__main__":
    main() 