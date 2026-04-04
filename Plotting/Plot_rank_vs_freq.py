import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params
setup_plot_params()

# Updated rcParams to match Plot_freq_wise_decom.py style
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.2,
    "legend.labelspacing": 0.1,
})

def plot_rank_vs_freq(rank_data, contrast_vals, gamma=None):
    """
    Plot numerical rank versus frequency for different contrast values.
    
    Args:
        rank_data: Dictionary with contrast as key and {'freq': array, 'rank': array} as value
        contrast_vals: List of contrast values to plot
        gamma: Gamma value for labeling (optional)
    
    Returns:
        fig, ax: Figure and axes objects
    """
    fig, ax = plt.subplots()
    
    # Dynamically create colors based on the number of contrast values
    contrast_values = np.array(contrast_vals)
    # Normalize contrast values to [0,1] range for colormap, handling the case of a single contrast
    if contrast_values.size > 1:
        positions = (contrast_values - contrast_values.min()) / (contrast_values.max() - contrast_values.min())
    else:
        positions = np.array([0.5])  # A default position if there's only one value
    
    # Adjust the range of the colormap (0.2 to 0.8 for reds)
    positions = 0.2 + positions * 0.6
    
    cmap = matplotlib.colormaps.get_cmap('Reds')
    colors = [cmap(pos) for pos in positions]
    ax.set_prop_cycle(color=colors)
    
    custom_handles = []
    
    for i, contrast in enumerate(contrast_vals):
        if contrast not in rank_data:
            print(f"Warning: Data for contrast={contrast} not found in rank_data. Skipping.")
            continue
            
        data = rank_data[contrast]
        freq = data['freq']
        rank = data['rank']
        
        # Plot rank vs frequency
        ax.plot(freq, rank, '-', linewidth=2, alpha=0.8)
        
        # Create handle for legend
        label = f'c = {contrast:.3f}'
        custom_handles.append(mpatches.Patch(color=colors[i], label=label))
    
    # Set labels and ticks
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Numerical Rank')
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)  # Minor ticks every 10 Hz
    ax.set_xticklabels([str(x) for x in xticks])

    # Add tick padding
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Add legend
    ax.legend(handles=custom_handles, loc='best', frameon=False,
              fontsize=plt.rcParams['legend.fontsize'])

    return fig, ax


def plot_rank_vs_freq_with_colorbar(rank_data, contrast_vals, gamma=None):
    """
    Plot numerical rank versus frequency with a colorbar for contrast values.
    
    Args:
        rank_data: Dictionary with contrast as key and {'freq': array, 'rank': array} as value
        contrast_vals: List of contrast values to plot
        gamma: Gamma value for labeling (optional)
    
    Returns:
        fig, ax: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dynamically create colors based on the number of contrast values
    contrast_values = np.array(contrast_vals)
    
    # Normalize contrast values to [0,1] range for colormap
    if contrast_values.size > 1:
        norm = matplotlib.colors.LogNorm(vmin=contrast_values.min(), vmax=contrast_values.max())
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    cmap = matplotlib.colormaps.get_cmap('viridis')
    
    for i, contrast in enumerate(contrast_vals):
        if contrast not in rank_data:
            print(f"Warning: Data for contrast={contrast} not found in rank_data. Skipping.")
            continue
            
        data = rank_data[contrast]
        freq = data['freq']
        rank = data['rank']
        
        # Get color from colormap
        color = cmap(norm(contrast))
        
        # Plot rank vs frequency
        ax.plot(freq, rank, '-', color=color, linewidth=2, alpha=0.8)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Contrast')
    
    # Set labels and ticks
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Numerical Rank', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)  # Minor ticks every 10 Hz
    ax.set_xticklabels([str(x) for x in xticks])

    # Add tick padding
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if gamma is not None:
        ax.set_title(f'Cross-Spectral Matrix Rank vs Frequency (γ={gamma})', 
                    fontsize=14, fontweight='bold')

    return fig, ax


def plot_rank_frequency_analysis(results_dir):
    """
    Plot rank vs. frequency analysis results.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    data_dir = os.path.join(results_dir, 'Data')
    plots_dir = os.path.join(results_dir, 'Plots', 'RankFrequency')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results directory for Rank vs. Freq plots: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plots will be saved to: {plots_dir}")

    data_file = os.path.join(data_dir, 'rank_vs_freq_data.npy')
        
    if os.path.exists(data_file):
        # Load the rank vs frequency data
        rank_data = np.load(data_file, allow_pickle=True).item()
        
        # Extract contrast values from the data
        contrast_vals = sorted(list(rank_data.keys()))
        
        print(f"Contrast values found: {contrast_vals}")
        print(f"Number of frequency points: {len(rank_data[contrast_vals[0]]['freq'])}")
        
        # Plot with standard legend
        fig, ax = plot_rank_vs_freq(
            rank_data=rank_data,
            contrast_vals=contrast_vals,
            gamma=None
        )
        
        save_path = os.path.join(plots_dir, 'Rank_vs_Freq.pdf')
        fig.savefig(save_path, dpi=400, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")
        
        # Also plot with colorbar version
        fig, ax = plot_rank_vs_freq_with_colorbar(
            rank_data=rank_data,
            contrast_vals=contrast_vals,
            gamma=None
        )
        
        save_path = os.path.join(plots_dir, 'Rank_vs_Freq_colorbar.pdf')
        fig.savefig(save_path, dpi=400, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")
        
    else:
        print(f"Data file not found: {data_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_rank_vs_freq.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir_arg = sys.argv[1]
    print(f"Processing Rank vs. Frequency results from: {results_dir_arg}")
    plot_rank_frequency_analysis(results_dir_arg)

