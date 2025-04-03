import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def plot_pred_perf_vs_dim_3(performance_data, pairs, contrast, labelsize=42, legendsize=42):
    """
    Plot prediction performance vs dimensions for different parameter pairs.
    
    Args:
        performance_data: Dictionary containing performance metrics
        pairs: List of (g, gamma4, gamma5) tuples
        contrast: Contrast value
        labelsize: Size of axis labels
        legendsize: Size of legend text
    """
    colors = ['#DC143C', '#00BFFF', '#32CD32', '#000000']  # Red, Blue, Green, Black
    fig, ax = plt.subplots(figsize=(14, 10))
    
    custom_handles = []
    
    # Create marker styles for V4 and V5
    markers = {'V4': 'o', 'V5': 's'}
    
    for i, (gamma4, gamma5) in enumerate(pairs):
        # Access data assuming the key is (gamma4, gamma5, contrast)
        data = performance_data[(gamma4, gamma5, contrast)]
        
        # Plot V4 data
        ax.fill_between(data['V4']['dims'], 
                       data['V4']['mean'] - 0.5*data['V4']['std'], 
                       data['V4']['mean'] + 0.5*data['V4']['std'], 
                       color='gray', alpha=0.1)
        ax.plot(data['V4']['dims'], data['V4']['mean'], 
                marker=markers['V4'], markeredgecolor='black',
                linestyle='None', markersize=12, color=colors[i])
        
        # Plot V5 data
        ax.fill_between(data['V5']['dims'], 
                       data['V5']['mean'] - 0.5*data['V5']['std'], 
                       data['V5']['mean'] + 0.5*data['V5']['std'], 
                       color='gray', alpha=0.1)
        ax.plot(data['V5']['dims'], data['V5']['mean'], 
                marker=markers['V5'], markeredgecolor='black',
                linestyle='None', markersize=12, color=colors[i])
        
        # Create label for the parameter combination, excluding 'g'
        label = rf'$\gamma_4={gamma4:.2f},\gamma_5={gamma5:.2f}$'
        custom_handles.append(mpatches.Patch(color=colors[i], label=label))
    
    # First legend for parameter combinations
    legend1 = ax.legend(handles=custom_handles, fontsize=legendsize,
                       loc='upper left', frameon=False,
                       handletextpad=0.1, labelspacing=0.15,
                       handlelength=1.0)
    ax.add_artist(legend1)
    
    # Second legend for markers
    marker_legend = [
        mlines.Line2D([0], [0], color='black', marker='o',
                     linestyle='None', markersize=10,
                     label=r'$\mathrm{V1\text{-}V4}$'),
        mlines.Line2D([0], [0], color='black', marker='s',
                     linestyle='None', markersize=10,
                     label=r'$\mathrm{V1\text{-}V5}$')
    ]
    legend2 = ax.legend(handles=marker_legend, loc='upper right',
                       bbox_to_anchor=(0.80, 1.0), fontsize=legendsize,
                       frameon=False, handletextpad=-0.2,
                       labelspacing=0.15)
    ax.add_artist(legend2)
    
    # Set labels and ticks
    ax.set_xlabel(r'$\mathrm{Dimensions}$', fontsize=labelsize)
    ax.set_ylabel(r'$\mathrm{Prediction\;performance}$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    # Set x-axis limits and ticks
    x_min, x_max = 0, 8
    ax.set_xlim(x_min, x_max)
    
    # Manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

def plot_communication_analysis_3(results_dir):
    """
    Plot three-area communication analysis results.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication_3')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check for communication data
    comm_data_file = os.path.join(data_dir, 'Pred_perf_vs_dim_data.npz')
    if os.path.exists(comm_data_file):
        print(f"Loading communication data from: {comm_data_file}")
        data = np.load(comm_data_file, allow_pickle=True)
        
        # Extract data
        performance_data = data['performance_data'].item()
        pairs = data['pairs']
        contrast = data['contrast'].item()
        
        print(f"\nPairs found in data: {pairs}")
        print(f"Contrast value: {contrast}")
        
        # Create plot
        print("Creating communication analysis plot")
        fig, ax = plot_pred_perf_vs_dim_3(
            performance_data,
            pairs,
            contrast,
            labelsize=42,
            legendsize=42
        )
        
        # Save the figure
        save_path = os.path.join(plots_dir, f'Communication_analysis_3_contrast_{contrast}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")
    else:
        print(f"Warning: Data file not found at {comm_data_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_communication_analysis_3.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_communication_analysis_3(results_dir) 