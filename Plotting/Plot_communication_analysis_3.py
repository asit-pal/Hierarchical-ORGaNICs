import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Keep the existing colors for parameter pairs
colors = ['#DC143C', '#00BFFF', '#32CD32', '#000000']  # Red, Blue, Green, Black

# Set up plotting parameters for journal-quality figures (matching reference)
from Plotting import setup_plot_params # Import the setup function

# Set up common plot parameters
setup_plot_params()

# Script-specific overrides to match Plot_communication_analysis.py
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 35,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3, 
    "legend.labelspacing": 0.2,
})

def plot_pred_perf_vs_dim_3(performance_data, gamma4, gamma5, contrast):
    """
    Plot prediction performance vs dimensions for a SINGLE parameter pair.
    V1-V4 is blue, V1-V5 is red.
    (Style matched to Plot_communication_analysis.py)
    
    Args:
        performance_data: Dictionary containing performance metrics
        gamma4: Single gamma4 value
        gamma5: Single gamma5 value
        contrast: Contrast value
    """
    fig, ax = plt.subplots() # Create a new figure for this pair
    
    # Define specific colors
    color_v4 = '#00BFFF'  # Blue for V1-V4
    color_v5 = '#DC143C'  # Red for V1-V5
    
    # Marker styles (V4 = V1-V4 prediction, V5 = V1-V5 prediction)
    markers = {'V4': 's', 'V5': 'o'}
    
    # Access data for the specific pair
    key = (gamma4, gamma5, contrast)
    if key not in performance_data:
        print(f"Warning: Data not found for key {key}")
        plt.close(fig) # Close the empty figure
        return None, None # Return None if data is missing

    data = performance_data[key]
    
    # Calculate sample size for SEM
    sample_size = 200
        
    # --- V4 Data (Plot in Blue) ---
    V4_dims = data['V4']['dims']
    V4_mean = data['V4']['mean']
    V4_std = data['V4']['std']
    V4_sem = V4_std / np.sqrt(sample_size)  # Calculate SEM
    
    # Apply slicing [1:7] to match Plot_communication_analysis.py
    dims_slice = V4_dims[1:6]
    mean_slice = V4_mean[1:6]
    sem_slice = V4_sem[1:6]

    # Fill between using SEM
    ax.fill_between(dims_slice, 
                    mean_slice - sem_slice, 
                    mean_slice + sem_slice, 
                    color='gray', alpha=0.1)
    # Plot markers (using sliced data)
    ax.plot(dims_slice, mean_slice, 
            marker=markers['V4'], markeredgecolor='black',
            linestyle='None', # No line for markers
            color=color_v4) # Use blue color
    # Plot dashed line (using sliced data)
    ax.plot(dims_slice, mean_slice, 
            linestyle='--',  # Match reference line style
            color=color_v4) # Use blue color

    # --- V5 Data (Plot in Red) ---
    V5_dims = data['V5']['dims']
    V5_mean = data['V5']['mean']
    V5_std = data['V5']['std']
    V5_sem = V5_std / np.sqrt(sample_size)  # Calculate SEM
    
    # Apply slicing [1:7] to match Plot_communication_analysis.py
    dims_slice = V5_dims[1:6]
    mean_slice = V5_mean[1:6]
    sem_slice = V5_sem[1:6]
    
    # Fill between using SEM
    ax.fill_between(dims_slice, 
                    mean_slice - sem_slice, 
                    mean_slice + sem_slice, 
                    color='gray', alpha=0.1)
    # Plot markers (using sliced data)
    ax.plot(dims_slice, mean_slice, 
            marker=markers['V5'], markeredgecolor='black',
            linestyle='None', # No line for markers
            color=color_v5) # Use red color
    # Plot dashed line (using sliced data)
    ax.plot(dims_slice, mean_slice, 
            linestyle='--', # Match reference line style
            color=color_v5) # Use red color
            
    # Legend for markers (uses the specific colors)
    marker_legend = [
        mlines.Line2D([0], [0], color=color_v4, marker=markers['V4'], # Blue, Square
                     linestyle='--',
                     label='V1-V4'),
        mlines.Line2D([0], [0], color=color_v5, marker=markers['V5'], # Red, Circle
                     linestyle='--',
                     label='V1-V5')
    ]
    # Position and style matching reference/rcParams
    legend2 = ax.legend(handles=marker_legend, loc='best',
                       fontsize=plt.rcParams['legend.fontsize'],
                       frameon=False, handletextpad=0.3,
                       labelspacing=0.2)

    # Set labels and ticks
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Prediction Performance')
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['xtick.labelsize'])
    
    # Set x-axis limits and ticks to match Plot_communication_analysis.py
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(np.array([1, 3, 5]))
    ax.set_xticks(np.array([2, 4]), minor=True)
    
    # Set y-axis limits and ticks to match Plot_communication_analysis.py
    ax.set_ylim(0.00, 0.12)
    ax.set_yticks(np.array([0.00, 0.06, 0.12]))
    ax.set_yticks(np.array([0.03, 0.09]), minor=True)
    
    return fig, ax

def plot_communication_analysis_3(results_dir):
    """
    Plot three-area communication analysis results, creating a separate
    figure for each parameter pair.
    
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
        
        # --- Loop through each pair and create a plot ---
        for i, (gamma4, gamma5) in enumerate(pairs): # Keep enumerate for potential future use, but 'i' is not passed
            print(f"Creating communication analysis plot for pair: gamma4={gamma4}, gamma5={gamma5}")
            
            fig, ax = plot_pred_perf_vs_dim_3( # Removed color_index argument
                performance_data,
                gamma4,
                gamma5,
                contrast
                # i # Removed color index
            )
            
            # Save the figure if plot was created successfully
            if fig is not None:
                # Create a unique filename for this pair
                save_path = os.path.join(
                    plots_dir, 
                    f'Comm_analysis_3_g4_{gamma4:.2f}_g5_{gamma5:.2f}_contrast_{contrast}.pdf'
                )
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig) # Close the figure to free memory
                print(f"Saved plot to: {save_path}")
            else:
                 print(f"Skipping save for pair gamma4={gamma4}, gamma5={gamma5} due to missing data.")

    else:
        print(f"Warning: Data file not found at {comm_data_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_communication_analysis_3.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_communication_analysis_3(results_dir) 