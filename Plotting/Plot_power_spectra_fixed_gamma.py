import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

# Add project root to Python path if necessary
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params

# Create a color gradient using contrast-based spacing
contrast_values = np.array([2.5, 3.7, 6.1, 9.7, 16.3, 35.9, 50.3, 72.0])
# Normalize contrast values to [0,1] range for colormap
positions = (contrast_values - contrast_values.min()) / (contrast_values.max() - contrast_values.min())
# Adjust the range of the colormap
positions = 0.2 + positions * 0.6  # This maps positions to range [0.2, 0.8]

cmap = plt.cm.get_cmap('Reds')
colors = [cmap(pos) for pos in positions]

# Set up plotting parameters for journal-quality figures
setup_plot_params()
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.2,
    "legend.labelspacing": 0.1,
    "figure.constrained_layout.use": True,
    "axes.prop_cycle": plt.cycler(color=colors)
})

def plot_fixed_gamma_power_spectra(results_dir, area='V1'):
    """
    Create power spectra plots for each gamma value, showing different contrasts.
    All power spectra are normalized by the maximum power at contrast=1 for each gamma.
    
    Args:
        results_dir (str): Path to the results directory containing data
        area (str): Brain area to plot ('V1' or 'V2')
    """
    print(f"\n=== Starting Power Spectra Analysis for {area} ===")
    print(f"Processing directory: {results_dir}")
    
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Load power spectra data from the single file
    gain_type = f'fb_gain_{area.lower()}'
    power_file = os.path.join(data_dir, f'power_spectra_{gain_type}.npy')
    if not os.path.exists(power_file):
        print(f"Error: Power spectra data file not found at {power_file}")
        sys.exit(1)
        
    print(f"Loading power spectra data from: {power_file}")
    power_data = np.load(power_file, allow_pickle=True).item()
    
    # Extract gamma and contrast values from the data
    gamma_vals = sorted(set(g for g, _ in power_data.keys()))
    c_vals = sorted(set(c for _, c in power_data.keys()))
    print(f"\nGamma values found in data: {gamma_vals}")
    print(f"Contrast values found in data: {c_vals}")
    
    # Find normalization factor (maximum power at contrast=1)
    norm_factor = max(np.max(power_data[key]['power']) for key in power_data.keys())
    print(f"Global normalization factor (max power at c=1): {norm_factor}")
    
    # Normalize all power data by this single factor
    normalized_power_data = {}
    for key, data in power_data.items():
        normalized_power_data[key] = {
            'freq': data['freq'],
            'power': data['power'] / norm_factor
        }
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Power_spectra')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a plot for each gamma value
    for gamma in gamma_vals:
        print(f"Creating plot for gamma = {gamma}")
        
        # Create the plot using global rcParams
        fig, ax = plot_power_spectra_fixed_gamma(
            normalized_power_data,
            c_vals,
            gamma
        )
        
        # Save the plot
        save_path = os.path.join(plots_dir, f'power_spectrum_gamma_{gamma}_{gain_type}.pdf')
        fig.savefig(save_path, dpi=400, format='pdf')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")

def plot_power_spectra_fixed_gamma(power_data, c_vals, gamma, ax=None):
    """
    Create a power spectrum plot for a fixed gamma value.
    Uses global rcParams for styling.
    If an axis is provided, plot on it; otherwise, create a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Create a list to store legend handles and labels
    legend_handles = []
    legend_labels = []

    # Plot power spectrum for each contrast value
    for c in c_vals:
        try:
            data = power_data[(gamma, c)]
            line = ax.plot(data['freq'], data['power'], '-')[0]
            legend_handles.append(line)
            legend_labels.append(f'{(c * 100):.1f}')
        except KeyError:
            print(f"Warning: No data for gamma={gamma}, contrast={c}")
            continue

    # Set labels and limits
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized power')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 100)
    ax.set_ylim(1e-4, 1)

    # Add legend with contrast as title
    ax.legend(legend_handles, legend_labels, 
              bbox_to_anchor=(-0.035, -0.035),
              loc='lower left',
              title='Contrast (%)',
              title_fontsize=plt.rcParams['legend.fontsize'])

    ax.set_box_aspect(1)
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot power spectra for V1 or V2')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()
    
    plot_fixed_gamma_power_spectra(args.results_dir, args.area) 