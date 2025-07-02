import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import matplotlib.colors as mcolors
# from Plotting import plot_power_spectra 

# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Create a color gradient using gray colormap with contrast-based spacing
gamma_values = np.array([0.00,0.25,0.5,0.75,1.0])
# Normalize contrast values to [0,1] range for colormap
positions = (gamma_values - gamma_values.min()) / (gamma_values.max() - gamma_values.min())
# Adjust the range of the colormap (0.1 to 0.8 for grays)
positions = 0.3 + positions * 0.7  # This maps positions to range [0.1, 0.8]
cmap = plt.cm.get_cmap('Blues')
colors = [cmap(pos) for pos in positions]

# Set up plotting parameters for journal-quality figures
from Plotting import setup_plot_params # Import the setup function

# Set up common plot parameters
setup_plot_params()
# Specific overrides for this script
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
    "axes.prop_cycle": plt.cycler(color=colors) # Script-specific color cycle
})

def plot_power_spectra_fixed_contrast(power_data, param_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Create power spectra plot for a fixed contrast value.
    """
    # Create main plot
    fig, ax = plt.subplots()

    # Create lists for legend
    legend_handles = []
    legend_labels = []

    # Plot the data for each parameter value
    for param in param_vals:
        key = (param, contrast)
        if key not in power_data:
            print(f"Warning: No data found for param={param}, contrast={contrast}")
            continue

        data = power_data[key]
        freq = data['freq']
        normalized_power = data['power']  # Already normalized

        # Plot the data
        line = ax.plot(freq, normalized_power, '-')[0]

        # Add to legend with appropriate label
        if fb_gain:
            # Calculate percentage change relative to baseline 1.0
            label = f'{param:.2f}'
        elif input_gain_beta1:
            # Calculate percentage change relative to baseline 1.0
            label = f'{param:.2f}'
        elif input_gain_beta4:
             # Keep original formatting if needed, or apply similar logic if baseline is known
            label = f'{param:.2f}'

        legend_handles.append(line)
        legend_labels.append(label)

    # Configure plot (labels, ticks, legend, etc.)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('V1 Power Normalized')

    ax.set_xlim(0, 80)
    # Match grid plot ticks
    major_xticks = np.array([20, 40, 60, 80])
    minor_xticks = np.array([10, 30, 50, 70])
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels([str(x) for x in major_xticks])

    # ax.set_ylim(-0.1, 0.6)
    # ax.set_ylim(-0.25, 1.0)
    # ax.set_ylim(-0.1, 0.6)
    # yticks = np.array([0.0,0.3,0.6])
    # yticks = np.array([0.0,0.5,1.0])
    # Match grid plot limits
    ax.set_ylim(-0.25, 1.05)
    yticks = np.array([0.0, 0.5, 1.0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])

    # Set up ticks to match main plot
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

    # # Enable minor ticks
    # ax.minorticks_on()

     # Create legend with parameter name as title
    param_name = ''
    if fb_gain:
        param_name = 'Feedback gain'
    elif input_gain_beta1:
        param_name = 'Input gain'
    elif input_gain_beta4:
        param_name = 'Input gain'

    if param_name:
        ax.legend(legend_handles, legend_labels,
                #  loc='center',
                #  bbox_to_anchor=(0.40, 0.25),
                 loc='upper right',
                 title=param_name,
                 ncol=1,
                 handlelength=1.5,
                 columnspacing=1.0,
                 handletextpad=0.2,
                 borderaxespad=0.1,
                 labelspacing=0.25,
                 title_fontsize=plt.rcParams['legend.fontsize'])

    return fig, ax

def plot_feedback_gain_results(results_dir, area='V1'):
    """
    Plot power spectra results for both feedback gain and input gain beta1.

    Args:
        results_dir (str): Path to results directory
        area (str): Brain area to plot ('V1' or 'V2')
    """
    print(f"\n=== Starting Power Spectra Analysis for {area} ===")
    print(f"Processing directory: {results_dir}")

    # Verify directories exist
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    data_dir = os.path.join(results_dir, 'Data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Create plots directory
    plots_dir = os.path.join(results_dir, 'Plots', 'Power_Spectra')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot feedback gain data if it exists
    gain_type = f'fb_gain_{area.lower()}'
    data_file = os.path.join(data_dir, f'power_spectra_{gain_type}.npy')
    if os.path.exists(data_file):
        print(f"Loading feedback gain data from: {data_file}")
        power_data = np.load(data_file, allow_pickle=True).item()

        # Extract gamma and contrast values from the data
        gamma_vals = sorted(set(g for g, _ in power_data.keys()))
        c_vals = sorted(set(c for _, c in power_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")

        # Get background power (lowest parameter and contrast)
        min_param = min(gamma_vals)
        min_contrast = min(c_vals)
        background_key = (min_param, min_contrast)
        background_power = power_data[background_key]['power']
        print(f"Using background from gamma={min_param}, contrast={min_contrast}")

        # Normalize all power data using (power - background)/(power + background)
        normalized_power_data = {}
        for key, data in power_data.items():
            power = data['power']
            # Avoid division by zero or near-zero if power and background are very close

            normalized_power = (power - background_power) / (power + background_power)
            # Or handle as appropriate, e.g., np.nan
            normalized_power_data[key] = {
                'freq': data['freq'],
                'power': normalized_power
            }

        # Create plots for each contrast value (skip lowest contrast as it's background)
        for contrast in c_vals[1:]:  # Skip lowest contrast
            print(f"Creating plot for contrast = {contrast}")

            # Create plots using normalized data
            fig, ax = plot_power_spectra_fixed_contrast(
                normalized_power_data,
                gamma_vals,
                contrast,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False
            )

            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectra_{gain_type}_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")

    # Plot input gain beta1 data if it exists
    gain_type = f'input_gain_beta1_{area.lower()}'
    data_file = os.path.join(data_dir, f'power_spectra_{gain_type}.npy')
    if os.path.exists(data_file):
        print(f"Loading input gain beta1 data from: {data_file}")
        power_data = np.load(data_file, allow_pickle=True).item()

        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(set(b for b, _ in power_data.keys()))
        c_vals = sorted(set(c for _, c in power_data.keys()))
        print(f"\nBeta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")

        # Get background power (lowest parameter and contrast)
        min_param = min(beta1_vals)
        min_contrast = min(c_vals)
        background_key = (min_param, min_contrast)
        background_power = power_data[background_key]['power']
        print(f"Using background from beta1={min_param}, contrast={min_contrast}")

        # Normalize all power data using (power - background)/(power + background)
        normalized_power_data = {}
        for key, data in power_data.items():
            power = data['power']
            # Avoid division by zero or near-zero

            normalized_power = (power - background_power) / (power + background_power)

            normalized_power_data[key] = {
                'freq': data['freq'],
                'power': normalized_power
            }

        # Create plots for each contrast value
        for contrast in c_vals[1:]:  # Skip lowest contrast
            print(f"Creating plot for contrast = {contrast}")

            # Create plots using normalized data
            fig, ax = plot_power_spectra_fixed_contrast(
                normalized_power_data,
                beta1_vals,
                contrast,
                fb_gain=False,
                input_gain_beta1=True,
                input_gain_beta4=False
            )

            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectra_{gain_type}_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot power spectra for V1 or V2')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()

    plot_feedback_gain_results(args.results_dir, args.area)