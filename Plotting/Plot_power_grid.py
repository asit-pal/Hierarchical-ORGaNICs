import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import matplotlib.colors as mcolors
import argparse

# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params # Import the setup function

# Create a color gradient using gray colormap with contrast-based spacing
# Copied from Plot_power_fixed_contrast.py for consistency
gamma_values_for_color = np.array([0.00,0.25,0.5,0.75,1.0])
# Normalize contrast values to [0,1] range for colormap
positions = (gamma_values_for_color - gamma_values_for_color.min()) / (gamma_values_for_color.max() - gamma_values_for_color.min()) if gamma_values_for_color.max() > gamma_values_for_color.min() else np.linspace(0,1,len(gamma_values_for_color))
# Adjust the range of the colormap
positions = 0.3 + positions * 0.7  # This maps positions to range [0.1, 0.8]
cmap = matplotlib.colormaps.get_cmap('Blues')
colors = [cmap(pos) for pos in positions]

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=colors) # Script-specific color cycle
})

def plot_power_grid(results_dir, area='V1'):
    """
    Plots power spectra for feedback gain and input gain beta1 in a 2x6 grid.
    Top row: Feedback gain vs contrast.
    Bottom row: Input gain beta1 vs contrast.
    """

    # --- Setup ---
    print(f"\n=== Starting Power Spectra Grid Plotting for {area} ===")
    print(f"Processing directory: {results_dir}")

    # Verify directories exist
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    data_dir = os.path.join(results_dir, 'Data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    plots_dir = os.path.join(results_dir, 'Plots', 'Power_Spectra')
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load and Normalize Data ---
    data_sets = {}
    param_value_sets = {}
    normalized_data = {}

    gain_types_to_load = {
        'fb_gain': f'fb_gain_{area.lower()}',
        'input_gain_beta1': f'input_gain_beta1_{area.lower()}'
    }

    all_c_vals = None # To store contrast values

    for key, gain_type_str in gain_types_to_load.items():
        data_file = os.path.join(data_dir, f'power_spectra_{gain_type_str}.npy')
        if not os.path.exists(data_file):
            print(f"Warning: Data file not found, skipping {key}: {data_file}")
            continue

        print(f"Loading {key} data from: {data_file}")
        raw_power_data = np.load(data_file, allow_pickle=True).item()
        data_sets[key] = raw_power_data

        # Extract param and contrast values
        params = sorted(set(p for p, _ in raw_power_data.keys()))
        contrasts = sorted(set(c for _, c in raw_power_data.keys()))
        param_value_sets[key] = params
        if all_c_vals is None:
            all_c_vals = contrasts
        elif all_c_vals != contrasts:
             print(f"Warning: Contrast values differ between datasets! Using {all_c_vals}")

        print(f"{key} values found: {params}")
        print(f"Contrast values found: {contrasts}")

        # Get background power (lowest parameter and contrast)
        min_param = min(params)
        min_contrast = min(contrasts)
        background_key = (min_param, min_contrast)
        background_power = raw_power_data[background_key]['power']
        print(f"Using background for {key} from param={min_param}, contrast={min_contrast}")

        # Normalize
        current_normalized_data = {}
        for data_key, data in raw_power_data.items():
            power = data['power']
            # Use np.where for safe division
            norm_power = np.where(power + background_power != 0,
                                  (power - background_power) / (power + background_power),
                                  0)
            current_normalized_data[data_key] = {
                'freq': data['freq'],
                'power': norm_power
            }
        normalized_data[key] = current_normalized_data

    if not normalized_data:
        print("Error: No data loaded successfully. Exiting.")
        return

    if all_c_vals is None or len(all_c_vals) < 2:
        print("Error: Not enough contrast values found to plot.")
        return

    contrasts_to_plot = all_c_vals[1:]
    num_contrasts = len(contrasts_to_plot)
    num_cols = 6
    if num_contrasts < num_cols:
        print(f"Warning: Found only {num_contrasts} contrasts to plot, but grid has {num_cols} columns. Some columns will be empty.")
    elif num_contrasts > num_cols:
        print(f"Warning: Found {num_contrasts} contrasts but grid has only {num_cols} columns. Plotting only the first {num_cols}.")
        contrasts_to_plot = contrasts_to_plot[:num_cols]

    # --- Plotting Setup ---
    setup_plot_params()
    plt.rcParams.update({
        "lines.linewidth": 4, 
        "lines.markersize": 5,
        "axes.labelsize": 40, 
        "axes.titlesize": 40,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "legend.fontsize": 36,
        "figure.figsize": (30, 10),
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    })

    # --- Create Grid Plot ---
    fig, axes = plt.subplots(2, num_cols, sharex=True, sharey=True, figsize=plt.rcParams['figure.figsize'])
    
    fb_handles, fb_labels = [], []
    beta1_handles, beta1_labels = [], []

    # Top Row: Feedback Gain
    if 'fb_gain' in normalized_data:
        norm_fb_data = normalized_data['fb_gain']
        gamma_vals = param_value_sets['fb_gain']
        axes[0, 0].set_prop_cycle(plt.cycler(color=colors))
        for col_idx, contrast in enumerate(contrasts_to_plot):
            ax = axes[0, col_idx]
            for i, param in enumerate(gamma_vals):
                key = (param, contrast)
                if key not in norm_fb_data:
                    continue
                data = norm_fb_data[key]
                line = ax.plot(data['freq'], data['power'], '-')[0]
                if col_idx == 0:
                     label = f'{param:.2f}'
                     fb_handles.append(line)
                     fb_labels.append(label)

            contrast_percent = contrast * 100
            ax.set_title(f'Contrast {contrast_percent:.1f}%')
            ax.set_xlim(0, 80)
            ax.set_ylim(-0.25, 1.05)
            if col_idx == 0:
                ax.set_ylabel('Norm. Power')
            else:
                 plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)

    # Bottom Row: Input Gain Beta1
    if 'input_gain_beta1' in normalized_data:
        norm_beta1_data = normalized_data['input_gain_beta1']
        beta1_vals = param_value_sets['input_gain_beta1']
        axes[1, 0].set_prop_cycle(plt.cycler(color=colors))
        for col_idx, contrast in enumerate(contrasts_to_plot):
            ax = axes[1, col_idx]
            for i, param in enumerate(beta1_vals):
                key = (param, contrast)
                if key not in norm_beta1_data:
                    continue
                data = norm_beta1_data[key]
                line = ax.plot(data['freq'], data['power'], '-')[0]
                if col_idx == 0:
                     label = f'{param:.2f}'
                     beta1_handles.append(line)
                     beta1_labels.append(label)

            ax.set_ylim(-0.25, 1.05)
            if col_idx == 0:
                ax.set_ylabel('Norm. Power')
            else:
                 plt.setp(ax.get_yticklabels(), visible=False)

    # Handle empty plots
    for row in range(2):
        for col in range(num_contrasts, num_cols):
            axes[row, col].axis('off')

    fig.supxlabel('Frequency (Hz)', fontsize=plt.rcParams['axes.labelsize'])

    bottom_ax = axes[1, 0]
    major_ticks = np.array([ 40, 80])
    minor_ticks = np.array([20, 60])
    bottom_ax.set_xticks(major_ticks)
    bottom_ax.set_xticks(minor_ticks, minor=True)
    bottom_ax.set_xticklabels([str(x) for x in major_ticks])
    bottom_ax.set_xlim(0, 80)

    # --- Add Legends ---
    if fb_handles:
        leg1 = fig.legend(fb_handles, fb_labels, loc='center left', bbox_to_anchor=(1.0, 0.75),
                   title='Feedback Gain', ncol=1,
                   fontsize=plt.rcParams['legend.fontsize'], title_fontsize=plt.rcParams['legend.fontsize'])
        leg1.get_title().set_fontweight('bold')

    if beta1_handles:
        leg2 = fig.legend(beta1_handles, beta1_labels, loc='center left', bbox_to_anchor=(1.0, 0.25),
                   title='Input Gain', ncol=1,
                   fontsize=plt.rcParams['legend.fontsize'], title_fontsize=plt.rcParams['legend.fontsize'])
        leg2.get_title().set_fontweight('bold')

    # --- Finalize and Save ---
    fig.subplots_adjust(left=0.05, right=0.8, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

    save_path = os.path.join(plots_dir, f'power_spectra_grid_{area.lower()}.pdf')
    fig.savefig(save_path, dpi=400, format='pdf')
    plt.close(fig)
    print(f"Saved grid plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot power spectra grid for V1 or V2')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()

    plot_power_grid(args.results_dir, args.area)
