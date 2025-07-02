import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
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

def plot_coherence_grid(results_dir):
    """
    Plots coherence for feedback gain and input gain beta1 in a 2x6 grid.
    Top row: Feedback gain vs contrast.
    Bottom row: Input gain beta1 vs contrast.
    Coherence is normalized by the maximum value found for each contrast across parameters.
    """

    # --- Setup ---
    print(f"\n=== Starting Coherence Grid Plotting ===")
    print(f"Processing directory: {results_dir}")

    # Verify directories exist
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    data_dir = os.path.join(results_dir, 'Data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence') # Save in Coherence subdir
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load Raw Data --- 
    raw_data_sets = {}
    param_value_sets = {}

    # Define filenames directly based on Plot_coherence_fixed_contrast.py
    gain_types_to_load = {
        'fb_gain': 'coherence_fb_gain_all.npy',
        'input_gain_beta1': 'coherence_input_gain_beta1_all.npy'
    }

    all_c_vals = None # To store contrast values

    for key, filename in gain_types_to_load.items():
        data_file = os.path.join(data_dir, filename)
        if not os.path.exists(data_file):
            print(f"Warning: Data file not found, skipping {key}: {data_file}")
            continue

        print(f"Loading {key} data from: {data_file}")
        raw_coherence_data = np.load(data_file, allow_pickle=True).item()
        raw_data_sets[key] = raw_coherence_data

        # Extract param and contrast values
        params = sorted(set(p for p, _ in raw_coherence_data.keys()))
        contrasts = sorted(set(c for _, c in raw_coherence_data.keys()))
        param_value_sets[key] = params
        if all_c_vals is None:
            all_c_vals = contrasts
        elif set(all_c_vals) != set(contrasts): # Check set equality
             print(f"Warning: Contrast values differ between datasets! Using {all_c_vals}")

        print(f"{key} values found: {params}")
        if all_c_vals is not None:
             print(f"Contrast values found: {all_c_vals}")

    if not raw_data_sets:
        print("Error: No data loaded successfully. Exiting.")
        return

    if all_c_vals is None or len(all_c_vals) < 1: # Need at least one contrast
        print("Error: Not enough contrast values found to plot.")
        return

    # --- Normalize Data --- 
    normalized_data = {key: {} for key in raw_data_sets}
    all_c_vals = sorted(list(all_c_vals)) # Ensure sorted list
    
    for key, raw_coh_data in raw_data_sets.items():
        params_for_key = param_value_sets[key]
        for contrast in all_c_vals:
            # Find max coherence for this contrast across all params for this key
            max_coh_for_contrast = 0
            valid_data_found = False
            for param in params_for_key:
                data_key = (param, contrast)
                if data_key in raw_coh_data:
                    coh = raw_coh_data[data_key]['coh']
                    if coh.size > 0:
                        max_coh_for_contrast = max(max_coh_for_contrast, np.max(coh))
                    valid_data_found = True
            
            if not valid_data_found:
                print(f"Warning: No data found for {key} at contrast {contrast}")
                continue

            # Normalize data for this contrast
            if max_coh_for_contrast == 0:
                print(f"Warning: Max coherence is 0 for {key}, contrast={contrast}. Cannot normalize.")
                for param in params_for_key:
                    data_key = (param, contrast)
                    if data_key in raw_coh_data:
                        freq = raw_coh_data[data_key]['freq']
                        normalized_data[key][data_key] = {
                            'freq': freq,
                            'coh': np.zeros_like(freq)
                        }
            else:
                 for param in params_for_key:
                    data_key = (param, contrast)
                    if data_key in raw_coh_data:
                        freq = raw_coh_data[data_key]['freq']
                        coh = raw_coh_data[data_key]['coh']
                        norm_coh = coh / max_coh_for_contrast
                        normalized_data[key][data_key] = {
                            'freq': freq,
                            'coh': norm_coh
                        }

    # Determine contrasts to plot (all contrasts)
    contrasts_to_plot = all_c_vals[1:] # Skip the first contrast (likely 0%)
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
            plot_occurred = False
            for i, param in enumerate(gamma_vals):
                key = (param, contrast)
                if key not in norm_fb_data:
                    continue
                data = norm_fb_data[key]
                line = ax.plot(data['freq'], data['coh'], '-')[0]
                plot_occurred = True
                if col_idx == 0:
                     label = f'{param:.2f}'
                     fb_handles.append(line)
                     fb_labels.append(label)
            
            if plot_occurred:
                contrast_percent = contrast * 100
                ax.set_title(f'Contrast {contrast_percent:.1f}%')
                ax.set_ylim(-0.05, 1.05)
                if col_idx == 0:
                    ax.set_ylabel('Norm. Coherence')
                else:
                     plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.axis('off')

    # Bottom Row: Input Gain Beta1
    if 'input_gain_beta1' in normalized_data:
        norm_beta1_data = normalized_data['input_gain_beta1']
        beta1_vals = param_value_sets['input_gain_beta1']
        axes[1, 0].set_prop_cycle(plt.cycler(color=colors))
        for col_idx, contrast in enumerate(contrasts_to_plot):
            ax = axes[1, col_idx]
            plot_occurred = False
            for i, param in enumerate(beta1_vals):
                key = (param, contrast)
                if key not in norm_beta1_data:
                    continue
                data = norm_beta1_data[key]
                line = ax.plot(data['freq'], data['coh'], '-')[0]
                plot_occurred = True
                if col_idx == 0:
                     label = f'{param:.2f}'   
                     beta1_handles.append(line)
                     beta1_labels.append(label)
            
            if plot_occurred:
                ax.set_ylim(-0.05, 1.05)
                if col_idx == 0:
                    ax.set_ylabel('Norm. Coherence')
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
            else:
                 ax.axis('off')

    # Handle remaining empty plots
    for row in range(2):
        for col in range(num_contrasts, num_cols):
             if col < axes.shape[1]:
                axes[row, col].axis('off')

    fig.supxlabel('Frequency (Hz)', fontsize=plt.rcParams['axes.labelsize'])

    bottom_ax = axes[1, 0]
    major_ticks = np.array([ 40,  80])
    minor_ticks = np.array([20, 60])
    bottom_ax.set_xticks(major_ticks)
    bottom_ax.set_xticks(minor_ticks, minor=True)
    bottom_ax.set_xticklabels([str(x) for x in major_ticks])
    bottom_ax.set_xlim(0, 80)

    # --- Add Legends --- 
    if fb_handles:
        legend_fb = fig.legend(fb_handles, fb_labels, loc='center left', bbox_to_anchor=(1.0, 0.75),
                   title='Feedback Gain', ncol=1,
                   fontsize=plt.rcParams['legend.fontsize'], title_fontsize=plt.rcParams['legend.fontsize'])
        legend_fb.get_title().set_fontweight('bold')
        fig.text(1.0, 0.60, 'Input gain = 1.00', 
                 ha='left', va='top', transform=fig.transFigure, 
                 fontsize=plt.rcParams['legend.fontsize'])

    if beta1_handles:
        legend_beta1 = fig.legend(beta1_handles, beta1_labels, loc='center left', bbox_to_anchor=(1.0, 0.35),
                   title='Input Gain', ncol=1,
                   fontsize=plt.rcParams['legend.fontsize'], title_fontsize=plt.rcParams['legend.fontsize'])
        legend_beta1.get_title().set_fontweight('bold')
        fig.text(1.0, 0.15, 'Feedback gain = 1.00', 
                 ha='left', va='top', transform=fig.transFigure, 
                 fontsize=plt.rcParams['legend.fontsize'])

    # --- Finalize and Save --- 
    fig.subplots_adjust(left=0.05, right=0.8, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

    save_path = os.path.join(plots_dir, 'coherence_grid.pdf')
    fig.savefig(save_path, dpi=400, format='pdf')
    plt.close(fig)
    print(f"Saved coherence grid plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot coherence grid')
    parser.add_argument('results_dir', help='Path to results directory')
    args = parser.parse_args()

    plot_coherence_grid(args.results_dir)
