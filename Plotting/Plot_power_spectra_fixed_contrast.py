import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
# from Plotting import plot_power_spectra 

def plot_feedback_gain_results(results_dir, area='V1'):
    """
    Plot power spectra for feedback gain analysis.
    
    Args:
        results_dir (str): Path to the results directory containing data
        area (str): Brain area to plot ('V1' or 'V2')
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"\n=== Starting Power Spectra Analysis for {area} ===")
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
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
    
    # Create plots for each contrast value
    for contrast in c_vals:
        print(f"Creating plot for contrast = {contrast}")
        
        # Create plots
        fig, axs = plot_power_spectra(
            normalized_power_data,
            gamma_vals,
            contrast,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False
        )
        
        # Save the figure
        save_path = os.path.join(plots_dir, f'power_spectra_{area}_{gain_type}_contrast_{contrast}.pdf')
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")
def plot_power_spectra_fixed_gamma(power_data, contrast_vals, gamma, line_width=5, line_labelsize=42, legendsize=42):
    """
    Plot power spectra for a fixed gamma value across different contrasts.
    Expects pre-normalized power data (normalized by max power at contrast=1).
    
    Args:
        power_data (dict): Dictionary containing normalized power spectra data
        contrast_vals (list): List of contrast values to plot
        gamma (float): The gamma value to plot
        line_width (int): Width of plotted lines
        line_labelsize (int): Size of axis labels
        legendsize (int): Size of legend text
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a truncated viridis colormap that does not include the last yellow part
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    
    norm = mcolors.Normalize(vmin=min(contrast_vals), vmax=max(contrast_vals))
    
    for contrast in contrast_vals:
        key = (gamma, contrast)
        if key not in power_data:
            print(f"Warning: No power data found for contrast = {contrast} and gamma = {gamma}")
            continue
            
        data = power_data[key]
        freq = data['freq']
        power = data['power']  # Data is already normalized
        
        # Get color from truncated colormap
        color = truncated_viridis(norm(contrast))
        
        # Label showing contrast value
        label = rf'$c={contrast}$'
        
        ax.plot(freq, power, lw=line_width, linestyle='-', label=label, color=color)
    
    # Fix the LaTeX formatting in labels
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{Normalized\;V1\;Power}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    ax.legend(fontsize=legendsize, loc='best', frameon=False, handletextpad=0.2, handlelength=1.0, labelspacing=0.2)
    # ax.set_xlim(0,1000)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # Add title showing gamma value
    ax.set_title(rf'$\gamma_1={gamma}$', fontsize=line_labelsize)

    # Use manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot power spectra for V1 or V2')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()
    
    plot_feedback_gain_results(args.results_dir, args.area)