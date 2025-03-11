import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as mcolors
# from Plotting import plot_power_spectra_fixed_gamma

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
    print(f"Global normalization factor: {norm_factor}")
    
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
        
        # Create the plot
        fig, ax = plot_power_spectra_fixed_gamma(
            normalized_power_data,
            c_vals,
            gamma,
            line_width=5,
            line_labelsize=42,
            legendsize=42
        )
        
        # Save the plot
        save_path = os.path.join(plots_dir, f'power_spectrum_{area}_gamma_{gamma}_{gain_type}.pdf')
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
    
    # Add 1/f^4 reference line
    # Create reference frequencies (avoid 0 for log scale)
    ref_freqs = np.logspace(0, 3, 1000)  # 1 to 1000 Hz
    
    # Create 1/f^4 scaling
    # Scale amplitude to be visible on the plot (adjust this value as needed)
    scale_factor = 0.1  
    ref_power = scale_factor * ref_freqs**(-4)
    
    # Plot the reference line
    ax.plot(ref_freqs, ref_power, 'k--', lw=3, label=r'$1/f^4$')
    
    # Fix the LaTeX formatting in labels
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{Normalized\;V1\;Power}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    ax.legend(fontsize=legendsize, loc='best', frameon=False, handletextpad=0.2, handlelength=1.0, labelspacing=0.2)
    
    # Set log scales first
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Then set appropriate limits for log scale (can't start at 0)
    ax.set_xlim(1, 1000)  # Changed from 0 to 1 for lower limit
    
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
    
    plot_fixed_gamma_power_spectra(args.results_dir, args.area) 