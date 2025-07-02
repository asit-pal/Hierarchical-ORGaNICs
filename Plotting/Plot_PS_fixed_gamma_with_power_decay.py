import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import os
import argparse

from Plotting import setup_plot_params # Import the setup function

# Set up common plot parameters
setup_plot_params()

# --- Script-specific setup ---

# Create a color gradient using gray colormap with contrast-based spacing
contrast_values = np.array([2.5, 3.7, 6.1, 9.7, 16.3, 35.9, 50.3, 72.0])
# Normalize contrast values to [0,1] range for colormap
positions = (contrast_values - contrast_values.min()) / (contrast_values.max() - contrast_values.min())
# Adjust the range of the colormap (0.1 to 0.8 for grays)
positions = 0.2 + positions * 0.6  # This maps positions to range [0.1, 0.8]

cmap = plt.cm.get_cmap('Reds')
colors = [cmap(pos) for pos in positions]

# Specific overrides for this script
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
    "axes.prop_cycle": plt.cycler(color=colors) # Script-specific color cycle
})

def one_over_f(f, A, alpha):
    """Calculate 1/f^α noise."""
    return A / (f ** alpha)

def generate_background(freq, params):
    """Generate background noise using fitted parameters."""
    A, alpha = params
    return one_over_f(freq, A, alpha)

def fit_background_noise(freq, power):
    """
    Fit 1/f^α noise to the power spectrum, using only frequencies between 1-10 Hz
    """
    # Use only frequencies between 1-10 Hz for fitting
    mask = (freq > 1) & (freq <= 1000)
    freq_fit = freq[mask]
    power_fit = power[mask]
    
    # Fit in log-log space
    log_freq = np.log(freq_fit)
    log_power = np.log(power_fit)
    
    # Initial guess for parameters [A, alpha]
    p0 = [1.0, 1.0]
    
    try:
        # Fit the model
        params, _ = curve_fit(lambda x, A, alpha: np.log(one_over_f(np.exp(x), A, alpha)),
                            log_freq, log_power, p0=p0)
        return params
    except Exception as e:
        print(f"Error during background fitting: {str(e)}")
        return None

def plot_power_spectra_linear(power_data, c_vals, gamma, param_name='gamma1'):
    """
    Create power spectra plot for a specific gamma/beta1 value.
    """
    print(f"\nPlotting power spectra for {param_name}={gamma}")
    
    # Create main plot
    fig, ax = plt.subplots()
    
    # Get frequency array (assumes same for all contrasts)
    first_key = (gamma, c_vals[0])
    if first_key not in power_data:
        print(f"Error: Data not found for {param_name}={gamma}, contrast={c_vals[0]}")
        return None, None
    
    freq = power_data[first_key]['freq']
    
    # Use the lowest contrast as the background instead of fitting
    lowest_contrast = min(c_vals)
    background_key = (gamma, lowest_contrast)
    
    if background_key not in power_data:
        print(f"Error: Background data not found for {param_name}={gamma}, contrast={lowest_contrast}")
        return None, None
        
    background = power_data[background_key]['power']
    print(f"Using contrast={lowest_contrast} as background")
    
    # Create lists for legend in the main plot
    legend_handles = []
    legend_labels = []
    
    # Skip the lowest contrast in the main plot as it's used as background
    for c in c_vals:
        if c == lowest_contrast:
            continue  # Skip the background contrast
            
        key = (gamma, c)
        if key not in power_data:
            continue
            
        power = power_data[key]['power']
        # Subtract the background and normalize the power spectrum
        power_sub = power - background
        power_deno = power + background
        normalized_power = power_sub / power_deno
        
        line = ax.plot(freq, normalized_power, '-')[0]
        legend_handles.append(line)
        legend_labels.append(f'{(c * 100):.1f}')
    
    # Configure main plot (labels, ticks, legend, etc.)
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('V1 Power Normalized')
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    
    # Set y axis limits
    ax.set_ylim(-0.25,1.0)
    yticks = np.array([0.0,0.4,0.8])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])

    # Add tick padding
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Create inset in top right with larger size
    ax_inset = inset_axes(ax, width="100%", height="100%",
                         bbox_to_anchor=(0.750, 0.740, 0.275, 0.275),
                         bbox_transform=ax.transAxes)
    
    # Find normalization factor (maximum power at contrast=1)
    norm_factor = max(np.max(power_data[key]['power']) for key in power_data.keys())
    
    # Create normalized data dictionary
    normalized_power_data = {}
    for key, data in power_data.items():
        normalized_power_data[key] = {
            'freq': data['freq'],
            'power': data['power'] / norm_factor
        }
    max_power = max(np.max(normalized_power_data[key]['power']) for key in normalized_power_data.keys())
    print(f"Maximum power in normalized data: {max_power}")
    
    # Plot normalized power spectrum for each contrast value in the inset
    for c in c_vals:
        if c == lowest_contrast:
            continue
        key = (gamma, c)
        if key not in normalized_power_data:
            continue
        ax_inset.plot(normalized_power_data[key]['freq'], 
                     normalized_power_data[key]['power'], '-',linewidth=8)

    # Use highest contrast for reference lines
    key = (gamma, c_vals[0])
    
    freq = normalized_power_data[key]['freq']
    power = normalized_power_data[key]['power']
    
    # Find closest points to our target frequencies
    idx_high = np.abs(freq - 300).argmin()
    power_at_300Hz = power[idx_high]
    
    # Add 1/f^4 reference line (high frequency)
    high_f_min, high_f_max = 300, 1000
    high_freqs = np.array([high_f_min, high_f_max])
    high_alpha = 4.0
    high_amp = power_at_300Hz * (high_f_min ** high_alpha)
    high_power = high_amp / (high_freqs ** high_alpha) 
    shift_factor_high = 5
    high_power_shifted = high_power / shift_factor_high
    
    ax_inset.plot(high_freqs, high_power_shifted, dashes=[4,2],
                 color='red', linewidth=4)
    
    label_x_pos =  high_f_min/6.0
    label_y_pos = high_power_shifted[1] * 1.5
    ax_inset.text(label_x_pos, label_y_pos, '1/f^4', 
                 color='red', fontsize=plt.rcParams['legend.fontsize']*0.9, ha='center', va='bottom')
    
    # Configure inset axes
    ax_inset.set_xlabel('Frequency', labelpad=-8, fontsize=plt.rcParams['legend.fontsize']*0.9)
    ax_inset.set_ylabel('Power', labelpad=-8, fontsize=plt.rcParams['legend.fontsize']*0.9)
    ax_inset.set_xscale('log')
    ax_inset.set_yscale('log')
    ax_inset.set_xlim(1, 1000)
    
    # Set custom x-ticks for the inset (logarithmic scale)
    xticks_inset = np.array([ 1e0,  1e3])
    ax_inset.set_xticks(xticks_inset)
    ax_inset.set_xticklabels(['10^0', '10^3'], fontsize=plt.rcParams['legend.fontsize']*0.9)
    ax_inset.tick_params(axis='x', pad=10)
    # Set custom y-ticks for the inset (logarithmic scale)
    yticks_inset = np.array([1e-8,   1e-0])
    ax_inset.set_yticks(yticks_inset)
    ax_inset.set_yticklabels(['10^-8', '10^0'], fontsize=plt.rcParams['legend.fontsize']*0.9)
    
    # Set spines for inset
    for spine in ['top', 'bottom', 'left', 'right']:
        ax_inset.spines[spine].set_linewidth(plt.rcParams['axes.linewidth'])
    
    # Set aspect ratio for the inset
    ax_inset.set_box_aspect(1)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

def plot_power_spectra_results(results_dir, area='V1'):
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
        
        # Create plot for each gamma value
        for gamma in gamma_vals:
            print(f"\nProcessing gamma = {gamma}")
            fig, ax = plot_power_spectra_linear(
                power_data=power_data,
                c_vals=c_vals,
                gamma=gamma,
                param_name='gamma1'
            )
            
            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectrum_{gain_type}_{gamma}_high_freq.pdf')
                fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
            else:
                print(f"Failed to create plot for gamma={gamma}")
    
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
        
        # Create plot for each beta1 value
        for beta1 in beta1_vals:
            print(f"\nProcessing beta1 = {beta1}")
            fig, ax = plot_power_spectra_linear(
                power_data=power_data,
                c_vals=c_vals,
                gamma=beta1,
                param_name='beta1'
            )
            
            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectrum_{area}_{gain_type}_{beta1}.pdf')
                fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
            else:
                print(f"Failed to create plot for beta1={beta1}")

def main():
    parser = argparse.ArgumentParser(description='Plot power spectra with 1/f^4 decay for a specific gamma')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()
    
    plot_power_spectra_results(args.results_dir, args.area)

if __name__ == "__main__":
    main()
