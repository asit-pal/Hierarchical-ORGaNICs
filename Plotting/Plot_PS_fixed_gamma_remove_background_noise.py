import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import os
import sys
import yaml
import argparse

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
    mask = (freq > 1) & (freq <= 10)
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

def plot_power_spectra_fixed_gamma(power_data, c_vals, gamma, line_width=5, line_labelsize=24, legendsize=20, param_name='gamma1'):
    """
    Create power spectra plot for a specific gamma/beta1 value.
    
    Args:
        power_data (dict): Dictionary containing power spectra data
        c_vals (list): List of contrast values to plot
        gamma (float): The gamma/beta1 value to plot
        line_width (int): Width of plotted lines
        line_labelsize (int): Size of axis labels
        legendsize (int): Size of legend text
        param_name (str): Parameter name ('gamma1' or 'beta1')
    """
    print(f"\nPlotting power spectra for {param_name}={gamma}")
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Color map for different contrast values
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    colors = truncated_viridis(np.linspace(0, 1, len(c_vals)))
    
    # Get frequency array (same for all contrasts)
    first_key = (gamma, np.min(c_vals))
    if first_key not in power_data:
        print(f"Error: Data not found for {param_name}={gamma}, contrast={c_vals[0]}")
        return None, None
    
    freq = power_data[first_key]['freq']
    
    # # Calculate average power spectrum across all contrasts
    # power_sum = np.zeros_like(power_data[first_key]['power'])
    # valid_contrasts = 0
    
    # for c in c_vals:
    #     key = (gamma, c)
    #     if key in power_data:
    #         power_sum += power_data[key]['power']
    #         valid_contrasts += 1
    
    # if valid_contrasts == 0:
    #     print("Error: No valid data found for any contrast value")
    #     return None, None
    
    # power_avg = power_sum / valid_contrasts
    
    # Fit background using average power spectrum
    # params = fit_background_noise(freq, power_avg)
    # if params is None:
    #     print("Error: Failed to fit background noise")
    #     return None, None
        
    # background = generate_background(freq, params)
    background = power_data[first_key]['power']
    # Store normalized powers for y-axis limits
    all_normalized_powers = []
    
    # Plot normalized spectra for all contrasts
    lines = []
    labels = []
    for c, color in zip(c_vals, colors):
        key = (gamma, c)
        if key not in power_data:
            continue
            
        power = power_data[key]['power']
        # subtract the background
        power_sub = power - background
        power_deno = power + background
        # Normalize by dividing by the background fit
        normalized_power = power_sub / power_deno
        all_normalized_powers.append(normalized_power)
        
        line = ax.plot(freq, normalized_power, color=color, linewidth=line_width)
        lines.append(line[0])
        labels.append(f'{int(c * 100)}%')
    
    # Set plot properties
    ax.set_xlabel('Frequency (Hz)', fontsize=line_labelsize)
    ax.set_ylabel('Normalized Power', fontsize=line_labelsize)
    ax.tick_params(labelsize=line_labelsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set title with appropriate symbol based on param_name
    symbol = r'$\gamma$' if param_name == 'gamma1' else r'$\beta_1$'
    ax.set_title(f'Normalized Power Spectra ({symbol}={gamma})', fontsize=line_labelsize)
    
    # Set axis limits
    all_normalized_powers = np.array(all_normalized_powers)
    # freq_mask = (freq >= 5) & (freq <= 80)
    # ymax = np.max(all_normalized_powers[:, freq_mask]) * 1.1
    # ax.set_ylim(, ymax)
    ax.set_xlim(0, 80)
    
    # Set x-axis ticks
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    
    # Add legend
    ax.legend(lines, labels, fontsize=legendsize, loc='upper right', 
             title='Contrast', title_fontsize=legendsize)
    
    plt.tight_layout()
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
            fig, ax = plot_power_spectra_fixed_gamma(
                power_data=power_data,
                c_vals=c_vals,
                gamma=gamma,
                param_name='gamma1'  # Explicitly set for feedback gain
            )
            
            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectrum_{area}_{gain_type}_{gamma}.pdf')
                fig.savefig(save_path, dpi=400, bbox_inches='tight')
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
            fig, ax = plot_power_spectra_fixed_gamma(
                power_data=power_data,
                c_vals=c_vals,
                gamma=beta1,  # Use beta1 value in place of gamma
                param_name='beta1'  # Explicitly set for input gain beta1
            )
            
            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectrum_{area}_{gain_type}_{beta1}.pdf')
                fig.savefig(save_path, dpi=400, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
            else:
                print(f"Failed to create plot for beta1={beta1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot power spectra for V1 or V2')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--area', choices=['V1', 'V2'], default='V1',
                      help='Brain area to plot (V1 or V2)')
    args = parser.parse_args()
    
    plot_power_spectra_results(args.results_dir, args.area)
