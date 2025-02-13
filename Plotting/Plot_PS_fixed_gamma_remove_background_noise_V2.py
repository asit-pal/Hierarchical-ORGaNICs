import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import os
import sys
import yaml

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

def plot_power_spectra_fixed_gamma(power_data, c_vals, gamma, line_width=5, line_labelsize=24, legendsize=20):
    """Create power spectra plot for a specific gamma value."""
    print(f"\nPlotting power spectra for gamma={gamma}")
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Color map for different contrast values
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    colors = truncated_viridis(np.linspace(0, 1, len(c_vals)))
    
    # Get frequency array (same for all contrasts)
    first_key = (gamma, c_vals[0])
    if first_key not in power_data:
        print(f"Error: Data not found for gamma={gamma}, contrast={c_vals[0]}")
        return None, None
    
    freq = power_data[first_key]['freq']
    
    # Calculate average power spectrum across all contrasts
    power_sum = np.zeros_like(power_data[first_key]['power'])
    valid_contrasts = 0
    
    for c in c_vals:
        key = (gamma, c)
        if key in power_data:
            power_sum += power_data[key]['power']
            valid_contrasts += 1
    
    if valid_contrasts == 0:
        print("Error: No valid data found for any contrast value")
        return None, None
    
    power_avg = power_sum / valid_contrasts
    
    # Fit background using average power spectrum
    params = fit_background_noise(freq, power_avg)
    if params is None:
        print("Error: Failed to fit background noise")
        return None, None
        
    background = generate_background(freq, params)
    
    # Store cleaned powers for y-axis limits
    all_normalized_powers = []
    
    # Plot spectra for all contrasts
    lines = []
    labels = []
    for c, color in zip(c_vals, colors):
        key = (gamma, c)
        if key not in power_data:
            continue
            
        power = power_data[key]['power']
        # First subtract background, then normalize by dividing by background
        normalized_power = (power - background) / background
        all_normalized_powers.append(normalized_power)
        
        line = ax.plot(freq, normalized_power, color=color, linewidth=line_width)
        lines.append(line[0])
        labels.append(f'{int(c * 100)}%')
    
    # Set plot properties
    ax.set_xlabel('Frequency (Hz)', fontsize=line_labelsize)
    ax.set_ylabel('(Power - Background) / Background', fontsize=line_labelsize)
    ax.tick_params(labelsize=line_labelsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'Normalized Power Spectra (γ={gamma})', fontsize=line_labelsize)
    
    # Set axis limits
    all_normalized_powers = np.array(all_normalized_powers)
    freq_mask = (freq >= 5) & (freq <= 80)
    ymax = np.max(np.abs(all_normalized_powers[:, freq_mask])) * 1.1
    ax.set_ylim(-ymax, ymax)  # Symmetric around 0
    
    # Set x-axis ticks
    xticks = np.array([0, 20, 40, 60])
    ax.set_xticks(xticks)
    
    # Add legend with larger size
    ax.legend(lines, labels, fontsize=legendsize, loc='upper right', 
             title='Contrast', title_fontsize=legendsize,frameon=False)
    
    plt.tight_layout()
    return fig, ax

def plot_power_spectra_results(results_dir):
    """Main function to load data and create power spectra plots."""
    print(f"\n=== Starting Power Spectra Analysis ===")
    print(f"Processing directory: {results_dir}")
    
    # Verify directories exist
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    data_dir = os.path.join(results_dir, 'Data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Load config
    config_files = [f for f in os.listdir(results_dir) if f.endswith('.yaml')]
    if not config_files:
        print(f"Error: No yaml config file found in {results_dir}")
        return
    
    config_path = os.path.join(results_dir, config_files[0])
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("\nLoaded configuration:")
            print("Power_spectra.Feedback_gain settings:")
            print(f"  enabled: {config['Power_spectra']['Feedback_gain']['enabled']}")
            print(f"  c_vals: {config['Power_spectra']['Feedback_gain']['c_vals']}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'Plots', 'Power_Spectra')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Process feedback gain data
    if config['Power_spectra']['Feedback_gain']['enabled']:
        gain_type = 'fb_gain'
        c_vals = [float(c) for c in config['Power_spectra']['Feedback_gain']['c_vals']]
        
        data_file = os.path.join(data_dir, f'power_spectra_{gain_type}_V2.npy')
        if not os.path.exists(data_file):
            print(f"Error: Power spectra data file not found: {data_file}")
            return
            
        print(f"Loading power spectra data...")
        power_data = np.load(data_file, allow_pickle=True).item()
        
        # Extract gamma values from the data
        gamma_vals = sorted(set(g for g, _ in power_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Using contrast values: {c_vals}")
        
        # Create plot for each gamma value
        for gamma in gamma_vals:
            print(f"\nProcessing gamma = {gamma}")
            fig, ax = plot_power_spectra_fixed_gamma(
                power_data=power_data,
                c_vals=c_vals,  # Use contrast values from config
                gamma=gamma
            )
            
            if fig is not None:
                save_path = os.path.join(plots_dir, f'power_spectrum_gamma_{gamma}_{gain_type}.pdf')
                fig.savefig(save_path, dpi=400, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
            else:
                print(f"Failed to create plot for gamma={gamma}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_PS_fixed_gamma_remove_background_noise.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    plot_power_spectra_results(results_dir)
