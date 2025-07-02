import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

cmap = matplotlib.colormaps.get_cmap('Reds')
colors = [cmap(pos) for pos in positions]

# Specific overrides for this script
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.2,
    "legend.labelspacing": 0.3,
    "axes.prop_cycle": plt.cycler(color=colors) # Script-specific color cycle
})

def plot_fixed_gamma_coherence(results_dir):
    """
    Create coherence plots for each gamma/beta1 value, showing different contrasts.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot feedback gain data if it exists
    coherence_file = os.path.join(data_dir, 'coherence_fb_gain_all.npy')
    if os.path.exists(coherence_file):
        print(f"Loading feedback gain coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()
        
        # Extract gamma and contrast values from the data
        gamma_vals = sorted(set(g for g, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each gamma value
        for gamma in gamma_vals:
            print(f"Creating plot for gamma = {gamma}")
            
            # Create the plot
            fig, ax = plot_coherence_data_fixed_gamma(
                coherence_data,
                c_vals,
                gamma,
                param_name='gamma1'
            )
            
            # Save the plot
            if fig is not None:
                save_path = os.path.join(plots_dir, f'coherence_fixed_gamma_{gamma}_fb_gain.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    
    # Plot input gain beta1 data if it exists
    coherence_file = os.path.join(data_dir, 'coherence_input_gain_beta1_all.npy')
    if os.path.exists(coherence_file):
        print(f"Loading input gain beta1 coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()
        
        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(set(b for b, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"\nBeta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # Create plots for each beta1 value
        for beta1 in beta1_vals:
            print(f"Creating plot for beta1 = {beta1}")
            
            # Create the plot
            fig, ax = plot_coherence_data_fixed_gamma(
                coherence_data,
                c_vals,
                beta1,  # Use beta1 value in place of gamma
                param_name='beta1'
            )
            
            # Save the plot
            if fig is not None:
                save_path = os.path.join(plots_dir, f'coherence_fixed_beta1_{beta1}_input_gain_beta1.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
            
def plot_coherence_data_fixed_gamma(coherence_data, contrast_vals, gamma, param_name='gamma1'):
    """
    Plot coherence for a fixed gamma/beta1 value across different contrasts.
    Uses linear scales for both axes.
    """
    fig, ax = plt.subplots()
    
    # First pass to find global maximum coherence across all contrasts for this gamma
    max_coherence = 0
    for contrast in contrast_vals:
        key = (gamma, contrast)
        if key not in coherence_data:
            continue
        
        data = coherence_data[key]
        coh = data['coh']
        if len(coh) > 0:
            max_coherence = max(max_coherence, np.max(coh))
    
    if max_coherence == 0:
        print(f"Warning: Maximum coherence is 0 for {param_name}={gamma}. Skipping plot.")
        plt.close(fig)
        return fig, ax

    print(f"Maximum coherence for {param_name}={gamma} across all contrasts: {max_coherence:.3f}")
    
    # Second pass to plot normalized data
    for contrast in contrast_vals:
        key = (gamma, contrast)
        if key not in coherence_data:
            print(f"Warning: No coherence data found for contrast = {contrast} and {param_name} = {gamma}")
            continue
            
        data = coherence_data[key]
        freq = data['freq']
        coh = data['coh']
        
        # Normalize coherence by the maximum value
        normalized_coh = coh / max_coherence
        
        ax.plot(freq, normalized_coh, '-')
    
    # Set labels
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('V1-V2 Coherence')
    
    # Set axis limits and ticks
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    
    ax.set_ylim(0, 1.05)
    yticks = np.array([0, 0.5, 1.0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])
    
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_coherence_fixed_gamma.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_fixed_gamma_coherence(results_dir) 