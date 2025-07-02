import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as mcolors

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


def plot_coherence_data_fixed_contrast(coherence_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Plot coherence for a fixed contrast value across different gamma/beta values.
    """
    # Create main plot
    fig, ax = plt.subplots()

    # Create lists for legend
    legend_handles = []
    legend_labels = []

    # First pass to find global maximum coherence across all gain values
    max_coherence = 0
    for gamma in gamma_vals:
        key = (gamma, contrast)
        if key not in coherence_data:
            continue
        data = coherence_data[key]
        coh = data['coh']
        if len(coh) > 0:
            max_coherence = max(max_coherence, np.max(coh))

    if max_coherence == 0:
        print(f"Warning: Maximum coherence is 0 for contrast={contrast}. Skipping plot.")
        plt.close(fig)
        return None, None

    print(f"Maximum coherence for contrast={contrast} across all gain values: {max_coherence:.3f}")

    # Second pass to plot normalized data
    for gamma in gamma_vals:
        key = (gamma, contrast)
        if key not in coherence_data:
            print(f"Warning: No coherence data found for gamma={gamma}, contrast={contrast}")
            continue

        data = coherence_data[key]
        freq = data['freq']
        coh = data['coh']

        # Normalize coherence by the maximum value
        normalized_coh = coh / max_coherence

        # Plot the data
        line = ax.plot(freq, normalized_coh, '-')[0]

        # Determine label based on which gain is being varied
        label = ''
        if fb_gain:
            label = f'{gamma:.2f}'
        elif input_gain_beta1:
            label = f'{gamma:.2f}'
        elif input_gain_beta4:
            label = f'{gamma:.2f}'

        legend_handles.append(line)
        legend_labels.append(label)

    # Configure plot (labels, ticks, legend, etc.)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('V1-V2 Coherence')

    # Match grid plot settings
    ax.set_xlim(0, 80)
    major_xticks = np.array([20, 40, 60, 80])
    minor_xticks = np.array([10, 30, 50, 70])
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels([str(x) for x in major_xticks])

    # Set y-ticks with normalized range
    ax.set_ylim(-0.05, 1.05) # Matches grid plot
    yticks = np.array([0.0,  0.5,  1.0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks]) # Use .1f for consistency

    # Set up ticks to match main plot
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

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

def plot_feedback_gain_results(results_dir):
    """
    Plot coherence for feedback gain analysis.

    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')

    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Coherence')
    os.makedirs(plots_dir, exist_ok=True)

    # Load feedback gain data
    coherence_file = os.path.join(data_dir, 'coherence_fb_gain_all.npy')
    if os.path.exists(coherence_file):
        print(f"Loading feedback gain coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()

        # Extract gamma and contrast values from the data
        gamma_vals = sorted(set(g for g, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")

        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating feedback gain plot for contrast = {contrast}")

            # Create plots
            fig, ax = plot_coherence_data_fixed_contrast(
                coherence_data,
                gamma_vals,
                contrast,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False
            )

            if fig is not None:
                # Save the figure
                save_path = os.path.join(plots_dir, f'coherence_fb_gain_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    else:
        print(f"Warning: Coherence data file not found at {coherence_file}")

    # Load beta1 gain data
    coherence_file = os.path.join(data_dir, 'coherence_input_gain_beta1_all.npy')
    if os.path.exists(coherence_file):
        print(f"\nLoading beta1 gain coherence data from: {coherence_file}")
        coherence_data = np.load(coherence_file, allow_pickle=True).item()

        # Extract beta1 and contrast values from the data
        beta1_vals = sorted(set(b for b, _ in coherence_data.keys()))
        c_vals = sorted(set(c for _, c in coherence_data.keys()))
        print(f"Beta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")

        # Create plots for each contrast value
        for contrast in c_vals:
            print(f"Creating beta1 gain plot for contrast = {contrast}")

            # Create plots
            fig, ax = plot_coherence_data_fixed_contrast(
                coherence_data,
                beta1_vals,
                contrast,
                fb_gain=False,
                input_gain_beta1=True,
                input_gain_beta4=False
            )

            if fig is not None:
                # Save the figure
                save_path = os.path.join(plots_dir, f'coherence_input_beta1_gain_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    else:
        print(f"Warning: Beta1 gain coherence data file not found at {coherence_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_coherence.py /path/to/results_dir")
        sys.exit(1)

    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir)