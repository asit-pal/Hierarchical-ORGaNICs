import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import os
import sys

from Plotting import setup_plot_params # Import the setup function

# Set up common plot parameters
setup_plot_params()

# --- Script-specific setup ---

gamma_values = np.array([0.0,0.25,0.5,0.75,1.0])
# Normalize contrast values to [0,1] range for colormap
positions = (gamma_values - gamma_values.min()) / (gamma_values.max() - gamma_values.min())
# Adjust the range of the colormap (0.1 to 0.8 for grays)
positions = 0.3 + positions * 0.7  # This maps positions to range [0.1, 0.8]
cmap = matplotlib.colormaps.get_cmap('Blues')
colors = [cmap(pos) for pos in positions]

# Specific overrides for this script
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.1,
    "figure.constrained_layout.use": False, # Specific override for this script
    "axes.prop_cycle": plt.cycler(color=colors) # Script-specific color cycle
})

def plot_gain_modulation_results(results_dir):
    """
    Plot steady states for gain modulation analysis.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    # Get the data directory path
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Gain_modulation')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot for each enabled gain type
    # Load feedback gain data
    data_file = os.path.join(data_dir, 'steady_states_fb_gain_all.npy')
    
    if os.path.exists(data_file):
        print(f"Loading feedback gain data from: {data_file}")
        steady_states = np.load(data_file, allow_pickle=True).item()
        
        # Process data to only keep center neuron data (N//2)
        first_value = next(iter(steady_states.values()), None)
        if first_value and isinstance(first_value, list) and isinstance(first_value[0], np.ndarray):
            N = 36  # Assuming 36 neurons for spatial data
            center_idx = N // 2
            steady_states = {
                key: [v[center_idx] for v in value]
                for key, value in steady_states.items()
            }
        
        # Extract gamma and contrast values from the data
        gamma_vals = sorted(list(set([g for g, _ in steady_states.keys()])))
        c_vals = np.array(sorted(list(set([c for _, c in steady_states.keys()]))))
        print(f"Found gamma values: {gamma_vals}")
        print(f"Found contrast values: {c_vals}")
        
        fig, axs = plot_steady_states(
            steady_states=steady_states,
            c_vals=c_vals,
            gamma_vals=gamma_vals,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            index_y1=1, # index for y1Plus
            index_y4=3  # index for y4Plus
        )
        save_steady_states_plot(plots_dir, fig, 'fb_gain')
        
        # Add V1-V2 comparison plot for feedback gain
        plot_v1_v2_comparison(steady_states, c_vals, 1.0, plots_dir, gain_type='feedback')
    
    # Load beta1 gain data
    data_file = os.path.join(data_dir, 'steady_states_input_gain_beta1_all.npy')
    if os.path.exists(data_file):
        print(f"Loading beta1 gain data from: {data_file}")
        steady_states = np.load(data_file, allow_pickle=True).item()
        
        # Process data to only keep center neuron data (N//2)
        first_value = next(iter(steady_states.values()), None)
        if first_value and isinstance(first_value, list) and isinstance(first_value[0], np.ndarray):
            N = 36  # Assuming 36 neurons for spatial data
            center_idx = N // 2
            steady_states = {
                key: [v[center_idx] for v in value]
                for key, value in steady_states.items()
            }
        
        # Extract beta1 and contrast values from the data
        # beta1_vals = sorted(list(set([b for b, _ in steady_states.keys()])))
        beta1_vals = np.array(sorted(list(set([b for b, _ in steady_states.keys()]))))
        c_vals = np.array(sorted(list(set([c for _, c in steady_states.keys()]))))
        print(f"Found beta1 values: {beta1_vals}")
        print(f"Found contrast values: {c_vals}")
        
        fig, axs = plot_steady_states(
            steady_states=steady_states,
            c_vals=c_vals,
            gamma_vals=beta1_vals,  # Use beta1_vals instead of gamma_vals
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            index_y1=1, # index for y1Plus
            index_y4=3  # index for y4Plus
        )
        save_steady_states_plot(plots_dir, fig, 'input_gain_beta1')
        
        # Add V1-V2 comparison plot for input gain
        # plot_v1_v2_comparison(steady_states, c_vals, plots_dir, gain_type='input')

def save_steady_states_plot(plots_dir, fig, gain_type):
    """Save the steady states plot to file."""
    save_path = os.path.join(plots_dir, f'steady_states_{gain_type}.pdf')
    fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")
    
def plot_steady_states(steady_states, c_vals, gamma_vals, fb_gain, input_gain_beta1, input_gain_beta4, index_y1, index_y4):
    """
    Plot steady states using global rcParams settings.
    """
    # Figure is created using the global figsize from rcParams
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=False)
    
    # Convert input values to numpy arrays with specific dtype
    c_vals = np.array(c_vals, dtype=np.float64)
    gamma_vals = np.array(gamma_vals, dtype=np.float64)
    
    # Create a list to store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # First pass to find maximum values for normalization
    max_y1 = 0
    max_y4 = 0
    for gamma in gamma_vals:
        for c in c_vals:
            for key in steady_states.keys():
                if (np.isclose(key[0], gamma, rtol=1e-14, atol=1e-14) and 
                    np.isclose(key[1], c, rtol=1e-14, atol=1e-14)):
                    max_y1 = max(max_y1, steady_states[key][index_y1])
                    max_y4 = max(max_y4, steady_states[key][index_y4])
    
    for gamma in gamma_vals:
        steady_states_gamma = []
        valid_contrasts = []
        
        # Find matching keys with tolerance
        for c in c_vals:
            found = False
            for key in steady_states.keys():
                if (np.isclose(key[0], gamma, rtol=1e-14, atol=1e-14) and 
                    np.isclose(key[1], c, rtol=1e-14, atol=1e-14)):
                    steady_states_gamma.append(steady_states[key])
                    valid_contrasts.append(c)
                    found = True
                    break
            if not found:
                print(f"Warning: No match found for gamma={gamma}, c={c}")

        if not steady_states_gamma:
            print(f"No data found for gamma={gamma}")
            continue

        # Convert to numpy arrays for plotting and normalize
        y1_values = np.array([state[index_y1] for state in steady_states_gamma]) / max_y1
        y4_values = np.array([state[index_y4] for state in steady_states_gamma]) / max_y4
        x_values = np.array(valid_contrasts) * 100
        
        # Sort arrays by x_values
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y1_values = y1_values[sort_idx]
        y4_values = y4_values[sort_idx]
        
        # Plot V1 and V4 firing rates
        line1 = axs[0].plot(x_values, y1_values, '-')[0]
        axs[1].plot(x_values, y4_values, '-', color=line1.get_color())
        
        # Store handle and label for legend
        legend_handles.append(line1)
        legend_labels.append(f'{gamma}')

    # Set scales and labels
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_ylim(-0.05, 1.05)  # Set y-axis limit from 0 to slightly above 1
        
        ax.set_yticks(np.arange(0, 1.1, 0.50))
        ax.set_yticklabels(['0', '0.5', '1.0'])
        
    # Set labels
    axs[0].set_xlabel('Contrast (%)')
    axs[1].set_xlabel('Contrast (%)')
    axs[0].set_ylabel('Normalized Firing Rate')
    # axs[1].set_ylabel('Relative firing rate')
    
    # Set titles
    axs[0].set_title('V1')
    axs[1].set_title('V2')
    
    # Create legend with parameter name as title
    if fb_gain:
        param_name = 'Feedback gain'
    elif input_gain_beta1:
        param_name = 'Input gain'
    elif input_gain_beta4:
        param_name = 'Input gain'
    
    # Add legend with title
    legend = axs[0].legend(legend_handles, legend_labels,bbox_to_anchor=(-0.025, 1),
                           loc='upper left', 
                          title=param_name,
                          title_fontsize=plt.rcParams['legend.fontsize'])
    
    # Layout handling for cluster environment without tight_layout support
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, axs

def plot_v1_v2_comparison(steady_states, c_vals, gamma_val, plots_dir, gain_type='feedback'):
    """
    Plot V1 and V2 responses for gamma/beta = gamma_val on the same plot.
    
    Args:
        steady_states (dict): Dictionary containing steady state data
        c_vals (array): Array of contrast values
        gamma_val (float): Value of gamma or beta to plot
        plots_dir (str): Directory to save the plot
        gain_type (str): Type of gain ('feedback' or 'input')
    """
    # Create figure using square figsize
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Set scaled parameters for this figure
    ax.tick_params(axis='both', which='major')   
    ax.tick_params(axis='both', which='minor')
    
    # Rest of the data preparation remains the same
    c_vals = np.array(c_vals, dtype=np.float64)
    param_value = np.float64(gamma_val)
    
    steady_states_param = []
    valid_contrasts = []
    
    # Find matching keys with tolerance
    for c in c_vals:
        found = False
        for key in steady_states.keys():
            if (np.isclose(key[0], param_value, rtol=1e-14, atol=1e-14) and 
                np.isclose(key[1], c, rtol=1e-14, atol=1e-14)):
                steady_states_param.append(steady_states[key])
                valid_contrasts.append(c)
                found = True
                break
        if not found:
            print(f"Warning: No match found for {'gamma' if gain_type == 'feedback' else 'beta'}={param_value}, c={c}")
    
    if not steady_states_param:
        print(f"No data found for {'gamma' if gain_type == 'feedback' else 'beta'}={param_value}")
        return
    
    # Data processing remains the same
    y1_values = np.array([state[1] for state in steady_states_param])
    y4_values = np.array([state[3] for state in steady_states_param])
    x_values = np.array(valid_contrasts) * 100
    
    sort_idx = np.argsort(x_values)
    x_values = x_values[sort_idx]
    y1_values = y1_values[sort_idx]
    y4_values = y4_values[sort_idx]
    
    # Normalize V1 and V2 firing rates to their maximum values
    y1_values = y1_values / np.max(y1_values)
    y4_values = y4_values / np.max(y4_values)
    
    # Get color for gamma = 1.0
    gamma_values = np.array([0.25, 0.5, 0.75, 1.0])
    positions = (gamma_values - gamma_values.min()) / (gamma_values.max() - gamma_values.min())
    positions = 0.3 + positions * 0.7
    cmap = matplotlib.colormaps.get_cmap('Blues')
    gamma_1_color = cmap(positions[-1])
    
    # Plot V1 with more frequent dashes and V2 with solid line
    line1 = ax.plot(x_values, y1_values, '--', color=gamma_1_color, 
                     )[0]
    line2 = ax.plot(x_values, y4_values, '-', color=gamma_1_color
                    )[0]
    
    legend_handles = []
    legend_labels = []
    
    legend_handles.extend([line1, line2])
    legend_labels.extend(['V1', 'V2'])
    
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'])
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_ylim(-0.05, 1.05)  # Set y-axis limit from 0 to slightly above 1
    ax.set_xlim(1, 100)
    ax.set_yticks(np.array([0.0,0.5,1.0]))
    ax.set_yticklabels(['0.0', '0.5', '1.0'])
    
    # Set labels with scaled font size
    ax.set_xlabel('Contrast (%)')
    ax.set_ylabel('Normalized Firing Rate')
    
    # Add legend with scaled font sizes
    legend = ax.legend(legend_handles, legend_labels,
                      loc='upper left',
                      fontsize=plt.rcParams['legend.fontsize'],
                      )  # Increased spacing between entries
    
    # Scale down tick parameters and spine widths
    ax.tick_params(axis='both', which='major' )
    ax.tick_params(axis='both', which='minor')
    
    # Layout handling
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    # Save the plot
    filename = f'v1_v2_comparison_{"fb" if gain_type == "feedback" else "input"}_gain_{gamma_val:.2f}.pdf'
    save_path = os.path.join(plots_dir, filename)
    fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"Saved V1-V2 comparison plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_gain_modulation.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_gain_modulation_results(results_dir)