import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params
setup_plot_params()

# Updated rcParams to match Plot_freq_wise_decom.py style
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.2,
    "legend.labelspacing": 0.1,
})

def plot_dim_vs_freq(dimension_data, freq, gamma, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Plot dimension versus frequency with a style similar to plot_pred_perf_vs_freq.
    Plots SEM instead of STD.
    """
    fig, ax = plt.subplots()
    
    # Dynamically create colors based on the number of contrast values
    contrast_values = np.array(contrast_vals)
    # Normalize contrast values to [0,1] range for colormap, handling the case of a single contrast
    if contrast_values.size > 1:
        positions = (contrast_values - contrast_values.min()) / (contrast_values.max() - contrast_values.min())
    else:
        positions = np.array([0.5]) # A default position if there's only one value
    
    # Adjust the range of the colormap (0.2 to 0.8 for grays)
    positions = 0.2 + positions * 0.6
    
    cmap = matplotlib.colormaps.get_cmap('Reds')
    colors = [cmap(pos) for pos in positions]
    ax.set_prop_cycle(color=colors)
    
    custom_handles = []
    
    # Create line styles for V1 and V4
    linestyles = {'V4': '-'}  # Solid for V1, Dashed for V4
    
    for i, contrast in enumerate(contrast_vals):
        key = (gamma, contrast)
        if key not in dimension_data:
            print(f"Warning: Data for gamma={gamma}, contrast={contrast} not found in dimension_data. Skipping.")
            continue
            
        data = dimension_data[key]
        # color = param_colors[i % len(param_colors)] # Original color assignment

        sample_size = 200
        # Retrieve V1 and V4 dimension data
        # V1_mean = data['V1']['mean']
        # V1_std = data['V1']['std']
        # V1_sem = V1_std / np.sqrt(sample_size) 
        
        V4_mean = data['V4']['mean']
        V4_std = data['V4']['std']
        V4_sem = V4_std / np.sqrt(sample_size) 
        
        # Fill between for V1 data using SEM
        # ax.fill_between(freq, V1_mean - V1_sem, V1_mean + V1_sem, 
        #                color=color, alpha=0.1)
        # ax.plot(freq, V1_mean, linestyle=linestyles['V1'], 
        #         color=color)
        
        # Fill between for V4 data using SEM
        ax.fill_between(freq, V4_mean - V4_sem, V4_mean + V4_sem, 
                       alpha=0.1)
        ax.plot(freq, V4_mean, linestyle=linestyles['V4']
                )
        
        # Determine label based on which gain is being varied
        if fb_gain:
            label = f'gamma_1={gamma:.2f}'
        elif input_gain_beta1:
            label = f'beta_1={gamma:.2f}'
        elif input_gain_beta4:
            label = f'beta_4={gamma:.2f}'
        else: # Default label if no specific gain is identified
            label = f'Param={gamma:.2f}'

        custom_handles.append(mpatches.Patch(color=colors[i], label=label))
    
    # First legend for gamma/beta values (styled like Plot_freq_wise_decom.py)
    # legend1 = ax.legend(handles=custom_handles, 
    #                    loc='upper right', frameon=False,
    #                    fontsize=plt.rcParams['legend.fontsize'])
    # ax.add_artist(legend1)
    
    # Second legend for line styles (styled like Plot_freq_wise_decom.py)
    # line_legend = [
    #     # mlines.Line2D([0], [0], color='black', linestyle=linestyles['V1'],
    #     #              label='V1 Dimension'),
    #     mlines.Line2D([0], [0], color='black', linestyle=linestyles['V4'],
    #                  label='V1-V2')
    # ]
    # legend2 = ax.legend(handles=line_legend, loc='lower left',
    #                    
    #                    fontsize=plt.rcParams['legend.fontsize'],
    #                    frameon=False)
    # ax.add_artist(legend2)
    
    # Set labels and ticks (styled like Plot_freq_wise_decom.py)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Dimensionality')
    ax.tick_params(axis='both', which='major', pad=10)
    
    # Set x-axis to log scale and y-axis limits
    # ax.set_xscale('log')

    # Set axis limits and ticks
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)  # Minor ticks every 10 Hz
    ax.set_xticklabels([str(x) for x in xticks])

    # Set y-axis limits and ticks to match Plot_raw_V1V2_coherence.py
    ax.set_ylim(0.5,3.25)
    yticks = np.array([ 1,  2,  3])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])

    # Add tick padding to match Plot_raw_V1V2_coherence.py
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

    return fig, ax

def plot_dimension_frequency_analysis(results_dir):
    """
    Plot dimension vs. frequency analysis results.
    
    Args:
        results_dir (str): Path to the results directory containing data
    """
    data_dir = os.path.join(results_dir, 'Data')
    plots_dir = os.path.join(results_dir, 'Plots', 'DimensionFrequency')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Results directory for Dimension vs. Freq plots: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plots will be saved to: {plots_dir}")

   

    data_file = os.path.join(data_dir, f'Dimension_data_fb_gain.npy')
        
    if os.path.exists(data_file):
        loaded_data = np.load(data_file, allow_pickle=True).item()
        
        dimension_data = loaded_data.get('dimension_data')
        freq = loaded_data.get('freq')
        
        # Extract parameter values (gamma_vals or beta_vals etc.) and contrast values
        gamma = sorted(list(set(k[0] for k in dimension_data.keys())))[0] # e.g., gamma values
        contrast_vals = sorted(list(set(k[1] for k in dimension_data.keys()))) # contrast values
        
        print(f"Gamma values found: {gamma}")
        print(f"Contrast values found: {contrast_vals}")
        
        fig, ax = plot_dim_vs_freq(
            dimension_data=dimension_data,
            freq=freq,
            gamma=gamma,
            contrast_vals=contrast_vals,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False
        )
        
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        save_path = os.path.join(plots_dir, f'Dimension_vs_Freq_fb_gain.pdf')
        fig.savefig(save_path, dpi=400, format='pdf')
        plt.close(fig)
        print(f"Saved plot to: {save_path}")
    else:
        print(f"Data file not found for fb_gain: {data_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_dim_vs_freq.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir_arg = sys.argv[1]
    print(f"Processing Dimension vs. Frequency results from: {results_dir_arg}")
    plot_dimension_frequency_analysis(results_dir_arg) 