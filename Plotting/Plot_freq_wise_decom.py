import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from Plotting import setup_plot_params

# Add project root to Python path if necessary (adjust based on your structure)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

setup_plot_params()

# Create a color gradient using Reds colormap with contrast-based spacing
contrast_values = np.array([2.5, 3.7, 6.1, 9.7, 16.3, 35.9, 50.3, 72.0]) # Assuming these are the contrasts you have
# Normalize contrast values to [0,1] range for colormap
positions = (contrast_values - contrast_values.min()) / (contrast_values.max() - contrast_values.min())
# Adjust the range of the colormap
positions = 0.2 + positions * 0.6  # This maps positions to range [0.2, 0.8]
cmap = matplotlib.colormaps.get_cmap('Reds')
colors_contrast = [cmap(pos) for pos in positions]

# Updated rcParams to match Plot_coherence_fixed_gamma.py style
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.2,
    "legend.labelspacing": 0.1,
    "axes.prop_cycle": plt.cycler(color=colors_contrast)
})

def plot_pred_perf_vs_freq(performance_data, freq, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Plot prediction performance versus frequency. Uses specific colors for gamma/beta values.
    Style adapted from Plot_coherence_fixed_gamma.py. Legends moved to upper right.
    """
    param_colors = ['#DC143C', '#00BFFF', '#32CD32', '#000000']
    fig, ax = plt.subplots()

    custom_handles = []

    linestyles = {'V1_V1': '-', 'V1_V2': '--'}

    for i, gamma in enumerate(gamma_vals):
        key = (gamma, contrast)
        if key not in performance_data:
            print(f"Warning: Data for key {key} not found. Skipping.")
            continue

        data = performance_data[key]
        color = param_colors[i % len(param_colors)]
        sample_size = 200

        # Use V1_V1 and V1_V4 data instead of V1 and V4
        V1_V1_mean = data['V1_V1']['mean']
        V1_V1_std = data['V1_V1']['std']
        V1_V1_sem = V1_V1_std / np.sqrt(sample_size)
        V1_V2_mean = data['V1_V2']['mean']
        V1_V2_std = data['V1_V2']['std']
        V1_V2_sem = V1_V2_std / np.sqrt(sample_size)

        # Plot V1-V1 communication
        ax.fill_between(freq, V1_V1_mean - V1_V1_sem, V1_V1_mean + V1_V1_sem,
                       color=color, alpha=0.1)
        ax.plot(freq, V1_V1_mean, linestyle=linestyles['V1_V1'],
                color=color)

        # Plot V1-V4 communication
        ax.fill_between(freq, V1_V2_mean - V1_V2_sem, V1_V2_mean + V1_V2_sem,
                       color=color, alpha=0.1)
        ax.plot(freq, V1_V2_mean, linestyle=linestyles['V1_V2'],
                color=color)

        if fb_gain:
            label = f'gamma_1={gamma:.2f}'
        elif input_gain_beta1:
            label = f'beta_1={gamma:.2f}'
        elif input_gain_beta4:
            label = f'beta_4={gamma:.2f}'
        else:
             label = f'Param={gamma:.2f}'

        custom_handles.append(mpatches.Patch(color=color, label=label))

    legend1 = ax.legend(handles=custom_handles, fontsize=plt.rcParams['legend.fontsize'],
                       loc='upper right', frameon=False)
    ax.add_artist(legend1)

    line_legend = [
        mlines.Line2D([0], [0], color='black', linestyle=linestyles['V1_V1'],
                     label='V1-V1'),
        mlines.Line2D([0], [0], color='black', linestyle=linestyles['V1_V2'],
                     label='V1-V2')
    ]
    legend2 = ax.legend(handles=line_legend, loc='upper right',
                       bbox_to_anchor=(0.98, 0.85),
                       fontsize=plt.rcParams['legend.fontsize'],
                       frameon=False)
    ax.add_artist(legend2)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Prediction Performance')
    ax.tick_params(axis='both', which='major')

    # ax.set_xscale('log')
    ax.set_ylim(bottom=0)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, ax

def plot_V1V4_pred_perf_vs_freq_by_contrast(performance_data, freq, gamma, contrasts, fb_gain, input_gain_beta1, input_gain_beta4):
    """
    Plot V1-V4 prediction performance versus frequency for different contrasts at a fixed gamma value.
    Uses 'Reds' colormap defined in rcParams. Legend moved to upper right.
    Style adapted from Plot_coherence_fixed_gamma.py.
    """
    fig, ax = plt.subplots()

    legend_handles = []
    legend_labels = []

    for contrast in contrasts:
        key = (gamma, contrast)
        if key not in performance_data:
            print(f"Warning: Data for key {key} not found. Skipping.")
            continue

        data = performance_data[key]

        # Use V1_V4 data instead of V4
        V1_V2_mean = data['V1_V2']['mean']
        V1_V2_std = data['V1_V2']['std']

        line, = ax.plot(freq, V1_V2_mean, label=f'{(contrast * 100):.1f}')
        line_color = line.get_color()
        ax.fill_between(freq, V1_V2_mean - 0.5*V1_V2_std, V1_V2_mean + 0.5*V1_V2_std,
                       color=line_color, alpha=0.2)
        legend_handles.append(line)
        legend_labels.append(f'{(contrast * 100):.1f}')

    ax.legend(legend_handles, legend_labels,
              title='Contrast (%)',
              title_fontsize=plt.rcParams['legend.fontsize'],
              loc='upper right', frameon=False)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Prediction Performance')
    ax.tick_params(axis='both', which='major')

    ax.set_xlim(1, 100)
    xticks = np.array([1, 10, 100])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    
    ax.set_ylim(bottom=0)
    ax.set_xscale('log')

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, ax

def plot_frequency_analysis(results_dir):
    """
    Plot frequency analysis results for both feedback gain and input gain beta1.
    Style adapted from Plot_coherence_fixed_gamma.py.
    """
    data_dir = os.path.join(results_dir, 'Data')
    
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    plots_dir = os.path.join(results_dir, 'Plots', 'Frequency')
    os.makedirs(plots_dir, exist_ok=True)
    
    fb_gain_file = os.path.join(data_dir, 'Frequency_data_fb_gain.npy')
    if os.path.exists(fb_gain_file):
        print(f"Loading feedback gain data from: {fb_gain_file}")
        loaded_data = np.load(fb_gain_file, allow_pickle=True).item()
        
        performance_data = loaded_data['performance_data']
        freq = loaded_data['freq']
        
        gamma_vals = sorted(set(g for g, _ in performance_data.keys()))
        c_vals = sorted(set(c for _, c in performance_data.keys()))
        print(f"\nGamma values found in data: {gamma_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        # for contrast in c_vals:
        #     print(f"Creating feedback gain plot for contrast = {contrast}")
            
        #     fig, ax = plot_pred_perf_vs_freq(
        #         performance_data=performance_data,
        #         freq=freq,
        #         gamma_vals=gamma_vals,
        #         contrast=contrast,
        #         fb_gain=True,
        #         input_gain_beta1=False,
        #         input_gain_beta4=False
        #     )
            
        #     save_path = os.path.join(plots_dir, f'Frequency_analysis_fb_gain_contrast_{contrast}.pdf')
        #     fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
        #     plt.close(fig)
        #     print(f"Saved plot to: {save_path}")
            
        for gamma in gamma_vals:
            print(f"Creating V1-V4 plot for gamma = {gamma}")
            
            fig, ax = plot_V1V4_pred_perf_vs_freq_by_contrast(
                performance_data=performance_data,
                freq=freq,
                gamma=gamma,
                contrasts=c_vals,
                fb_gain=True,
                input_gain_beta1=False,
                input_gain_beta4=False
            )
            
            save_path = os.path.join(plots_dir, f'V1V4_Frequency_analysis_fb_gain_gamma_{gamma}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")
    
    input_gain_beta1_file = os.path.join(data_dir, 'Frequency_data_input_gain_beta1.npy')
    if os.path.exists(input_gain_beta1_file):
        print(f"Loading input gain beta1 data from: {input_gain_beta1_file}")
        loaded_data = np.load(input_gain_beta1_file, allow_pickle=True).item()
        
        performance_data = loaded_data['performance_data']
        freq = loaded_data['freq']
        
        beta1_vals = sorted(set(b for b, _ in performance_data.keys()))
        c_vals = sorted(set(c for _, c in performance_data.keys()))
        print(f"\nBeta1 values found in data: {beta1_vals}")
        print(f"Contrast values found in data: {c_vals}")
        
        for contrast in c_vals:
            print(f"Creating input gain beta1 plot for contrast = {contrast}")
            
            fig, ax = plot_pred_perf_vs_freq(
                performance_data=performance_data,
                freq=freq,
                gamma_vals=beta1_vals,
                contrast=contrast,
                fb_gain=False,
                input_gain_beta1=True,
                input_gain_beta4=False
            )
            
            save_path = os.path.join(plots_dir, f'Frequency_analysis_input_gain_beta1_contrast_{contrast}.pdf')
            fig.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
            plt.close(fig)
            print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_freq_wise_decom.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_frequency_analysis(results_dir)
