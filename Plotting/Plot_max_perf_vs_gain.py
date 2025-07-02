import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Add project root to Python path if necessary (adjust based on your structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Plotting import setup_plot_params

# Define the gain values explicitly
GAMMA_VALS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
BETA1_VALS = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

# Create color mapping function (similar to Plot_communication_analysis.py)
def get_colors_for_gains(gain_values):
    """Get colors for gain values using Blues colormap."""
    if len(gain_values) == 1:
        # If only one value, use a mid-range blue
        return [plt.cm.Blues(0.7)]
    else:
        # Map multiple values to color range [0.3, 1.0]
        min_val, max_val = gain_values.min(), gain_values.max()
        if max_val == min_val: # Avoid division by zero if all values are the same
            positions = np.full_like(gain_values, 0.5)
        else:
            positions = (gain_values - min_val) / (max_val - min_val)
        positions = 0.3 + positions * 0.7  # Map to range [0.3, 1.0]
        return [plt.cm.Blues(pos) for pos in positions]

# Set up plotting parameters for journal-quality figures
setup_plot_params()
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
})

def get_max_performance(performance_data, gain_vals, contrast):
    """
    Extracts the maximum prediction performance for V1-V1 and V1-V2
    across dimensions for each gain value.
    """
    max_perf_v1v1 = []
    max_perf_v1v2 = []

    for gain in gain_vals:
        if (gain, contrast) in performance_data:
            data = performance_data[(gain, contrast)]
            v1_mean = data['V1']['mean']
            v2_mean = data['V4']['mean']

            max_perf_v1v1.append(np.max(v1_mean) if len(v1_mean)>0 else np.nan)
            max_perf_v1v2.append(np.max(v2_mean) if len(v2_mean)>0 else np.nan)
        else:
            max_perf_v1v1.append(np.nan)
            max_perf_v1v2.append(np.nan)

    return np.array(max_perf_v1v1), np.array(max_perf_v1v2)

def plot_max_perf_vs_gain(performance_data, gain_vals, contrast, gain_type):
    """
    Plots maximum prediction performance versus gain values.
    """
    fig, ax = plt.subplots()

    max_v1v1, max_v1v2 = get_max_performance(performance_data, gain_vals, contrast)
    colors = get_colors_for_gains(gain_vals)

    markers = {'V1': 's', 'V2': 'o'} 
    line_styles = {'V1': (0, (2, 2)), 'V2': '-'}

    for i in range(len(gain_vals) - 1):
        ax.plot(gain_vals[i:i+2], max_v1v1[i:i+2], linestyle=line_styles['V1'], color=colors[i])
        ax.plot(gain_vals[i:i+2], max_v1v2[i:i+2], linestyle=line_styles['V2'], color=colors[i])

    for i in range(len(gain_vals)):
        ax.plot(gain_vals[i], max_v1v1[i], marker=markers['V1'], color=colors[i],
                markeredgecolor='black', linestyle='none', markersize=30)
        ax.plot(gain_vals[i], max_v1v2[i], marker=markers['V2'], color=colors[i],
                markeredgecolor='black', linestyle='none', markersize=30)

    legend_color = colors[-1]
    marker_legend = [
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V2'],
                     linestyle=line_styles['V2'], linewidth=4, markersize=30,
                     label='V1-V2'),
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V1'],
                     linestyle=line_styles['V1'], linewidth=4, markersize=30,
                     label='V1-V1')
    ]
    ax.legend(handles=marker_legend, loc='best', frameon=False)

    if gain_type == 'feedback':
        ax.set_xlabel('Feedback Gain')
        ax.set_xlim(-0.05, 1.05)
    elif gain_type == 'input_beta1':
        ax.set_xlabel('Input Gain')
        ax.set_xlim(0.45, 1.55)

    ax.set_ylabel('Max Pred. Perf.')
    ax.set_xticks(gain_vals)
    ax.set_xticklabels([f'{x:.2f}' for x in gain_vals])

    ax.set_ylim(0.003, 0.48)
    ax.set_yscale('log')
    yticks = np.array([0.01,0.1])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.2f}' for y in yticks])
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

    return fig, ax

def main():
    if len(sys.argv) != 2:
        print("Usage: python Plot_max_perf_vs_gain.py /path/to/results_dir")
        sys.exit(1)

    results_dir = sys.argv[1]
    data_dir = os.path.join(results_dir, 'Data')
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication_MaxPerf')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plots will be saved in: {plots_dir}")

    fb_gain_file = os.path.join(data_dir, 'Communication_data_fb_gain_all.npy')
    if os.path.exists(fb_gain_file):
        print(f"\nProcessing Feedback Gain data from: {fb_gain_file}")
        comm_data_fb = np.load(fb_gain_file, allow_pickle=True).item()
        
        c_vals_fb = sorted(list(set(c for _, c in comm_data_fb.keys())))
        print(f"Contrast values found: {c_vals_fb}")

        for contrast in c_vals_fb:
            print(f"Creating Feedback Gain plot for contrast = {contrast}")
            fig, ax = plot_max_perf_vs_gain(
                comm_data_fb,
                GAMMA_VALS,
                contrast,
                gain_type='feedback'
            )
            
            if fig:
                save_path = os.path.join(plots_dir, f'MaxPerf_vs_FeedbackGain_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    else:
        print(f"Feedback gain data file not found: {fb_gain_file}")

    input_gain_beta1_file = os.path.join(data_dir, 'Communication_data_input_gain_beta1_all.npy')
    if os.path.exists(input_gain_beta1_file):
        print(f"\nProcessing Input Gain (beta1) data from: {input_gain_beta1_file}")
        comm_data_beta1 = np.load(input_gain_beta1_file, allow_pickle=True).item()

        c_vals_beta1 = sorted(list(set(c for _, c in comm_data_beta1.keys())))
        print(f"Contrast values found: {c_vals_beta1}")

        for contrast in c_vals_beta1:
            print(f"Creating Input Gain (beta1) plot for contrast = {contrast}")
            fig, ax = plot_max_perf_vs_gain(
                comm_data_beta1,
                BETA1_VALS,
                contrast,
                gain_type='input_beta1'
            )
            
            if fig:
                save_path = os.path.join(plots_dir, f'MaxPerf_vs_InputGainBeta1_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    else:
        print(f"Input gain (beta1) data file not found: {input_gain_beta1_file}")

if __name__ == "__main__":
    main()
