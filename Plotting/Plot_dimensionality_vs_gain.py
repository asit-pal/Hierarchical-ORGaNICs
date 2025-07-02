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

# Create color mapping function
def get_colors_for_gains(gain_values):
    """Get colors for gain values using Blues colormap."""
    if len(gain_values) == 1:
        return [plt.cm.Blues(0.7)]
    else:
        positions = (gain_values - gain_values.min()) / (gain_values.max() - gain_values.min())
        positions = 0.3 + positions * 0.7  # Map to range [0.3, 1.0]
        return [plt.cm.Blues(pos) for pos in positions]

# Set up plotting parameters
setup_plot_params()
plt.rcParams.update({
    "lines.linewidth": 12,
    "lines.markersize": 10,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
})

def get_dimensionality(performance_data, gain_vals, contrast, threshold=.999):
    """
    Calculates the dimensionality required to reach a threshold percentage
    of the maximum performance for V1-V1 and V1-V4 (treated as V1-V2).
    """
    dim_v1v1 = []
    dim_v1v2 = [] # Naming consistency with user's latest plot code

    for gain in gain_vals:
        dim_v1_found = np.nan
        dim_v2_found = np.nan
        
        if (gain, contrast) in performance_data:
            data = performance_data[(gain, contrast)]
            
            # Get V1 data
            v1_data = data.get('V1', None)
            if v1_data:
                v1_mean = v1_data.get('mean', [])
                v1_dims = v1_data.get('dims', np.arange(1, len(v1_mean) + 1))
                if len(v1_mean) > 0:
                    max_perf_v1 = np.max(v1_mean)
                    target_perf_v1 = threshold * max_perf_v1
                    # Find the first dimension index where performance meets or exceeds target
                    indices_v1 = np.where(v1_mean >= target_perf_v1)[0]
                    if len(indices_v1) > 0:
                        # Get the corresponding dimension value
                        dim_v1_found = v1_dims[indices_v1[0]]
                        
            # Get V4 data (treating as V2 for consistency)
            v4_data = data.get('V4', None) # Accessing V4 key
            if v4_data:
                v4_mean = v4_data.get('mean', [])
                v4_dims = v4_data.get('dims', np.arange(1, len(v4_mean) + 1))
                if len(v4_mean) > 0:
                    max_perf_v4 = np.max(v4_mean)
                    target_perf_v4 = threshold * max_perf_v4
                    # Find the first dimension index where performance meets or exceeds target
                    indices_v4 = np.where(v4_mean >= target_perf_v4)[0]
                    if len(indices_v4) > 0:
                        # Get the corresponding dimension value
                        dim_v2_found = v4_dims[indices_v4[0]] # Storing as v2 dim
                        
        dim_v1v1.append(dim_v1_found)
        dim_v1v2.append(dim_v2_found)
            
    return np.array(dim_v1v1), np.array(dim_v1v2)

def plot_dimensionality_vs_gain(performance_data, gain_vals, contrast, gain_type):
    """
    Plots dimensionality (99% threshold) versus gain values.
    """
    fig, ax = plt.subplots()
    
    dim_v1v1, dim_v1v2 = get_dimensionality(performance_data, gain_vals, contrast, threshold=0.999)   
    colors = get_colors_for_gains(gain_vals)
    
    markers = {'V1': 's', 'V2': 'o'} 
    
    for i in range(len(gain_vals) - 1):
        ax.plot(gain_vals[i:i+2], dim_v1v1[i:i+2], linestyle='--', color=colors[i], linewidth=4)
        ax.plot(gain_vals[i:i+2], dim_v1v2[i:i+2], linestyle='--', color=colors[i], linewidth=4)

    for i in range(len(gain_vals)):
        ax.plot(gain_vals[i], dim_v1v1[i], marker=markers['V1'], color=colors[i], 
                markeredgecolor='black', linestyle='none', markersize=30)
        ax.plot(gain_vals[i], dim_v1v2[i], marker=markers['V2'], color=colors[i], 
                markeredgecolor='black', linestyle='none', markersize=30)

    legend_color = colors[-1] 
    marker_legend = [
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V2'],
                     linestyle='--', linewidth=4,markersize=30,
                     label='V1-V2'), 
        mlines.Line2D([0], [0], color=legend_color, marker=markers['V1'],
                     linestyle='--', linewidth=4,markersize=30,
                     label='V1-V1')
    ]
    ax.legend(handles=marker_legend, loc='best', frameon=False)
    
    if gain_type == 'feedback':
        ax.set_xlabel('Feedback Gain')
    elif gain_type == 'input_beta1':
        ax.set_xlabel('Input Gain')
        
    ax.set_ylabel('Dimensionality')

    ax.set_ylim(3.5,7.5) 
    ax.set_yticks([4,6])
    ax.set_yticklabels(['4','6'])
    ax.set_yticks([5,7], minor=True)
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)
    ax.set_xticks(gain_vals)
    ax.set_xticklabels([f'{x:.2f}' for x in gain_vals])

    return fig, ax

def main():
    if len(sys.argv) != 2:
        print("Usage: python Plot_dimensionality_vs_gain.py /path/to/results_dir")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    data_dir = os.path.join(results_dir, 'Data')
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication_Dimensionality')
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
            print(f"Creating Feedback Gain Dimensionality plot for contrast = {contrast}")
            fig, ax = plot_dimensionality_vs_gain(
                comm_data_fb,
                GAMMA_VALS,
                contrast,
                gain_type='feedback'
            )
            
            if fig:
                save_path = os.path.join(plots_dir, f'Dimensionality_vs_FeedbackGain_contrast_{contrast}.pdf')
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
            print(f"Creating Input Gain (beta1) Dimensionality plot for contrast = {contrast}")
            fig, ax = plot_dimensionality_vs_gain(
                comm_data_beta1,
                BETA1_VALS,
                contrast,
                gain_type='input_beta1'
            )
            
            if fig:
                save_path = os.path.join(plots_dir, f'Dimensionality_vs_InputGainBeta1_contrast_{contrast}.pdf')
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
                fig.savefig(save_path, dpi=400, format='pdf')
                plt.close(fig)
                print(f"Saved plot to: {save_path}")
    else:
        print(f"Input gain (beta1) data file not found: {input_gain_beta1_file}")

if __name__ == "__main__":
    main()
