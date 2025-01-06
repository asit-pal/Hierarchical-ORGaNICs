import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from Plotting import plot_pred_perf_vs_dim
# %matplotlib inline
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     "font.size": 36,  # Set a consistent font size for all text in the plot
# })
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



def plot_feedback_gain_results(results_dir):
    """
    Plot feedback gain results for communication analysis.
    
    Args:
        results_dir (str): Path to the results directory containing config and data
        gain_type (str): Type of gain ('fb_gain' or 'input_gain_beta1' or 'input_gain_beta4')
    """
    # Get the data directory path   
    data_dir = os.path.join(results_dir, 'Data')
    
    # Debug prints
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    
    #Load config file from results directory
    config_file = [f for f in os.listdir(results_dir) if f.endswith('.yaml')]
    if not config_file:
        print(f"Error: No yaml config file found in {results_dir}")
        sys.exit(1)
        
    config_path = os.path.join(results_dir, config_file[0])
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # get parameters from config file
    fb_config = config['Communication']['Feedback_gain']
    gamma_vals = fb_config['gamma_vals']
    c_vals = fb_config['c_vals']
    
    if fb_config['enabled']:
        gain_type = 'fb_gain'
    
    
    print(f"Processing data for contrasts: {c_vals}")
    print(f"With gamma values: {gamma_vals}")
    
    # Load communication data from the single file
    communication_file = os.path.join(data_dir, f'Communication_data_{gain_type}_all.npy')
    if not os.path.exists(communication_file):
        print(f"Error: Communication data file not found at {communication_file}")
        sys.exit(1)
        
    print(f"Loading communication data from: {communication_file}")
    
    Communication_data = np.load(communication_file, allow_pickle=True).item()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, 'Plots', 'Communication')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create plots for each contrast value
    for contrast in c_vals:
        print(f"Creating plot for contrast = {contrast}")
        
        # Create plots
        fig, axs = plot_pred_perf_vs_dim(
            Communication_data,
            gamma_vals,
            contrast,
            fb_gain=fb_config['enabled'],
            input_gain_beta1=config['Communication']['Input_gain_beta1']['enabled'],
            input_gain_beta4=False
        )
        
        # Save the figure
        Save_communication_plots(plots_dir,fig,contrast,gain_type)

def Save_communication_plots(plots_dir,fig,contrast,gain_type):
    save_path = os.path.join(plots_dir, f'Communication_analysis_{gain_type}_contrast_{contrast}.pdf')
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

# def plot_input_gain_beta1_results(results_dir):
#     data_dir = os.path.join(results_dir, 'Data', 'Communication_data')
#     beta1_files = [f for f in os.listdir(data_dir) if f.startswith('Communication_input_gain_beta1')]
    
#     for file in beta1_files:
#         # Load data
#         data_path = os.path.join(data_dir, file)
#         Communication_data = np.load(data_path, allow_pickle=True).item()
        
#         # Extract beta1 values and contrast from the data
#         beta1_vals = sorted(list(set([key[0] for key in Communication_data.keys()])))
#         contrast = list(set([key[1] for key in Communication_data.keys()]))[0]  # Should be same for all
        
#         # Create plots
#         fig, axs = plot_pred_perf_vs_dim(
#             Communication_data,
#             beta1_vals,
#             contrast,
#             fb_gain=False,
#             input_gain_beta1=True,
#             input_gain_beta4=False
#         )
        
#         # Save the figure
#         save_path = os.path.join(results_dir, 'Plots', f'pred_perf_input_gain_beta1_contrast_{contrast}.pdf')
#         fig.savefig(save_path, dpi=400, bbox_inches='tight')
#         plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plotting_communication_analysis.py /path/to/results_dir_config_n")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print(f"Processing results from: {results_dir}")
    plot_feedback_gain_results(results_dir)