import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml
from matplotlib.ticker import NullFormatter

def load_bifurcation_data(data_file_path):
    """Loads bifurcation data from a .npy file."""
    try:
        data = np.load(data_file_path, allow_pickle=True).item()
        # Sort data by contrast values for consistent plotting
        sorted_contrasts = sorted(data.keys())
        valid_contrasts = [c for c in sorted_contrasts if not np.isnan(data[c])]
        valid_values = [data[c] for c in valid_contrasts]
        return valid_contrasts, valid_values
    except FileNotFoundError:
        print(f"Warning: Data file '{data_file_path}' not found. Skipping this curve.")
        return [], []

def main(config_file_path):
    """Loads data and generates separate plots for gamma and beta1 stability."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config from: {config_file_path}")

    # Determine results and data directory from config file path
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.abspath(config_file_path)
    
    results_dir = os.path.dirname(config_file_path)
    if not results_dir: 
        results_dir = os.getcwd()
    data_dir = os.path.join(results_dir, 'Data')

    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")

    # Define data file paths
    gamma_data_file = os.path.join(data_dir, 'contrast_vs_gamma_hopf.npy')
    beta1_data_file = os.path.join(data_dir, 'contrast_vs_beta1_hopf.npy')

    # Load data
    gamma_contrasts, critical_gammas = load_bifurcation_data(gamma_data_file)
    beta1_contrasts, critical_beta1s = load_bifurcation_data(beta1_data_file)

    # Convert contrast data to percentages for plotting
    gamma_contrasts_percent = [c * 100 for c in gamma_contrasts]
    beta1_contrasts_percent = [c * 100 for c in beta1_contrasts]

    # Plot for Gamma1
    if gamma_contrasts_percent:
        fig_gamma, ax_gamma = plt.subplots(figsize=(8, 6))
        ax_gamma.plot(gamma_contrasts_percent, critical_gammas, marker='o', linestyle='-')
        ax_gamma.set_xlabel(r'Contrast (%)')
        ax_gamma.set_ylabel(r'Critical Feedback Gain')
        ax_gamma.set_yscale('log')
        gamma_major_tick_locs = [1,  2,  3,  4,  5]
        gamma_major_tick_labels = [str(loc) for loc in gamma_major_tick_locs]
        ax_gamma.set_yticks(gamma_major_tick_locs)
        ax_gamma.set_yticklabels(gamma_major_tick_labels)
        gamma_minor_tick_locs = [1.5, 2.5, 3.5, 4.5]
        ax_gamma.set_yticks(gamma_minor_tick_locs, minor=True)
        ax_gamma.yaxis.set_minor_formatter(NullFormatter())
        gamma_plot_filename = os.path.join(results_dir, 'contrast_vs_gamma_hopf.pdf')
        fig_gamma.savefig(gamma_plot_filename, dpi=400, bbox_inches='tight')
        print(f"Gamma plot saved to {gamma_plot_filename}")
        plt.show()
    else:
        if not gamma_contrasts:
            print("No data found for Gamma1. Skipping Gamma1 plot.")

    # Plot for Beta1
    if beta1_contrasts_percent:
        fig_beta, ax_beta = plt.subplots(figsize=(8, 6))
        ax_beta.plot(beta1_contrasts_percent, critical_beta1s, marker='s', linestyle='--')
        ax_beta.set_xlabel(r'Contrast (%)')
        ax_beta.set_ylabel(r'Critical Input Gain')
        ax_beta.set_yscale('log')
        beta_major_tick_locs = [1, 2,  3,  4,  5]
        beta_major_tick_labels = [str(loc) for loc in beta_major_tick_locs]
        ax_beta.set_yticks(beta_major_tick_locs)
        ax_beta.set_yticklabels(beta_major_tick_labels)
        beta_minor_tick_locs = [1.5, 2.5, 3.5, 4.5]
        ax_beta.set_yticks(beta_minor_tick_locs, minor=True)
        ax_beta.yaxis.set_minor_formatter(NullFormatter())
        beta1_plot_filename = os.path.join(results_dir, 'contrast_vs_beta1_hopf.pdf')
        fig_beta.savefig(beta1_plot_filename, dpi=400, bbox_inches='tight')
        print(f"Beta1 plot saved to {beta1_plot_filename}")
        plt.show()
    else:
        print("No data found for Beta1. Skipping Beta1 plot.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_stability_curves.py path/to/your_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path) 