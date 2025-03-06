import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Communication import Calculate_Covariance_mean

def plot_correlation_matrix(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config from: {config_file}")
    
    # Load config with robust error handling
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
        print(f"Absolute path: {os.path.abspath(config_file)}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    
    # Get results directory from config file path
    results_dir = os.path.dirname(config_file)
    
    # Initialize model parameters
    params = setup_parameters(
        config=config,
        tau=1e-3,
        tauPlus=1e-3,
        N=36*3
    )
    
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Set up indices for V1 neurons
    N = params['N']
    N1_y = np.arange(N, 2*N)  # index for y1Plus (V1 neurons)
    
    # Communication parameters
    com_params = {
        'N1_y_idx': N1_y,
        'N4_y_idx': np.arange(3*N, 4*N),
        'V1_s': 15,
        'V1_t': 15,
        'V4_t': 15,
        'bw_y1_y4': False
    }
    
    # Get correlation matrix for a specific gamma and contrast
    gamma_vals = [config['Communication']['Feedback_gain']['gamma_vals'][0]]  # Use first gamma value
    contrast = config['Communication']['Feedback_gain']['c_vals'][0]  # Use first contrast value
    g = 0.5  # Default value for g
    
    # Calculate covariance matrix
    Py, ss = Calculate_Covariance_mean(
        Ring_Model,
        gamma_vals,
        contrast,
        g,
        fb_gain=True,
        input_gain_beta1=False,
        input_gain_beta4=False,
        method='RK45',
        com_params=com_params,
        delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
        noise_potential=config['noise_params']['noise_potential'],
        noise_firing_rate=config['noise_params']['noise_firing_rate'],
        GR_noise=config['noise_params']['GR_noise'],
        t_span=[0, 6]
    )
    
    # Extract V1 covariance matrix
    V1_cov = Py[N1_y][:, N1_y]
    
    # Convert covariance to correlation matrix
    # Get standard deviations from diagonal of covariance matrix
    # std_devs = np.sqrt(np.diag(V1_cov))
    # # Create outer product of standard deviations
    # outer_std = np.outer(std_devs, std_devs)
    # # Calculate correlation matrix
    # V1_corr = V1_cov / outer_std
    
    # # Ensure the diagonal is exactly 1 (might differ slightly due to numerical precision)
    # np.fill_diagonal(V1_corr, 1.0)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get the maximum absolute value for symmetric color scaling
    max_abs_val = np.max(np.abs(V1_cov))
    
    # Plot correlation matrix with dynamic range
    im = plt.imshow(V1_cov, cmap='RdBu_r', 
                    vmin=-max_abs_val, vmax=max_abs_val)
    
    # Add colorbar with scientific notation if values are very small
    plt.colorbar(im, label='Covariance', format='%.1e')
    
    # Add labels and title
    plt.title(f'V1 Neurons Covariance Matrix\n(γ={gamma_vals[0]}, contrast={contrast})')
    plt.xlabel('V1 Neuron Index')
    plt.ylabel('V1 Neuron Index')
    
    # Print the range of values before plotting
    print(f"Covariance matrix value range: [{np.min(V1_cov):.2e}, {np.max(V1_cov):.2e}]")
    print(f"Symmetric colorbar range: [-{max_abs_val:.2e}, {max_abs_val:.2e}]")
    
    # Create output directory if it doesn't exist
    os.makedirs('Tests', exist_ok=True)
    
    # Save plot in the same directory as config
    output_dir = os.path.dirname(config_file)
    output_file = os.path.join(output_dir, 'V1_covariance_matrix.pdf')
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.show()
    
    # Print statistics
    triu_indices = np.triu_indices_from(V1_cov, k=1)
    print(f"Number of V1 neurons: {len(N1_y)}")
    print(f"Mean covariance: {np.mean(V1_cov[triu_indices]):.3f}")
    print(f"Std covariance: {np.std(V1_cov[triu_indices]):.3f}")
    # Also print the range of correlations
    print(f"Covariance range: [{np.min(V1_cov[triu_indices]):.3f}, {np.max(V1_cov[triu_indices]):.3f}]")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_correlation_matrix.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    plot_correlation_matrix(config_file) 