import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import torch
import copy
from tqdm import tqdm
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Communication import Calculate_Covariance_mean
from Utils.Coherence import create_L_matrix, create_S_matrix
from Utils.matrix_spectrum import matrix_solution

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
        N=36
    )
    
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Set up indices for V1 neurons
    N = params['N']
    y1_idx = np.arange(0,N)
    N1_y = np.arange(N, 2*N)  # index for y1Plus (V1 neurons)
    N2_y = np.arange(3*N, 4*N)  # index for y2Plus (V2 neurons)
    # Communication parameters
    com_params = {
        'bw_y1_y4': False
    }
    
    # Define contrast values to plot
    contrast_vals = config['Communication']['Feedback_gain']['c_vals']
    
    # Get gamma value from config
    gamma_vals = config['Communication']['Feedback_gain']['gamma_vals']
    
    
    # Create figure with subplots
    n_contrasts = len(contrast_vals)
    n_cols = 3
    n_rows = int(np.ceil(n_contrasts / n_cols))
    
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=[1]*n_cols + [0.05])
    
    # Store all covariance matrices and steady states
    all_cov_matrices = []
    all_steady_states = []
    
    # Indices for normalization signals in the state vector
    # State vector order: y1, y1Plus, y4, y4Plus, u1, u1Plus, u4, u4Plus, ...
    # For simulate_firing_rates=True, indices are i*N:(i+1)*N where i is the variable index
    p1Plus_idx = np.arange(9*N, 10*N)  # V1 normalization signal
    p4Plus_idx = np.arange(11*N, 12*N)  # V4 normalization signal
    
    # print("\nCalculating covariance matrices for all contrasts...")
    # for contrast in contrast_vals:
    #     print(f"Processing contrast = {contrast}")
        
    #     # Calculate covariance matrix
    #     Py, ss = Calculate_Covariance_mean(
    #         Ring_Model,
    #         gamma_vals,
    #         contrast,
    #         fb_gain=True,
    #         input_gain_beta1=False,
    #         input_gain_beta4=False,
    #         method='RK45',
    #         com_params=com_params,
    #         delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
    #         noise_potential=config['noise_params']['noise_potential'],
    #         noise_firing_rate=config['noise_params']['noise_firing_rate'],
    #         GR_noise=config['noise_params']['GR_noise'],
    #         t_span=[0, 10]
    #     )
        
    #     # Extract V1-V4 cross-covariance matrix
    #     V1_V2_cov = Py[N1_y][:, N2_y]
    #     all_cov_matrices.append(V1_V2_cov)
        
    #     # Store steady state
    #     all_steady_states.append(ss)
    
    # Convert covariance to correlation matrix
    # Get standard deviations from diagonal of covariance matrix
    # std_devs = np.sqrt(np.diag(V1_cov))
    # # Create outer product of standard deviations
    # outer_std = np.outer(std_devs, std_devs)
    # # Calculate correlation matrix
    # V1_corr = V1_cov / outer_std
    
    # # Ensure the diagonal is exactly 1 (might differ slightly due to numerical precision)
    # np.fill_diagonal(V1_corr, 1.0)
    
    # Plot each covariance matrix
    # for idx, (contrast, V1_V2_cov) in enumerate(zip(contrast_vals, all_cov_matrices)):
    #     row = idx // n_cols
    #     col = idx % n_cols
        
    #     ax = fig.add_subplot(gs[row, col])
        
    #     vmin, vmax = np.percentile(V1_V2_cov, [0, 100]) # exclude extreme values
    #     # Plot covariance matrix
    #     im = ax.imshow(V1_V2_cov, cmap='coolwarm', vmin=vmin, vmax=vmax,
    #                    aspect='auto')
        
    #     # Add title and labels
    #     ax.set_title(f'Contrast = {contrast}', fontsize=12, fontweight='bold')
    #     ax.set_xlabel('V1 Neuron Index', fontsize=10)
    #     ax.set_ylabel('V2 Neuron Index', fontsize=10)
        
    #     # Print statistics for this contrast
    #     triu_indices = np.triu_indices_from(V1_V2_cov, k=1)
    #     print(f"\nContrast = {contrast}:")
    #     print(f"  Mean covariance: {np.mean(V1_V2_cov[triu_indices]):.3e}")
    #     print(f"  Std covariance: {np.std(V1_V2_cov[triu_indices]):.3e}")
    #     print(f"  Range: [{np.min(V1_V2_cov[triu_indices]):.3e}, {np.max(V1_V2_cov[triu_indices]):.3e}]")
    
    # # Add a single colorbar for all subplots
    # cbar_ax = fig.add_subplot(gs[:, -1])
    # cbar = fig.colorbar(im, cax=cbar_ax, format='%.1e')
    # cbar.set_label('Covariance', fontsize=12, fontweight='bold')
    
    # # Add main title
    # fig.suptitle(f'V1 Covariance Matrices Across Contrast Values (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold', y=0.98)
    
    # # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 0.96, 0.96])
    
    # # Save plot in the same directory as config
    output_dir = os.path.dirname(config_file)
    if not output_dir:  # If config_file is just a filename without path
        output_dir = os.getcwd()
    # output_file = os.path.join(output_dir, 'V1_V4_cross_covariance_matrices_contrast.pdf')
    # plt.savefig(output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved cross-covariance plot to: {output_file}")
    # plt.show()
    
    # # ==================== Plot Normalization Signals ====================
    # print("\nPlotting normalization signals (p1Plus and p4Plus)...")
    
    # # Collect steady state values for p1Plus and p4Plus at center neuron
    # p1Plus_vals = []
    # p4Plus_vals = []
    
    # for idx, (contrast, ss) in enumerate(zip(contrast_vals, all_steady_states)):
    #     # Extract normalization signals from steady state (center neuron)
    #     p1Plus_val = ss[p1Plus_idx][N//2]  # V1 normalization signal at center
    #     p4Plus_val = ss[p4Plus_idx][N//2]  # V4 normalization signal at center
        
    #     # Plot the mean p1Plus and p4Plus values
    #     # p1Plus_mean_val = np.mean(ss[p1Plus_idx])
    #     # p4Plus_mean_val = np.mean(ss[p4Plus_idx])
        
    #     p1Plus_vals.append(p1Plus_val)
    #     p4Plus_vals.append(p4Plus_val)
        
    #     # Print statistics
    #     print(f"\nContrast = {contrast}:")
    #     print(f"  V1 (p1Plus) center: {p1Plus_val:.3e}")
    #     print(f"  V4 (p4Plus) center: {p4Plus_val:.3e}")
    
    # # Create a new figure for normalization signals
    # fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # # Plot both normalization signals vs contrast
    # ax.plot(contrast_vals, p1Plus_vals, 'bo-', linewidth=2, markersize=8, 
    #         label='V1 (p1Plus)', alpha=0.8)
    # # ax.plot(contrast_vals, p4Plus_center_vals, 'rs-', linewidth=2, markersize=8, 
    # #         label='V4 (p4Plus)', alpha=0.8)
    
    # # Formatting
    # ax.set_xlabel('Contrast', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Normalization Signal (a)', fontsize=12, fontweight='bold')
    # ax.set_title(f'Normalization Signals vs Contrast (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=11)
    # ax.grid(True, alpha=0.3)
    # ax.set_xscale('log')  # Log scale for contrast since values span orders of magnitude
    # ax.set_yscale('linear')
    # # Adjust layout
    # plt.tight_layout()
    
    # # Save normalization plot
    # norm_output_file = os.path.join(output_dir, 'normalization_signals_vs_contrast.pdf')
    # plt.savefig(norm_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved normalization signals vs contrast plot to: {norm_output_file}")
    # plt.show()
    
    # # ==================== Plot steady state values of y1plus with contrast====================
    # print("\nPlotting steady state values of y1plus with contrast...")
    
    # # Collect steady state values for y1plus at center neuron
    # y1Plus_vals = []
    # y1Plus_std_vals = []
    
    # for idx, (contrast, ss) in enumerate(zip(contrast_vals, all_steady_states)):
    #     # Extract y1plus from steady state (center neuron)
    #     # y1Plus_mean_val = np.mean(ss[N1_y])  # V1 y1Plus at center
    #     y1Plus_val = ss[N1_y][N//2]
    #     # y1Plus_val = ss[y1_idx][N//2]
    #     y1Plus_vals.append(y1Plus_val)
    #     # y1Plus_std_vals.append(np.std(ss[N1_y]))
    #     y1Plus_std_vals.append(np.std(ss[y1_idx]))
    # # Create a new figure for steady state values of y1plus
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(contrast_vals, y1Plus_vals, 'o-', color='blue', linewidth=2, markersize=8, 
    #         label='V1 (y1Plus)', alpha=0.8)
    # # ax.plot(contrast_vals, y1Plus_std_vals, 'o-', color='red', linewidth=2, markersize=8, 
    # #         label='V1 (y1Plus) Std Deviation', alpha=0.8)
    # ax.plot(contrast_vals, np.array(y1Plus_vals)/np.array(p1Plus_vals), 'o-', color='green', linewidth=2, markersize=8, 
    #         label='V1 (y1Plus) / V1 (a1Plus)', alpha=0.8)
    # ax.set_xlabel('Contrast', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Mean and Ratio of Steady State Values of y1Plus and a1Plus', fontsize=12, fontweight='bold')
    # ax.set_title(f'Mean and Ratio of Steady State Values of y1Plus and a1Plus vs Contrast (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=11)
    # ax.grid(True, alpha=0.3)
    # ax.set_xscale('log')  # Log scale for contrast
    # ax.set_yscale('linear')
    
    # # Save steady state values of y1plus plot
    # y1Plus_output_file = os.path.join(output_dir, 'steady_state_values_y1plus_a1plus_contrast.pdf')
    # plt.savefig(y1Plus_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved steady state values of y1plus and a1plus plot to: {y1Plus_output_file}")
    # plt.close()
    # plt.show()
    
    # # ==================== Plot V1 Firing Rates Across All Neurons ====================
    # print("\nPlotting V1 firing rates across all neurons for each contrast...")
    
    # # Create figure with subplots - one subplot per contrast
    # n_contrasts = len(contrast_vals)
    # n_cols = 3
    # n_rows = int(np.ceil(n_contrasts / n_cols))
    
    # fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    # if n_rows == 1 and n_cols == 1:
    #     axes = np.array([axes])
    # elif n_rows == 1 or n_cols == 1:
    #     axes = axes.flatten()
    # else:
    #     axes = axes.flatten()
    
    # # Neuron indices
    # neuron_indices = np.arange(N)
    
    # for idx, (contrast, ss) in enumerate(zip(contrast_vals, all_steady_states)):
    #     # Extract all V1 neuron values
    #     v1_values = ss[N1_y]
        
    #     # Plot firing rates
    #     ax = axes[idx]
    #     ax.plot(neuron_indices, v1_values, 'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.8)
        
    #     # Calculate and show mean
    #     mean_val = np.mean(v1_values)
    #     ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3e}')
        
    #     ax.set_title(f'Contrast = {contrast}', fontsize=12, fontweight='bold')
    #     ax.set_xlabel('Neuron Index', fontsize=10)
    #     ax.set_ylabel('Firing Rate (y1Plus)', fontsize=10)
    #     ax.legend(loc='best', fontsize=8)
    #     ax.grid(True, alpha=0.3)
        
    #     # Print statistics
    #     print(f"\nContrast = {contrast}:")
    #     print(f"  Mean: {mean_val:.3e}")
    #     print(f"  Std: {np.std(v1_values):.3e}")
    #     print(f"  Min: {np.min(v1_values):.3e} (neuron {np.argmin(v1_values)})")
    #     print(f"  Max: {np.max(v1_values):.3e} (neuron {np.argmax(v1_values)})")
    
    # # Hide unused subplots
    # for idx in range(len(contrast_vals), len(axes)):
    #     axes[idx].axis('off')
    
    # # Add main title
    # fig_dist.suptitle(f'V1 Firing Rates Across Neurons for Different Contrasts (γ={gamma_vals[0]})', 
    #                   fontsize=14, fontweight='bold', y=0.98)
    
    # # Adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # # Save distribution plot
    # dist_output_file = os.path.join(output_dir, 'v1_firing_rates_by_neuron_contrast.pdf')
    # plt.savefig(dist_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved V1 firing rates by neuron plot to: {dist_output_file}")
    # plt.close()
    
    # # ==================== Plot All Contrasts on Single Plot ====================
    # print("\nPlotting V1 firing rates for all contrasts on single plot...")
    
    # fig_combined, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # # Define colors for different contrasts
    # colors = plt.cm.viridis(np.linspace(0, 1, len(contrast_vals)))
    
    # for idx, (contrast, ss) in enumerate(zip(contrast_vals, all_steady_states)):
    #     # Extract all V1 neuron values
    #     v1_values = ss[N1_y]
        
    #     # Plot firing rates
    #     ax.plot(neuron_indices, v1_values, 'o-', color=colors[idx], linewidth=2, 
    #             markersize=5, alpha=0.7, label=f'c = {contrast}')
    
    # ax.set_xlabel('Neuron Index', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Firing Rate (y1Plus)', fontsize=12, fontweight='bold')
    # ax.set_title(f'V1 Firing Rates Across Neurons for All Contrasts (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=10, ncol=2)
    # ax.grid(True, alpha=0.3)
    
    # # Adjust layout
    # plt.tight_layout()
    
    # # Save combined plot
    # combined_output_file = os.path.join(output_dir, 'v1_firing_rates_all_contrasts.pdf')
    # plt.savefig(combined_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved combined V1 firing rates plot to: {combined_output_file}")
    # plt.close()
    
   
    
    # # ==================== Plot Eigenvalue Spectra ====================
    # print("\nCalculating eigenvalues of covariance matrices...")
    
    # # Create a new figure for eigenvalue spectra
    # fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # # Define colors for different contrasts (using a colormap)
    # colors = plt.cm.viridis(np.linspace(0, 1, len(contrast_vals)))
    
    # # Store effective dimensions and ranks for later plotting
    # effective_dims_95 = []
    # numerical_ranks = []
    
    # # Calculate and plot eigenvalues for each contrast
    # for idx, (contrast, V1_V2_cov) in enumerate(zip(contrast_vals, all_cov_matrices)):
    #     # Calculate eigenvalues
    #     eigenvalues = np.linalg.eigvals(V1_V2_cov)  # For cross-covariance, use C*C^T
        
    #     # Sort eigenvalues in descending order
    #     eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        
    #     # Calculate cumulative sum (cumulative variance explained)
    #     cumsum_eigenvalues = np.cumsum(eigenvalues_sorted)
    #     total_variance = cumsum_eigenvalues[-1]
    #     cumsum_normalized = cumsum_eigenvalues / total_variance
        
    #     # Calculate effective dimension using threshold approach (similar to Perf_dim_vs_freq)
    #     threshold_95 = 0.95
        
    #     # Find the first dimension where cumulative variance exceeds the threshold
    #     dim_95 = next((dim for dim, var_explained in enumerate(cumsum_normalized, start=1) 
    #                   if var_explained >= threshold_95), len(eigenvalues_sorted))
        
    #     effective_dims_95.append(dim_95)
        
    #     # ========== Compute Rank of Covariance Matrix ==========
    #     # Method 1: Numerical rank using numpy's built-in (uses SVD with default tolerance)
    #     numerical_rank = np.linalg.matrix_rank(V1_V2_cov)
    #     numerical_ranks.append(numerical_rank)
        
    #     # Method 2: Rank based on eigenvalue thresholds
    #     # Count eigenvalues greater than max_eigenvalue * threshold
        
    #     # Method 3: Participation ratio (effective number of dimensions)
    #     # PR = (sum of eigenvalues)^2 / sum of squared eigenvalues
    #     # participation_ratio = (np.sum(eigenvalues_sorted)**2) / np.sum(eigenvalues_sorted**2)
    #     # participation_ratios.append(participation_ratio)
        
    #     # Plot eigenvalues
    #     ax.plot(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, 
    #             'o-', color=colors[idx], linewidth=2, markersize=4,
    #             label=f'c = {contrast}', alpha=0.8)
        
    #     # Print statistics
    #     print(f"\nContrast = {contrast}:")
    #     print(f"  Max eigenvalue: {eigenvalues_sorted[0]:.3e}")
    #     print(f"  Total variance: {total_variance:.3e}")
    #     print(f"  Numerical rank (np.linalg.matrix_rank): {numerical_rank}")
    #     print(f"  Effective dimension (95% variance): {dim_95}")
    #     print(f"  Variance explained by top 5 dims: {cumsum_normalized[4]:.3%}")
    
    # # Formatting
    # ax.set_xlabel('Eigenvalue Index (sorted)', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Eigenvalue Magnitude', fontsize=12, fontweight='bold')
    # ax.set_title(f'Eigenvalue Spectra of Cross-Covariance Matrices (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=10, ncol=2)
    # ax.grid(True, alpha=0.3)
    # ax.set_yscale('log')  # Log scale to see the decay better
    
    # # Adjust layout
    # plt.tight_layout()
    
    # # Save eigenvalue plot
    # eigen_output_file = os.path.join(output_dir, 'eigenvalue_spectra_contrast.pdf')
    # plt.savefig(eigen_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved eigenvalue spectra plot to: {eigen_output_file}")
    # # plt.show()
    
    # # ==================== Plot Rank and Effective Dimensionality vs Contrast ====================
    # print("\nPlotting rank and effective dimensionality vs contrast...")
    
    # # Create a figure with subplots
    # fig4, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # # Plot 1: Different rank measures
    # ax.plot(contrast_vals, numerical_ranks, 'ko-', linewidth=2, markersize=8, 
    #         label='Numerical rank (SVD)', alpha=0.8)
    # ax.plot(contrast_vals, effective_dims_95, 'mo-', linewidth=2, markersize=8, 
    #         label='Effective dim (95% variance)', alpha=0.8)
    
    # ax.set_xlabel('Contrast', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Dimensionality/Rank ', fontsize=12, fontweight='bold')
    # ax.set_title(f'Dimensionality/Rank vs Contrast (γ={gamma_vals[0]})', 
    #              fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=11)
    # ax.grid(True, alpha=0.3)
    # ax.set_xscale('log')  # Log scale for contrast
    
    # # Adjust layout
    # plt.tight_layout()
    
    # # Save dimensionality plot
    # dim_output_file = os.path.join(output_dir, 'rank_dimensionality_contrast.pdf')
    # plt.savefig(dim_output_file, dpi=400, bbox_inches='tight')
    # print(f"\nSaved rank and dimensionality plot to: {dim_output_file}")
    # plt.show()
    
    # ==================== Plot Rank vs Frequency ====================
    print("\nCalculating numerical rank as a function of frequency...")
    
    # Define frequency range
    min_freq = 1.0  # Hz
    max_freq = 80.0  # Hz
    n_freq = 200
    freq_range = torch.linspace(min_freq, max_freq, n_freq)
    
    # Store rank vs frequency for each contrast
    rank_vs_freq_data = {}
    
    # Initial conditions
    initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01
    
    for contrast in tqdm(contrast_vals, desc='Processing contrasts for rank vs frequency'):
        print(f"\nProcessing contrast = {contrast}")
        
        # Update model parameters with current gamma
        updated_params = copy.deepcopy(params)
        updated_params['gamma1'] = gamma_vals[0]
        
        updated_model = copy.deepcopy(Ring_Model)
        updated_model.params = updated_params
        
        # Get Jacobian and steady state
        J, ss = updated_model.get_Jacobian_augmented(
            contrast, 
            initial_conditions, 
            method='RK45', 
            t_span=[0, 10]
        )
        J = torch.tensor(J, dtype=torch.float32)
        
        # Create L and S matrices
        L = create_L_matrix(
            updated_model, 
            ss, 
            config['noise_params']['delta_tau'] * updated_model.params['tau'],
            config['noise_params']['noise_potential'],
            config['noise_params']['noise_firing_rate'],
            config['noise_params']['GR_noise']
        )
        S = create_S_matrix(updated_model)
        
        # Create matrix solution object
        mat_model = matrix_solution(
            J, L, S,
            noise_sigma=config['noise_params']['noise_sigma'],
            noise_tau=config['noise_params']['noise_tau'],
            low_pass_add=False,
            rho=config['noise_params']['rho']
        )
        
        # Calculate spectral matrix for all frequencies
        spectral_mat = mat_model.spectral_matrix(freq=freq_range)
        
        # Calculate rank at each frequency
        ranks_at_freq = []
        
        for f_idx in range(len(freq_range)):
            # Extract cross-spectral matrix between V1 and V4
            # spectral_mat has shape (n_freq, N_total, N_total)
            # We want the cross-spectral matrix between V1 (N1_y) and V4 (N2_y)
            cross_spec_matrix = spectral_mat[f_idx, N1_y, :][:, N2_y]
            
            # Convert to numpy and take real part (or absolute value)
            cross_spec_np = np.abs(cross_spec_matrix.cpu().numpy())
            
            # Calculate numerical rank
            rank = np.linalg.matrix_rank(cross_spec_np,rtol=0.01)
            ranks_at_freq.append(rank)
            
            
        
        rank_vs_freq_data[contrast] = {
            'freq': freq_range.numpy(),
            'rank': np.array(ranks_at_freq)
        }
        
        print(f"  Rank range: [{np.min(ranks_at_freq)}, {np.max(ranks_at_freq)}]")
    
    # Save rank vs frequency data for plotting 
    rank_vs_freq_data_file = os.path.join(output_dir, 'Data', 'rank_vs_freq_data.npy')
    np.save(rank_vs_freq_data_file, rank_vs_freq_data)
    print(f"\nSaved rank vs frequency data to: {rank_vs_freq_data_file}")
    
    # ==================== Plot Rank vs Frequency ====================
    print("\nPlotting rank vs frequency...")
    
    fig5, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for different contrasts
    colors_freq = plt.cm.viridis(np.linspace(0, 1, len(contrast_vals)))
    
    # Plot: Numerical rank vs frequency
    for idx, contrast in enumerate(contrast_vals):
        data = rank_vs_freq_data[contrast]
        ax.plot(data['freq'], data['rank'], '-', color=colors_freq[idx], 
                linewidth=2, label=f'c = {contrast}', alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Numerical Rank', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Matrix Rank vs Frequency (γ={gamma_vals[0]})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    # ax.set_xscale('log')
    
    
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    rank_freq_output_file = os.path.join(output_dir, 'rank_vs_frequency_contrast.pdf')
    plt.savefig(rank_freq_output_file, dpi=400, bbox_inches='tight')
    print(f"\nSaved rank vs frequency plot to: {rank_freq_output_file}")
    # plt.show()
    
    # # ==================== Plot Power Spectra at Different Frequencies and Contrasts ====================
    # print("\nCalculating power spectra at different frequencies and contrasts...")
    
    # # Get output directory from config file path
    # output_dir = os.path.dirname(config_file)
    # if not output_dir:  # If config_file is just a filename without path
    #     output_dir = os.getcwd()
    
    # # Define frequencies to plot
    # freq_vals = [1.0, 10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0]  # Hz
    # freq_range = torch.tensor(freq_vals, dtype=torch.float32)
    
    # # Store spectral matrices for each contrast and frequency
    # all_spectral_data = {}  # {contrast: {freq: matrix}}
    
    # # Initial conditions
    # initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01
    
    # # Calculate spectral matrices for each contrast
    # for contrast in tqdm(contrast_vals, desc='Processing contrasts for power spectra'):
    #     print(f"\nCalculating spectral matrices for contrast = {contrast}...")
        
    #     # Update model parameters with current gamma
    #     updated_params = copy.deepcopy(params)
    #     updated_params['gamma1'] = gamma_vals[0]
        
    #     updated_model = copy.deepcopy(Ring_Model)
    #     updated_model.params = updated_params
        
    #     # Get Jacobian and steady state
    #     J, ss = updated_model.get_Jacobian_augmented(
    #         contrast, 
    #         initial_conditions, 
    #         method='RK45', 
    #         t_span=[0, 10]
    #     )
    #     J = torch.tensor(J, dtype=torch.float32)
        
    #     # Create L and S matrices
    #     L = create_L_matrix(
    #         updated_model, 
    #         ss, 
    #         config['noise_params']['delta_tau'] * updated_model.params['tau'],
    #         config['noise_params']['noise_potential'],
    #         config['noise_params']['noise_firing_rate'],
    #         config['noise_params']['GR_noise']
    #     )
    #     S = create_S_matrix(updated_model)
        
    #     # Create matrix solution object
    #     mat_model = matrix_solution(
    #         J, L, S,
    #         noise_sigma=config['noise_params']['noise_sigma'],
    #         noise_tau=config['noise_params']['noise_tau'],
    #         low_pass_add=False,
    #         rho=config['noise_params']['rho']
    #     )
        
    #     # Calculate spectral matrix for all frequencies
    #     spectral_mat = mat_model.spectral_matrix(freq=freq_range)
        
    #     # Extract cross-spectral matrices between V1 and V2 for each frequency
    #     spectral_matrices_at_freqs = {}
    #     for f_idx, freq in enumerate(freq_vals):
    #         # Extract cross-spectral matrix between V1 (N1_y) and V2 (N2_y)
    #         cross_spec_matrix = spectral_mat[f_idx, N1_y, :][:, N2_y]
            
    #         # Convert to numpy and take absolute value (magnitude of complex spectral matrix)
    #         cross_spec_np = np.abs(cross_spec_matrix.cpu().numpy())
            
    #         spectral_matrices_at_freqs[freq] = cross_spec_np
            
    #         print(f"  Frequency = {freq} Hz: Mean power = {np.mean(cross_spec_np):.3e}, Max power = {np.max(cross_spec_np):.3e}")
        
    #     all_spectral_data[contrast] = spectral_matrices_at_freqs
    
    # # ==================== Plot Spectral Matrices for Each Contrast ====================
    # print("\nPlotting cross-spectral matrices at different frequencies for each contrast...")
    
    # # Plot spectral matrices for each contrast in separate figures
    # for contrast in contrast_vals:
    #     spectral_matrices_at_freqs = all_spectral_data[contrast]
        
    #     # Create figure with subplots - each with its own colorbar
    #     n_freqs = len(freq_vals)
    #     n_cols_freq = 3
    #     n_rows_freq = int(np.ceil(n_freqs / n_cols_freq))
        
    #     fig_spec = plt.figure(figsize=(6*n_cols_freq, 4.5*n_rows_freq))
    #     gs_spec = GridSpec(n_rows_freq, n_cols_freq, figure=fig_spec, 
    #                       hspace=0.4, wspace=0.4)
        
    #     # Calculate global vmin/vmax across all frequencies for this contrast
    #     all_values = []
    #     for freq in freq_vals:
    #         spec_matrix = spectral_matrices_at_freqs[freq]
    #         all_values.extend(spec_matrix[spec_matrix > 0].flatten())
        
    #     all_values = np.array(all_values)
    #     vmin_global = np.percentile(all_values, 0)  # Global min across all frequencies
    #     vmax_global = np.percentile(all_values, 100)  # Global max across all frequencies
        
    #     print(f"\nContrast {contrast}: Global log scale range = [{vmin_global:.3e}, {vmax_global:.3e}]")
        
    #     # Plot each spectral matrix with the same global colorbar scale
    #     for idx, freq in enumerate(freq_vals):
    #         spec_matrix = spectral_matrices_at_freqs[freq]
    #         row = idx // n_cols_freq
    #         col = idx % n_cols_freq
            
    #         ax = fig_spec.add_subplot(gs_spec[row, col])
            
    #         # Replace any zero or negative values with a small positive number for log scale
    #         spec_matrix_safe = np.where(spec_matrix > 0, spec_matrix, vmin_global * 0.1)
            
    #         # Plot spectral matrix with logarithmic color scale (same scale for all)
    #         im_spec = ax.imshow(spec_matrix_safe, cmap='hot', 
    #                            norm=LogNorm(vmin=vmin_global, vmax=vmax_global),
    #                            aspect='auto')
            
    #         # Add individual colorbar for each subplot
    #         cbar = plt.colorbar(im_spec, ax=ax, format='%.1e', fraction=0.046, pad=0.04)
    #         cbar.ax.tick_params(labelsize=8)
            
    #         # Add title and labels
    #         ax.set_title(f'f = {freq} Hz\n(Mean: {np.mean(spec_matrix):.2e}, Max: {np.max(spec_matrix):.2e})', 
    #                     fontsize=11, fontweight='bold')
    #         ax.set_xlabel('V1 Neuron Index', fontsize=10)
    #         ax.set_ylabel('V2 Neuron Index', fontsize=10)
    #         ax.tick_params(labelsize=8)
        
    #     # Add main title
    #     fig_spec.suptitle(f'V1-V2 Cross-Spectral Matrices at Different Frequencies (Log Scale)\n(γ={gamma_vals[0]}, c={contrast})', 
    #                      fontsize=14, fontweight='bold', y=0.995)
        
    #     # Adjust layout
    #     plt.tight_layout()
        
    #     # Save plot
    #     spec_output_file = os.path.join(output_dir, f'cross_spectral_matrices_frequency_c{contrast}.pdf')
    #     plt.savefig(spec_output_file, dpi=400, bbox_inches='tight')
    #     print(f"\nSaved cross-spectral matrices plot (c={contrast}) to: {spec_output_file}")
    #     plt.close()
    
    # # ==================== Plot Eigenvalues of Cross-Spectral Matrices ====================
    # print("\nCalculating eigenvalues of cross-spectral matrices...")
    
    # # Store eigenvalues for all contrasts and frequencies
    # eigenvalues_data = {}  # {contrast: {freq: eigenvalues_sorted}}
    
    # for contrast in contrast_vals:
    #     spectral_matrices_at_freqs = all_spectral_data[contrast]
    #     eigenvalues_data[contrast] = {}
        
    #     for freq in freq_vals:
    #         spec_matrix = spectral_matrices_at_freqs[freq]
            
    #         # Calculate eigenvalues (for cross-spectral matrix, can use spec_matrix @ spec_matrix.T)
    #         eigenvalues = np.linalg.eigvals(spec_matrix)
            
    #         # Sort eigenvalues in descending order
    #         eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
            
    #         eigenvalues_data[contrast][freq] = eigenvalues_sorted
            
    #         # Print statistics
    #         print(f"Contrast {contrast}, Frequency {freq} Hz:")
    #         print(f"  Max eigenvalue: {eigenvalues_sorted[0]:.3e}")
    #         print(f"  Sum of eigenvalues: {np.sum(eigenvalues_sorted):.3e}")
    #         print(f"  Numerical rank: {np.linalg.matrix_rank(spec_matrix)}")
    
    # # ==================== Plot Eigenvalue Spectra for Each Contrast ====================
    # print("\nPlotting eigenvalue spectra for each contrast (top 5 eigenvalues)...")
    
    # # Plot eigenvalues for each contrast (different frequencies in different colors)
    # n_top_eigenvalues = 5
    
    # for contrast in contrast_vals:
    #     fig_eigen, ax = plt.subplots(1, 1, figsize=(10, 6))
        
    #     # Define colors for different frequencies
    #     colors_freq = plt.cm.plasma(np.linspace(0, 1, len(freq_vals)))
        
    #     for idx, freq in enumerate(freq_vals):
    #         eigenvalues_sorted = eigenvalues_data[contrast][freq]
            
    #         # Plot only top 5 eigenvalues
    #         top_eigenvalues = eigenvalues_sorted[:n_top_eigenvalues]
            
    #         ax.plot(range(1, n_top_eigenvalues + 1), top_eigenvalues,
    #                'o-', color=colors_freq[idx], linewidth=2, markersize=8,
    #                label=f'f = {freq} Hz', alpha=0.8)
        
    #     ax.set_xlabel('Eigenvalue Index (sorted)', fontsize=12, fontweight='bold')
    #     ax.set_ylabel('Eigenvalue Magnitude', fontsize=12, fontweight='bold')
    #     ax.set_title(f'Top {n_top_eigenvalues} Eigenvalues of Cross-Spectral Matrices\n(γ={gamma_vals[0]}, c={contrast})', 
    #                  fontsize=14, fontweight='bold')
    #     ax.legend(loc='best', fontsize=10, ncol=2)
    #     ax.grid(True, alpha=0.3)
    #     ax.set_yscale('log')
    #     ax.set_xticks(range(1, n_top_eigenvalues + 1))
        
    #     plt.tight_layout()
        
    #     # Save plot
    #     eigen_output_file = os.path.join(output_dir, f'eigenvalue_spectra_top5_frequency_c{contrast}.pdf')
    #     plt.savefig(eigen_output_file, dpi=400, bbox_inches='tight')
    #     print(f"\nSaved top {n_top_eigenvalues} eigenvalue spectra plot (c={contrast}) to: {eigen_output_file}")
    #     plt.close()
    
    # print(f"\nNumber of V1 neurons: {len(N1_y)}")
    # print(f"Gamma value: {gamma_vals[0]}")
    # print("Processing complete!")
    
   

    
   

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Plot_correlation_matrix.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file = sys.argv[1]
    plot_correlation_matrix(config_file) 