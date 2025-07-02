import numpy as np
from control import lyap
from tqdm import tqdm
from Utils.Coherence import create_S_matrix, create_L_matrix
import copy
import torch
from Utils.matrix_spectrum import matrix_solution

def correlation(J, L, D,bw_y1_y2=False):
    """
    Returns the correlation matrix of the principal neurons in V1 and V2 at
    the specified neuron indices.
    """

    ## Full correlation matrix of the neurons
    # P = solve_continuous_lyapunov(J, -np.dot(L, np.dot(Q, L.T))) # The full correlation matrix
    A = (L @ D @ L.T) # Q
    P = lyap(J, A)
    
    P = P @ np.eye(P.shape[0]) # To make sure that the matrix is symmetric

    ## Correlation matrix for principal neurons of V1 and V2 at selected indices
    # Py = P[np.ix_(idx, idx)]
    if bw_y1_y2:
        P = 2*(P**2)

    return P

# def random_permutation(N1_y_idx, N2_y_idx, n_V1_s, n_V1_t, n_V2_t):
def random_permutation(N1_y_idx, N4_y_idx, V1_s, V1_t, V4_t, ss):
    """
    Randomly select neurons for analysis, considering only those with significant activity.
    
    Args:
        N1_y_idx: Indices for V1 neurons
        N4_y_idx: Indices for V4 neurons
        V1_s: Number of source neurons to select from V1
        V1_t: Number of target neurons to select from V1
        V4_t: Number of target neurons to select from V4
        ss: Steady state values containing firing rates
    """
    # Get firing rates for V1 and V4
    V1_rates = ss[N1_y_idx]
    V4_rates = ss[N4_y_idx]
    
    # Calculate thresholds (5% of maximum)
    # V1_threshold = 0.00005 * np.max(V1_rates)
    # V4_threshold = 0.00005 * np.max(V4_rates)
    V1_threshold = 0.0000 
    V4_threshold = 0.0000
    # V1_threshold = np.percentile(V1_rates, 70)
    # V4_threshold = np.percentile(V4_rates, 70)
    # Get indices of neurons above threshold
    V1_active = N1_y_idx[V1_rates >= V1_threshold]
    V4_active = N4_y_idx[V4_rates >= V4_threshold]
    # print(len(V1_active), len(V4_active),V1_s,V1_t,V4_t)
    # Ensure we have enough active neurons
    if len(V1_active) < (V1_s + V1_t) or len(V4_active) < V4_t:
        raise ValueError("Not enough active neurons above threshold")
    
    # Random selection from active neurons
    V1_perm = np.random.permutation(V1_active)
    V4_perm = np.random.permutation(V4_active)
    
    # Select required numbers of neurons
    V1s_idx = V1_perm[:V1_s]
    V1t_idx = V1_perm[V1_s:V1_s+V1_t]
    V4t_idx = V4_perm[:V4_t]
    
    return V1s_idx, V1t_idx, V4t_idx

def selection_mat(Py, V1s_idx, V1t_idx, V2t_idx): 
    """
    Returns the submatrices required for the analysis. See main text for the
    definition of Pi. 
    """
    
    # Extracting the different blocks

    # For V1
    P1 = Py[np.ix_(V1s_idx, V1s_idx)]
    P2 = Py[np.ix_(V1t_idx, V1t_idx)]
    P3 = Py[np.ix_(V1s_idx, V1t_idx)]

    # For V2
    P4 = Py[np.ix_(V2t_idx, V2t_idx)]
    P5 = Py[np.ix_(V1s_idx, V2t_idx)]

    return P1, P2, P3, P4, P5

def analysis_ss(mat, V1s_idx, V1t_idx, V2t_idx):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    
    # Select the values at randomized row and columns
    P1, P2, P3, P4, P5 = selection_mat(mat, V1s_idx, V1t_idx, V2t_idx)
    
    # For V1 target
    perf_V1, dims_V1 = performance(P1, P2, P3)
    
    # For V2 target
    perf_V2, dims_V2 = performance(P1, P4, P5)
    
    return dims_V1, dims_V2, perf_V1, perf_V2

# import numpy as np
# import numpy as np

# def performance(P1, P2, P3):
#     """
#     This function returns the prediction performance as a function of
#     predictive dimensions (using reduced-rank-regression) given the required 
#     correlation matrices.
#     """

#     # Step 1: Calculate the optimal weight matrix using OLS
#     W_opt = np.linalg.inv(P1) @ P3  # Optimal weight matrix (from OLS)

#     # Step 2: Calculate the reference trace of P2 for normalization
#     trace_P2 = np.trace(P2)

#     # Step 3: Calculate the predicted covariance of the target using W_opt
#     predicted_cov = W_opt.T @ P1 @ W_opt

#     # Step 4: Perform eigenvalue decomposition on the predicted covariance
#     eig_vals, V = np.linalg.eigh(predicted_cov)  # V contains the eigenvectors
#     # Sort eigenvalues and corresponding eigenvectors in descending order
#     idx = np.argsort(eig_vals)[::-1]
#     eig_vals = eig_vals[idx]
#     V = V[:, idx]

#     # Step 5: Initialize variables for reduced-rank regression
#     dim = len(eig_vals)
#     dims = np.arange(0, dim)
#     W = {}  # Dictionary to store all rank-reduced weight matrices
#     e = np.zeros(dim)  # Array to store error for each rank

#     # Step 6: Reduced-rank-regression by iterating over dimensions
#     for i in dims:
#         # Select the top i eigenvectors for the reduced-rank projection
#         V_i = V[:, :i]
#         # Construct the reduced-rank weight matrix
#         W[i] = W_opt @ V_i @ V_i.T  # Reduced-rank weight matrix

#         # Step 7: Calculate the error for the reduced-rank weight matrix
#         predicted_cov_reduced = W[i].T @ P1 @ W[i]
#         error = np.linalg.norm(P2 - predicted_cov_reduced, 'fro')**2 / trace_P2
#         e[i] = error

#     # Step 8: Calculate performance as 1 - Error
#     pred_perf = 1 - e

#     return pred_perf, dims


def performance(P1, P2, P3):
    """
    This function returns the prediction performance as a function of
    predictive dimensions (using reduced-rank-regression).

    The performance for a rank-i approximation is calculated based on the
    simplified formula:
    
    Performance(i) = sum_{j=1 to i} lambda_j(C4) / Tr(C2)
    
    where:
    - C1, C2, C3 are the input covariance matrices corresponding to P1, P2, P3.
    - C4 = C3^T * C1^-1 * C3 is the covariance of the predicted responses.
    - lambda_j(C4) are the eigenvalues of C4.
    - Tr(C2) is the total variance of the target population.

    Args:
        P1 (np.ndarray): Covariance matrix of the source population, C1 = E[ss^T].
        P2 (np.ndarray): Covariance matrix of the target population, C2 = E[tt^T].
        P3 (np.ndarray): Cross-covariance matrix between source and target, C3 = E[st^T].

    Returns:
        tuple: A tuple containing:
            - pred_perf (np.ndarray): The prediction performance for each rank from 0 to dim-1.
            - dims (np.ndarray): The corresponding dimensions (ranks).
    """
    # For clarity, map the input variables to the notation in the derivation
    C1, C2, C3 = P1, P2, P3

    # --- Step 1: Calculate the predicted covariance matrix, C4 ---
    # C4 = C3^T * C1^-1 * C3
    # Using np.linalg.solve(C1, C3) to compute (C1^-1 * C3) is more
    # numerically stable than computing the inverse directly.
    try:
        # Solves for X in C1 @ X = C3, where X = C1^-1 * C3
        inv_C1_C3 = np.linalg.solve(C1, C3)
    except np.linalg.LinAlgError:
        # If C1 is singular or ill-conditioned, use the pseudo-inverse
        inv_C1_C3 = np.linalg.pinv(C1) @ C3
    # inv_C1_C3 = np.linalg.inv(C1) @ C3
    C4 = C3.T @ inv_C1_C3

    # --- Step 2: Calculate the total variance of the target population ---
    # This is the reference error, epsilon_0 = Tr(C2)
    total_target_variance = np.trace(C2)

    # The number of dimensions is the size of the target population
    dim = C2.shape[0]
    dims = np.arange(0, dim)

    # --- Step 3: Find the eigenvalues of C4 ---
    # C4 is a covariance matrix, hence it is symmetric. We can use the
    # more efficient `eigvalsh` for Hermitian (or symmetric) matrices.
    eig_vals,_ = np.linalg.eigh(C4)
    # Sort eigenvalues in descending order to find the principal dimensions
    eig_vals = np.sort(eig_vals)[::-1]
    
    # Due to numerical precision, some eigenvalues might be slightly negative.
    # Since they represent variance, they should be clipped to zero.
    # eig_vals[eig_vals < 0] = 0

    # --- Step 4: Calculate the cumulative explained variance ---
    # The numerator, sum_{j=1 to i} lambda_j, is a cumulative sum.
    cumulative_explained_variance = np.cumsum(eig_vals)

    # --- Step 5: Calculate the final prediction performance ---
    # Performance for rank i = (cumulative variance up to i) / (total variance)
    # This array corresponds to ranks 1, 2, ..., dim.
    pred_perf = np.zeros(dim)
    # for i in dims:
    #     eig_vals_i = eig_vals[:i]
    #     pred_perf[i] = np.sum(eig_vals_i) / total_target_variance
    
    perf_for_ranks_1_to_dim = cumulative_explained_variance / total_target_variance
    
    # --- Step 6: Format the output to match the original function's structure ---
    # The output should be for ranks 0, 1, ..., dim-1.
    
    
    # Performance for rank 0 is 0.
    # For ranks 1 and higher (up to dim-1), populate the array.
    if dim > 1:
    #     # The performance for ranks 1..dim-1 are the first dim-1 elements
        # of the array calculated in the previous step.
        pred_perf[1:] = perf_for_ranks_1_to_dim[:-1]

    return pred_perf, dims


# def performance(P1, P2, P3):
#     """
#     This function returns the prediction performance as a function of
#     predictive dimensions (using reduced-rank-regression) given the required 
#     correlation matrices.
#     """

#     # Step 1: Calculate the optimal weight matrix using OLS
#     W_opt = np.linalg.inv(P1) @ P3  # Optimal weight matrix (from OLS)

#     # Step 2: Calculate the reference error (e_R) using the trace of P2
#     # e_R =  np.trace(P2)  # Reference error
#     # e_R = np.abs(e_R)
#     e_R = np.linalg.norm(P2,'fro')
#     # e_R = np.trace(P2+P1)

#     # Step 3: Calculate the predicted covariance of the target using W_opt
#     predicted_cov = W_opt.T @ P1 @ W_opt

#     # Step 4: Perform eigenvalue decomposition on the predicted covariance
#     eig_vals, V = np.linalg.eigh(predicted_cov)  # V contains the eigenvectors
#     # Sort eigenvalues and corresponding eigenvectors in descending order
#     idx = np.argsort(eig_vals)[::-1]
#     eig_vals = eig_vals[idx]
#     V = V[:, idx]

#     # Step 5: Initialize variables for reduced-rank regression
#     dim = len(eig_vals)
#     dims = np.arange(0, dim)
#     W = {}  # Dictionary to store all rank-reduced weight matrices
#     e = np.zeros(dim)  # Array to store absolute error for each rank

#     # Step 6: Reduced-rank-regression by iterating over dimensions
#     for i in dims:
#         # Select the top i eigenvectors for the reduced-rank projection
#         V_i = V[:, :i]
#         # Construct the reduced-rank weight matrix
#         W[i] = W_opt @ V_i @ V_i.T  # Reduced-rank weight matrix

#         # Step 7: Calculate the error for the reduced-rank weight matrix
#         # e[i] = np.trace(P2 + W[i].T @ P1 @ W[i] - 2 * W[i].T @ P3)
#         predicted_cov_reduced = W[i].T @ P1 @ W[i]
#         error_matrix = P2 - predicted_cov_reduced
#         e[i] = np.linalg.norm(error_matrix,'fro')

#     # Step 8: Calculate prediction performance as 1 - (e / e_R) (MSE/RMSE)
#     pred_perf = 1 - (e / e_R)
    

    return pred_perf, dims


# def performance(P1, P2, P3):
#     """
#     This function returns the prediction performance as a function of
#     predictive dimensions (using reduced-rank-regression) given the required 
#     correlation matrices.
#     """

#     # We first calculate the mean-squared error and the optimal weight matrix
#     W_opt = np.linalg.inv(P1) @ P3  # Optimal weight matrix (from OLS)

#     e_R = np.trace(P2)  # Stefano used this as the reference error for the B_0 matrix

#     # SVD of optimal 
#     # print('W_opt', W_opt.shape)
#     U, S, V = np.linalg.svd(W_opt)
#     # print('U', U.shape, 'S', S.shape, 'V', V.shape)
#     sing_vals = S
#     # print('S shpae: ', S.shape)
#     dim = min(S.shape)
#     dims = np.arange(0, dim )

#     W = {} # This dictionary will store all the rank-reduced weight matrices
#     e = np.zeros(dim) # Absolute error

#     # Reduced-rank-regression
#     for i in dims:
#         vec = np.zeros(S.shape) # Ensure vec has the right length
#         vec[:i] = sing_vals[:i]
        
#         S_new = np.zeros((U.shape[0],V.shape[0])) # Change this line, create S_new of shape (20, 15)
#         np.fill_diagonal(S_new[:i,:i], vec) # Fill the diagonal with the singular values
#         # print('S_new shape: ', S_new.shape, 'U shape: ', U.shape, 'V shape: ', V.shape)
#         # W[i] = U @ S_new @ V.T
#         W[i] = np.dot(U, np.dot(S_new, V))
#         # Absolute performance
#         e[i] = np.trace(P2 + W[i].T @ P1 @ W[i] - 2 * W[i].T @ P3)

#     # We calculate the MSE/RMSE
#     pred_perf = 1 - (e / e_R)

#     return pred_perf, dims


def Calculate_Pred_perf_Dim(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params, t_span=[0,6]):
    N = model.params['N']
    params = model.params
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    performance_data = {}
    covariance_data = {}
    
    for contrast in tqdm(contrast_vals):
        for gamma in tqdm(gamma_vals):
            updated_params = copy.deepcopy(params)
            if fb_gain:
                updated_params['gamma1'] = gamma
            elif input_gain_beta1:
                updated_params['beta1'] = gamma
            elif input_gain_beta4:
                updated_params['beta4'] = gamma
            
            updated_model = copy.deepcopy(model)
            updated_model.params = updated_params
            
            # Get Jacobian and steady state
            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)
            
            # Create S and L matrices
            S = create_S_matrix(updated_model)
            D = S**2
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)
            
            # Compute correlation matrices
            Py = correlation(J, L, D, com_params['bw_y1_y4'])
            # # Upscale the correlation matrix
            # Py = Py * 0.10
            perf_V1 = np.zeros((com_params['num_trials'], com_params['V1_t']))
            perf_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))
            
            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4t_idx = random_permutation(
                    com_params['N1_y_idx'], 
                    com_params['N4_y_idx'], 
                    com_params['V1_s'],
                    com_params['V1_t'],
                    com_params['V4_t'],
                    ss  # Pass steady state to random_permutation
                )
                dims_V1, dims_V4, perf_V1[kl, :], perf_V4[kl, :] = analysis_ss(Py, V1s_idx, V1t_idx, V4t_idx)
            
            # Store performance data
            performance_data[gamma, contrast] = {
                'V1': {
                    'mean': np.mean(perf_V1, axis=0),
                    'std': np.std(perf_V1, axis=0),
                    'dims': dims_V1
                },
                'V4': {
                    'mean': np.mean(perf_V4, axis=0),
                    'std': np.std(perf_V4, axis=0),
                    'dims': dims_V4
                }
            }
            covariance_data[gamma, contrast] = Py
    
    return performance_data, covariance_data

def Calculate_Covariance_mean(model,gamma_vals,contrast,g,fb_gain,input_gain_beta1,input_gain_beta4,method,com_params,delta_tau,noise_potential,noise_firing_rate,GR_noise,t_span=[0,6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    S = create_S_matrix(updated_model)
    D = S**2
    for gamma in tqdm(gamma_vals):
        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g
        
        # Get Jacobian
        J, ss = updated_model.get_Jacobian_augmented(contrast,initial_conditions, method, t_span)
        
        # Create S and L matrices
        
        L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

        # Compute correlation matrices
        Py = correlation(J, L, D, com_params['bw_y1_y4'])

    return Py,ss

###################### For frequency Decomposition of CS analysis ######################


def calculate_pred_performance_freq(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4, 
                                     delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params,
                                    freq,  t_span=None):
    N = model.params['N']
    
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    performance_data = {}

    for gamma in tqdm(gamma_vals):
        for contrast in tqdm(contrast_vals, leave=False):  # nested progress bar
            perf_V1 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency
            perf_V4 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency

            updated_model = copy.deepcopy(model)
            if fb_gain:
                updated_model.params['gamma1'] = gamma
            elif input_gain_beta1:
                updated_model.params['beta1'] = gamma
            elif input_gain_beta4:
                updated_model.params['beta4'] = gamma

            # Compute the Jacobian and steady state using the updated method
            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            # Create S and L matrices using the updated functions
            S = create_S_matrix(updated_model)  # Should return a Torch tensor for S
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)  # Updated call with new noise parameters

            # Calculate spectral matrix using the new Jacobian, L and S
            mat_model = matrix_solution(J, L, S)
            S_fij = mat_model.spectral_matrix(freq, J)
            S_fij = 2 * torch.real(S_fij)  # Take the real part of the spectral matrix

            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4t_idx = random_permutation(
                    com_params['N1_y_idx'], com_params['N4_y_idx'], 
                    com_params['V1_s'], com_params['V1_t'], com_params['V4_t'], ss
                )
                # Perform analysis for each frequency
                for f_idx, _ in enumerate(freq):
                    Py = S_fij[f_idx].cpu().numpy()  # Convert to numpy array
                    perf_V1[kl, f_idx], perf_V4[kl, f_idx] = analysis_ss_freq(Py, V1s_idx, V1t_idx, V4t_idx)

            # Store performance data in the dictionary, indexed by (gamma, contrast)
            performance_data[gamma, contrast] = {
                'V1': {
                    'mean': np.mean(perf_V1, axis=0),
                    'std': np.std(perf_V1, axis=0),
                },
                'V4': {
                    'mean': np.mean(perf_V4, axis=0),
                    'std': np.std(perf_V4, axis=0),
                }
            }
            
            # Print progress
            print(f"Completed gamma={gamma:.3f}, contrast={contrast:.3f}")

    return performance_data

def performance_all_dimensions(P1, P2, P3):
    """
    This function returns the prediction performance for the highest dimension
    given the required correlation matrices.
    """
    # We first calculate the mean-squared error and the optimal weight matrix
    W_opt = np.linalg.inv(P1) @ P3  # Optimal weight matrix (from OLS)
    
    # SVD of optimal 
    U, S, V = np.linalg.svd(W_opt)
    # print('U', U.shape, 'S', S.shape, 'V', V.shape)
    sing_vals = S
    # print('S shpae: ', S.shape)
    dim = min(S.shape)
    dims = np.arange(0, dim )
    vec = np.zeros(S.shape) # Ensure vec has the right length
    vec[:dim] = sing_vals[:dim]
    
    S_new = np.zeros((U.shape[0],V.shape[0])) # Change this line, create S_new of shape (20, 15)
    np.fill_diagonal(S_new[:dim,:dim], vec) # Fill the diagonal with the singular values
    # print('S_new shape: ', S_new.shape, 'U shape: ', U.shape, 'V shape: ', V.shape)
    # W[i] = U @ S_new @ V.T
    W_re = np.dot(U, np.dot(S_new, V))
    # Absolute performance
    e = np.trace(P2 + W_re.T @ P1 @ W_re - 2 * W_re.T @ P3)

    e_R = np.trace(P2)  # Reference error for the B_0 matrix

    # Calculate the error for the highest dimension (full rank)
    # e = np.trace(P2 + W_opt.T @ P1 @ W_opt - 2 * W_opt.T @ P3) #e[i] = np.trace(P2 + W[i].T @ P1 @ W[i] - 2 * W[i].T @ P3)

    # Calculate the prediction performance
    pred_perf = 1 - (e / e_R)

    return pred_perf

def analysis_ss_freq(mat, V1s_idx, V1t_idx, V4t_idx):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    
    # Select the values at randomized row and columns
    P1, P2, P3, P4, P5 = selection_mat(mat, V1s_idx, V1t_idx, V4t_idx)
    
    # For V1 target
    perf_V1 = performance_all_dimensions(P1, P2, P3)
    
    # For V2 target
    perf_V4 = performance_all_dimensions(P1, P4, P5)
    
    return perf_V1, perf_V4


#################### For Dimensionality vs Frequency analysis ######################

def calculate_dim_vs_freq(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params, freq, thresold, t_span=[0, 6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01

    dimension_data = {}

    for gamma in tqdm(gamma_vals):
        for contrast in tqdm(contrast_vals, leave=False): # nested progress bar
            dim_V1 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency
            dim_V4 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency

            updated_model = copy.deepcopy(model)
            if fb_gain:
                updated_model.params['gamma1'] = gamma
            elif input_gain_beta1:
                updated_model.params['beta1'] = gamma
            elif input_gain_beta4:
                updated_model.params['beta4'] = gamma
            
            # Compute the Jacobian
            J, ss = updated_model.get_Jacobian_augmented(contrast,initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            # Create S and L matrices
            S = create_S_matrix(updated_model)  # Torch tensor
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)  # Torch tensor

            # Calculate spectral matrix
            mat_model = matrix_solution(J, L, S)
            S_fij = mat_model.spectral_matrix(freq, J)
            S_fij = 2 * torch.real(S_fij)  # Take the real part of the spectral matrix

            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4t_idx = random_permutation(com_params['N1_y_idx'], com_params['N4_y_idx'], com_params['V1_s'], com_params['V1_t'], com_params['V4_t'], ss)
                # Perform analysis for each frequency
                for f_idx, _ in enumerate(freq):
                    Py = S_fij[f_idx].cpu().numpy()  # Convert to numpy array
                    dim_V1[kl, f_idx], dim_V4[kl, f_idx] = analysis_dim_vs_freq(Py, V1s_idx, V1t_idx, V4t_idx,thresold)

            # Store performance data in the dictionary
            dimension_data[gamma,contrast] = {
                'V1': {
                    'mean': np.mean(dim_V1, axis=0),
                    'std': np.std(dim_V1, axis=0),
                },
                'V4': {
                    'mean': np.mean(dim_V4, axis=0),
                    'std': np.std(dim_V4, axis=0),
                }
            }
            # Print progress
            print(f"Completed gamma={gamma:.3f}, contrast={contrast:.3f}")

    return dimension_data

def analysis_dim_vs_freq(mat, V1s_idx, V1t_idx, V4t_idx,threshold):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    
    # Select the values at randomized row and columns
    P1, P2, P3, P4, P5 = selection_mat(mat, V1s_idx, V1t_idx, V4t_idx)
    
    # For V1 target
    dim_V1 = Perf_dim_vs_freq(P1, P2, P3,threshold)
    
    # For V2 target
    dim_V4 = Perf_dim_vs_freq(P1, P4, P5,threshold)
    
    return dim_V1, dim_V4
    
def Perf_dim_vs_freq(P1, P2, P3,threshold):
    """
    Wrapper function that returns the dimension when the prediction performance
    reaches 95% of its highest value.
    """
    pred_perf, dims = performance(P1, P2, P3)
    
    max_perf = np.max(pred_perf)
    threshold_perf = threshold * max_perf
    
    # Find the first dimension where performance exceeds the threshold
    dim_95 = next((dim for dim, perf in zip(dims, pred_perf) if perf >= threshold_perf), None)
    
    return dim_95

def Calculate_Fano_Factor(model, gamma_vals, contrast, g, fb_gain, input_gain_beta1, input_gain_beta4, method, com_params, delta_tau, noise,baseline,poisson, t_span=[0,6]):
    """
    Calculate Fano Factor (variance/mean) for a specific neuron across different gamma values and contrasts.
    
    Args:
        ... (same as Calculate_Covariance_mean)
        neuron_idx (int): Index of the neuron to calculate Fano factor for
    
    Returns:
        dict: Dictionary containing Fano factors and mean firing rates for each gamma,contrast pair
    """
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    fano_data = {}
    neuron_idx = com_params['N_idx']
    for gamma in tqdm(gamma_vals):
        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g
        
        # Get Jacobian and steady state
        J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
        
        # Create S and L matrices
        S = create_S_matrix(updated_model)
        D = S**2
        L = create_L_matrix(updated_model, ss, delta_tau, noise,baseline=baseline,poisson=poisson)

        # Compute correlation matrices
        Py = correlation(J, L, D, com_params['bw_y1_y4'])
        
        # Calculate variance (from diagonal of covariance matrix)
        variance = Py[neuron_idx, neuron_idx]
        
        # Get mean firing rate from steady state
        mean_rate = ss[neuron_idx]
        
        # Calculate Fano factor
        fano_factor = variance / mean_rate if mean_rate > 0 else 0
        
        fano_data[gamma, contrast] = {
            'fano_factor': fano_factor,
            'mean_rate': mean_rate,
            'variance': variance
        }
    
    return fano_data


