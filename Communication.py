import numpy as np
from control import lyap
from tqdm import tqdm
from Coherence import create_S_matrix, create_L_matrix
import copy
import torch
from matrix_spectrum import matrix_solution

def correlation(J, L, D,bw_y1_y2=False):
    """
    Returns the correlation matrix of the principal neurons in V1 and V2 at
    the specified neuron indices.
    """

    ## Full correlation matrix of the neurons
    # P = solve_continuous_lyapunov(J, -np.dot(L, np.dot(Q, L.T))) # The full correlation matrix
    A = (L @ D @ L.T)
    P = lyap(J, A)
    
    P = P @ np.eye(P.shape[0]) # To make sure that the matrix is symmetric

    ## Correlation matrix for principal neurons of V1 and V2 at selected indices
    # Py = P[np.ix_(idx, idx)]
    if bw_y1_y2:
        P = 2*(P**2)

    return P

# def random_permutation(N1_y_idx, N2_y_idx, n_V1_s, n_V1_t, n_V2_t):
def random_permutation(N1_y_idx: np.ndarray, N2_y_idx: np.ndarray, n_V1_s: int, n_V1_t: int, n_V2_t: int) -> None:

    """
    Returns the indices of the neurons randomly selected for source and target
    (for V1 and V2).
    """

    # Randomly select source V1 neurons
    # a = np.arange(N1_y,2*N1_y)
    V1s_idx = np.random.permutation(N1_y_idx)[:n_V1_s]

    # Randomly select target V1 neurons. Note that these neurons are different from the source neurons.
    # V1t_idx = np.arange(N1_y,2*N1_y)
    # V1t_idx = np.delete(V1t_idx, V1s_idx)
    V1t_idx = np.setdiff1d(N1_y_idx, V1s_idx)
    V1t_idx = np.random.permutation(V1t_idx)[:n_V1_t]

    # Randomly select target V2 neurons.
    # a = np.arange(3*N1_y, 4*N1_y )
    V2t_idx = np.random.permutation(N2_y_idx)[:n_V2_t]

    return V1s_idx, V1t_idx, V2t_idx

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
    predictive dimensions (using reduced-rank-regression) given the required 
    correlation matrices.
    """

    # Step 1: Calculate the optimal weight matrix using OLS
    W_opt = np.linalg.inv(P1) @ P3  # Optimal weight matrix (from OLS)

    # Step 2: Calculate the reference error (e_R) using the trace of P2
    e_R = np.trace(P2)  # Reference error

    # Step 3: Calculate the predicted covariance of the target using W_opt
    predicted_cov = W_opt.T @ P1 @ W_opt

    # Step 4: Perform eigenvalue decomposition on the predicted covariance
    eig_vals, V = np.linalg.eigh(predicted_cov)  # V contains the eigenvectors
    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    V = V[:, idx]

    # Step 5: Initialize variables for reduced-rank regression
    dim = len(eig_vals)
    dims = np.arange(0, dim)
    W = {}  # Dictionary to store all rank-reduced weight matrices
    e = np.zeros(dim)  # Array to store absolute error for each rank

    # Step 6: Reduced-rank-regression by iterating over dimensions
    for i in dims:
        # Select the top i eigenvectors for the reduced-rank projection
        V_i = V[:, :i]
        # Construct the reduced-rank weight matrix
        W[i] = W_opt @ V_i @ V_i.T  # Reduced-rank weight matrix

        # Step 7: Calculate the error for the reduced-rank weight matrix
        e[i] = np.trace(P2 + W[i].T @ P1 @ W[i] - 2 * W[i].T @ P3)

    # Step 8: Calculate prediction performance as 1 - (e / e_R) (MSE/RMSE)
    pred_perf = 1 - (e / e_R)

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


def Calculate_Pred_perf_Dim(model,gamma_vals,contrast,g,fb_gain,input_gain_beta1,input_gain_beta4,method,com_params,delta_tau,noise,t_span=[0,6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    performance_data = {}
    
    for gamma in tqdm(gamma_vals):
        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g
        
        perf_V1 = np.zeros((com_params['num_trials'], com_params['V1_t']))
        perf_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))
        # Get Jacobian
        J, ss = updated_model.get_Jacobian( contrast,initial_conditions, method, t_span)
        
        # Create S and L matrices
        S = create_S_matrix(updated_model)
        D = S**2
        L = create_L_matrix(updated_model, ss, delta_tau, noise)

        # Compute correlation matrices
        Py = correlation(J, L, D, com_params['bw_y1_y4'])


        for kl in range(com_params['num_trials']):
            V1s_idx, V1t_idx, V4t_idx = random_permutation(com_params['N1_y_idx'], com_params['N4_y_idx'], com_params['V1_s'],com_params['V1_t'],com_params['V4_t'])
            dims_V1, dims_V4, perf_V1[kl, :], perf_V4[kl, :] = analysis_ss(Py, V1s_idx, V1t_idx, V4t_idx)
            
        # Store performance data in the dictionary
        performance_data[gamma,contrast] = {
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

    return performance_data

def Calculate_Covariance_mean(model,gamma_vals,contrast,g,fb_gain,input_gain_beta1,input_gain_beta4,method,com_params,delta_tau,noise,t_span=[0,6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    
    for gamma in tqdm(gamma_vals):
        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g
        
        # perf_V1 = np.zeros((com_params['num_trials'], com_params['V1_t']))
        # perf_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))
        # Get Jacobian
        J, ss = updated_model.get_Jacobian( contrast,initial_conditions, method, t_span)
        
        # Create S and L matrices
        S = create_S_matrix(updated_model)
        D = S**2
        L = create_L_matrix(updated_model, ss, delta_tau, noise)

        # Compute correlation matrices
        Py = correlation(J, L, D, com_params['bw_y1_y4'])


    return Py,ss

###################### For frequency Decomposition of CS analysis ######################


def calculate_pred_performance_freq(model, gamma_vals, contrast,g,fb_gain,input_gain_beta1,input_gain_beta4, method,com_params,  delta_tau, noise, freq, bw_y1_y4=False, t_span=[0, 6]):
    N = model.params['N']
    
    initial_conditions = np.ones((model.num_var * N)) * 0.01

    performance_data = {}

    for gamma in tqdm(gamma_vals):
        perf_V1 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency
        perf_V4 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency

        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g

        # Compute the Jacobian
        J, ss = updated_model.get_Jacobian(contrast,initial_conditions, method, t_span)
        J = torch.tensor(J, dtype=torch.float32)

        # Create S and L matrices
        S = create_S_matrix(updated_model)  # Torch tensor
        L = create_L_matrix(updated_model, ss, delta_tau, noise)  # Torch tensor

        # Calculate spectral matrix
        mat_model = matrix_solution(J, L, S)
        S_fij = mat_model.spectral_matrix(freq, J)
        S_fij = 2 * torch.real(S_fij)  # Take the real part of the spectral matrix

        for kl in range(com_params['num_trials']):
            V1s_idx, V1t_idx, V4t_idx = random_permutation(com_params['N1_y_idx'], com_params['N4_y_idx'], com_params['V1_s'], com_params['V1_t'], com_params['V4_t'])
            # Perform analysis for each frequency
            for f_idx, _ in enumerate(freq):
                Py = S_fij[f_idx].cpu().numpy()  # Convert to numpy array
                perf_V1[kl, f_idx], perf_V4[kl, f_idx] = analysis_ss_freq(Py, V1s_idx, V1t_idx, V4t_idx)

        # Store performance data in the dictionary
        performance_data[gamma,contrast] = {
            'V1': {
                'mean': np.mean(perf_V1, axis=0),
                'std': np.std(perf_V1, axis=0),
            },
            'V4': {
                'mean': np.mean(perf_V4, axis=0),
                'std': np.std(perf_V4, axis=0),
            }
        }

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

def calculate_dim_vs_freq(model, gamma_vals, contrast,g,fb_gain,input_gain_beta1,input_gain_beta4, method,com_params,  delta_tau, noise, freq,thresold = 0.95, bw_y1_y4=False, t_span=[0, 6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01

    dimension_data = {}

    for gamma in tqdm(gamma_vals):
        dim_V1 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency
        dim_V4 = np.zeros((com_params['num_trials'], len(freq)))  # each trial, each frequency

        updated_model = copy.deepcopy(model)
        if fb_gain:
            updated_model.params['gamma1'] = gamma
        elif input_gain_beta1:
            updated_model.params['beta1'] = gamma
        elif input_gain_beta4:
            updated_model.params['beta4'] = gamma
        
        updated_model.params['g1'] = g

        # Compute the Jacobian
        J, ss = updated_model.get_Jacobian(contrast,initial_conditions, method, t_span)
        J = torch.tensor(J, dtype=torch.float32)

        # Create S and L matrices
        S = create_S_matrix(updated_model)  # Torch tensor
        L = create_L_matrix(updated_model, ss, delta_tau, noise)  # Torch tensor

        # Calculate spectral matrix
        mat_model = matrix_solution(J, L, S)
        S_fij = mat_model.spectral_matrix(freq, J)
        S_fij = 2 * torch.real(S_fij)  # Take the real part of the spectral matrix

        for kl in range(com_params['num_trials']):
            V1s_idx, V1t_idx, V4t_idx = random_permutation(com_params['N1_y_idx'], com_params['N4_y_idx'], com_params['V1_s'], com_params['V1_t'], com_params['V4_t'])
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
    
    return dim_95 -1

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
        J, ss = updated_model.get_Jacobian(contrast, initial_conditions, method, t_span)
        
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


