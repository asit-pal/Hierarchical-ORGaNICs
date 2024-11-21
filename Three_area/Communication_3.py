# Standard library imports
import copy
import numpy as np

# Third-party library imports
from control import lyap
from tqdm.notebook import tqdm
import torch

# Utils imports
from Four_neuron_model.Communication import performance
from Four_neuron_model.Coherence import (
    calculate_noise_variance,
    calculate_effective_tau_yPlus,
    calculate_effective_tau_uPlus,
    calculate_effective_tau_pPlus
)
from .Three_area_model import RingModel

def create_S_matrix(model):
    N = model.params['N']
    
    if model.simulate_firing_rates:
        tau_list = [
            'tauY1', 'tauYPlus1', 'tauY4', 'tauYPlus4', 'tauY5', 'tauYPlus5',
            'tauU1', 'tauUPlus1', 'tauU4', 'tauUPlus4', 'tauU5', 'tauUPlus5',
            'tauP1', 'tauPPlus1', 'tauP4', 'tauPPlus4', 'tauP5', 'tauPPlus5',
            'tauS1', 'tauSPlus1', 'tauS4', 'tauSPlus4', 'tauS5', 'tauSPlus5',
        ]
    else:
        tau_list = [
            'tauY1', 'tauY4', 'tauY5', 
            'tauU1', 'tauU4', 'tauU5',
            'tauP1', 'tauP4', 'tauP5', 
            'tauS1', 'tauS4', 'tauS5',
        ]
    
    S_diag = []
    for tau in tau_list:
        S_diag.extend([np.sqrt(1 / model.params[tau])] * N)
    
    S = np.diag(S_diag)
    return torch.tensor(S, dtype=torch.float32)

def create_L_matrix(model, ss, delta_tau, noise):
    N = model.params['N']
    params = model.params
    if model.simulate_firing_rates:
        # Split steady state into chunks of size N
        ss_chunks = [ss[i*N:(i+1)*N] for i in range(model.jacobian_dimension)]
        
        # Unpack steady states for all areas
        _, y1Plus, _, y4Plus, _, y5Plus, \
        u1, u1Plus, u4, u4Plus, u5, u5Plus, \
        p1, p1Plus, p4, p4Plus, p5, p5Plus, \
        s1, s1Plus, s4, s4Plus, s5, s5Plus = ss_chunks
        
        # Calculate variances for all areas
        y1Plus_var = calculate_noise_variance(noise, y1Plus, delta_tau, calculate_effective_tau_yPlus(params['tauY1'], p1Plus))
        y4Plus_var = calculate_noise_variance(noise, y4Plus, delta_tau, calculate_effective_tau_yPlus(params['tauY4'], p4Plus))
        y5Plus_var = calculate_noise_variance(noise, y5Plus, delta_tau, calculate_effective_tau_yPlus(params['tauY5'], p5Plus))
        
        u1Plus_var = calculate_noise_variance(noise, u1Plus, delta_tau, calculate_effective_tau_uPlus(params['tauU1'], params['b1'], u1, params['sigma1']))
        u4Plus_var = calculate_noise_variance(noise, u4Plus, delta_tau, calculate_effective_tau_uPlus(params['tauU4'], params['b4'], u4, params['sigma4']))
        u5Plus_var = calculate_noise_variance(noise, u5Plus, delta_tau, calculate_effective_tau_uPlus(params['tauU5'], params['b5'], u5, params['sigma5']))
        
        p1Plus_var = calculate_noise_variance(noise, p1Plus, delta_tau, calculate_effective_tau_pPlus(params['tauP1'], u1Plus))
        p4Plus_var = calculate_noise_variance(noise, p4Plus, delta_tau, calculate_effective_tau_pPlus(params['tauP4'], u4Plus))
        p5Plus_var = calculate_noise_variance(noise, p5Plus, delta_tau, calculate_effective_tau_pPlus(params['tauP5'], u5Plus))
        
        s1Plus_var = calculate_noise_variance(noise, s1Plus, delta_tau, params['tauS1'])
        s4Plus_var = calculate_noise_variance(noise, s4Plus, delta_tau, params['tauS4'])
        s5Plus_var = calculate_noise_variance(noise, s5Plus, delta_tau, params['tauS5'])
        
        noise_vec = np.ones(N) * noise
        var_list = np.concatenate([
            noise_vec, y1Plus_var, noise_vec, y4Plus_var, noise_vec, y5Plus_var,
            noise_vec, u1Plus_var, noise_vec, u4Plus_var, noise_vec, u5Plus_var,
            noise_vec, p1Plus_var, noise_vec, p4Plus_var, noise_vec, p5Plus_var,
            noise_vec, s1Plus_var, noise_vec, s4Plus_var, noise_vec, s5Plus_var,
        ])
    else:
        jacobian_dimension = int(N * model.jacobian_dimension)
        var_list = np.ones(jacobian_dimension) * noise
    
    L = np.diag(var_list)
    return torch.tensor(L, dtype=torch.float32)

def correlation(J, L, D,bw_y1_y2=False):
    """
    Returns the correlation matrix of the principal neurons in V1 and V2 at
    the specified neuron indices.
    """

    ## Full correlation matrix of the neurons
    # P = solve_continuous_lyapunov(J, -np.dot(L, np.dot(Q, L.T))) # The full correlation matrix
    A = (L @ D @ L.T)
    # A = 0.5 * (A + A.T) 
    P = lyap(J, A,method='scipy')
    
    P = P @ np.eye(P.shape[0]) # To make sure that the matrix is symmetric
    if bw_y1_y2:
        P = 2*(P**2)

    return P

def random_permutation(N1_y_idx: np.ndarray, N4_y_idx: np.ndarray, N5_y_idx: np.ndarray, 
                      n_V1_s: int, n_V1_t: int, n_V4_t: int, n_V5_t: int) -> tuple:
    """
    Returns the indices of randomly selected neurons for source and target areas.
    
    Args:
        N1_y_idx: Indices of V1 neurons
        N4_y_idx: Indices of V4 neurons
        N5_y_idx: Indices of V5 neurons
        n_V1_s: Number of source V1 neurons
        n_V1_t: Number of target V1 neurons
        n_V4_t: Number of target V4 neurons
        n_V5_t: Number of target V5 neurons
    
    Returns:
        tuple: (V1s_idx, V1t_idx, V4t_idx, V5t_idx)
    """
    # Randomly select source V1 neurons
    V1s_idx = np.random.permutation(N1_y_idx)[:n_V1_s]
    
    # Select target V1 neurons from remaining V1 neurons
    V1t_idx = np.setdiff1d(N1_y_idx, V1s_idx)
    V1t_idx = np.random.permutation(V1t_idx)[:n_V1_t]
    
    # Select target V4 and V5 neurons
    V4t_idx = np.random.permutation(N4_y_idx)[:n_V4_t]
    V5t_idx = np.random.permutation(N5_y_idx)[:n_V5_t]
    
    return V1s_idx, V1t_idx, V4t_idx, V5t_idx

def selection_mat(Py, V1s_idx, V1t_idx, V4t_idx, V5t_idx): 
    """
    Returns the submatrices required for the analysis.
    
    Args:
        Py: Full correlation matrix
        V1s_idx: Source V1 indices
        V1t_idx: Target V1 indices
        V4t_idx: Target V4 indices
        V5t_idx: Target V5 indices
    
    Returns:
        tuple: (P1, P2, P3, P4, P5, P6, P7) correlation submatrices
    """
    P1 = Py[np.ix_(V1s_idx, V1s_idx)]  # V1 source - V1 source
    P2 = Py[np.ix_(V1t_idx, V1t_idx)]  # V1 target - V1 target
    P3 = Py[np.ix_(V1s_idx, V1t_idx)]  # V1 source - V1 target
    P4 = Py[np.ix_(V4t_idx, V4t_idx)]  # V4 target - V4 target
    P5 = Py[np.ix_(V1s_idx, V4t_idx)]  # V1 source - V4 target
    P6 = Py[np.ix_(V5t_idx, V5t_idx)]  # V5 target - V5 target
    P7 = Py[np.ix_(V1s_idx, V5t_idx)]  # V1 source - V5 target
    
    return P1, P2, P3, P4, P5, P6, P7

def analysis_ss(mat, V1s_idx, V1t_idx, V4t_idx, V5t_idx):
    """
    Calculates predictive performance as function of dimensionality.
    
    Args:
        mat: Full correlation matrix
        V1s_idx: Source V1 indices
        V1t_idx: Target V1 indices
        V4t_idx: Target V4 indices
        V5t_idx: Target V5 indices
    
    Returns:
        tuple: (dims_V1, dims_V4, dims_V5, perf_V1, perf_V4, perf_V5)
    """
    # Select matrices for all areas
    P1, P2, P3, P4, P5, P6, P7 = selection_mat(mat, V1s_idx, V1t_idx, V4t_idx, V5t_idx)
    
    # Calculate performance for all areas
    perf_V1, dims_V1 = performance(P1, P2, P3)
    perf_V4, dims_V4 = performance(P1, P4, P5)
    perf_V5, dims_V5 = performance(P1, P6, P7)
    
    return dims_V1, dims_V4, dims_V5, perf_V1, perf_V4, perf_V5

def calculate_pred_performance_dim_3(model, pairs, contrast, method, com_params, delta_tau, noise, t_span=[0, 6]):
    """
    This function calculates the predictive performance as a function of
    predictive dimensions for the 3-area model.
    It returns the performance data as a dictionary.
    Inputs:
        model: The model object.
        pairs: [g, gamma1, gamma4]
        contrast: The contrast value.
        method: The method to be used for computing the Jacobian.
        com_params: The communication parameters.
        delta_tau: The delta_tau value.
        noise: The noise value.
        bw_y1_y2: Whether to use the LFPs instead of firing rates.
    Outputs:
        performance_data: The performance data as a dictionary.
    """

    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    performance_data = {}

    for g, gamma1, gamma4 in tqdm(pairs):
        updated_model = copy.deepcopy(model)
        updated_model.params['g1'] = g
        updated_model.params['gamma14'] = gamma1
        updated_model.params['gamma15'] = gamma4
        
        perf_V1 = np.zeros((com_params['num_trials'], com_params['V1_t']))
        perf_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))
        perf_V5 = np.zeros((com_params['num_trials'], com_params['V5_t']))

        # Get Jacobian
        J, ss = updated_model.get_Jacobian(contrast, initial_conditions, method, t_span)
        
        # Create S and L matrices
        S = create_S_matrix(updated_model)
        D = S**2
        L = create_L_matrix(updated_model, ss, delta_tau, noise)
      
        Py = correlation(J, L, D, com_params['bw_y1_y4'])

        for kl in range(com_params['num_trials']):
            V1s_idx, V1t_idx, V4t_idx, V5t_idx = random_permutation(
                com_params['N1_y_idx'], 
                com_params['N4_y_idx'], 
                com_params['N5_y_idx'], 
                com_params['V1_s'], 
                com_params['V1_t'], 
                com_params['V4_t'],
                com_params['V5_t']
            )
            dims_V1, dims_V4, dims_V5, perf_V1[kl, :], perf_V4[kl, :], perf_V5[kl,:] = analysis_ss(
                Py, V1s_idx, V1t_idx, V4t_idx, V5t_idx
            )

        # Store performance data in the dictionary
        performance_data[(g, gamma1, gamma4, contrast)] = {
            'V1': {
                'mean': np.mean(perf_V1, axis=0),
                'std': np.std(perf_V1, axis=0),
                'dims': dims_V1
            },
            'V4': {
                'mean': np.mean(perf_V4, axis=0),
                'std': np.std(perf_V4, axis=0),
                'dims': dims_V4
            },
            'V5': {
                'mean': np.mean(perf_V5, axis=0),
                'std': np.std(perf_V5, axis=0),
                'dims': dims_V5
            }
        }

    return performance_data

