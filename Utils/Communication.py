import numpy as np
from control import lyap
from tqdm import tqdm
from Utils.Coherence import create_S_matrix, create_L_matrix
import copy
import torch
from Utils.matrix_spectrum import matrix_solution

def correlation(J, L, D, bw_y1_y2=False):
    """
    Returns the Covariance matrix of the principal neurons in V1 and V2 at
    the specified neuron indices.
    """
    A = (L @ D @ L.T)
    P = lyap(J, A)
    P = P @ np.eye(P.shape[0])

    if bw_y1_y2:
        P = 2*(P**2)

    return P


def random_permutation(N1_y_idx, N4_y_idx, V1_s, V1_t, V4_s, V4_t, ss):
    """
    Randomly select neurons for analysis, considering only those with significant activity.

    Args:
        N1_y_idx: Indices for V1 neurons
        N4_y_idx: Indices for V4 neurons
        V1_s: Number of source neurons to select from V1
        V1_t: Number of target neurons to select from V1
        V4_s: Number of source neurons to select from V4
        V4_t: Number of target neurons to select from V4
        ss: Steady state values containing firing rates
    """
    V1_rates = ss[N1_y_idx]
    V4_rates = ss[N4_y_idx]

    V1_threshold = 0.0000
    V4_threshold = 0.0000

    V1_active = N1_y_idx[V1_rates >= V1_threshold]
    V4_active = N4_y_idx[V4_rates >= V4_threshold]

    if len(V1_active) < (V1_s + V1_t) or len(V4_active) < (V4_s + V4_t):
        raise ValueError("Not enough active neurons above threshold")

    V1_perm = np.random.permutation(V1_active)
    V4_perm = np.random.permutation(V4_active)

    V1s_idx = V1_perm[:V1_s]
    V1t_idx = V1_perm[V1_s:V1_s+V1_t]
    V4s_idx = V4_perm[:V4_s]
    V4t_idx = V4_perm[V4_s:V4_s+V4_t]

    return V1s_idx, V1t_idx, V4s_idx, V4t_idx

def selection_mat(Py, V1s_idx, V1t_idx, V4s_idx, V4t_idx):
    """
    Returns the submatrices required for the analysis. See main text for the
    definition of Pi.
    """
    # For V1-V1 communication
    P1 = Py[np.ix_(V1s_idx, V1s_idx)]  # V1s to V1s
    P2 = Py[np.ix_(V1t_idx, V1t_idx)]  # V1t to V1t
    P3 = Py[np.ix_(V1s_idx, V1t_idx)]  # V1s to V1t

    # For V1-V4 communication
    P4 = Py[np.ix_(V4t_idx, V4t_idx)]  # V4t to V4t
    P5 = Py[np.ix_(V1s_idx, V4t_idx)]  # V1s to V4t

    # For V4-V4 communication
    P6 = Py[np.ix_(V4s_idx, V4s_idx)]  # V4s to V4s
    P7 = Py[np.ix_(V4s_idx, V4t_idx)]  # V4s to V4t

    return P1, P2, P3, P4, P5, P6, P7

def analysis_ss(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    P1, P2, P3, P4, P5, P6, P7 = selection_mat(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx)

    perf_V1_V1, dims_V1_V1 = performance(P1, P2, P3)
    perf_V1_V4, dims_V1_V4 = performance(P1, P4, P5)
    perf_V4_V4, dims_V4_V4 = performance(P6, P4, P7)

    return dims_V1_V1, dims_V1_V4, dims_V4_V4, perf_V1_V1, perf_V1_V4, perf_V4_V4


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
    C1, C2, C3 = P1, P2, P3

    # Calculate the predicted covariance matrix, C4 = C3^T * C1^-1 * C3
    try:
        inv_C1_C3 = np.linalg.solve(C1, C3)
    except np.linalg.LinAlgError:
        inv_C1_C3 = np.linalg.pinv(C1) @ C3
    C4 = C3.T @ inv_C1_C3

    total_target_variance = np.trace(C2)

    dim = C2.shape[0]
    dims = np.arange(0, dim)

    eig_vals, _ = np.linalg.eigh(C4)
    eig_vals = np.sort(eig_vals)[::-1]

    cumulative_explained_variance = np.cumsum(eig_vals)

    pred_perf = np.zeros(dim)
    perf_for_ranks_1_to_dim = cumulative_explained_variance / total_target_variance

    if dim > 1:
        pred_perf[1:] = perf_for_ranks_1_to_dim[:-1]

    return pred_perf, dims


def analysis_alignment(mat, V1s_idx, V1t_idx, V2s_idx, V2t_idx):
    """
    This function does the analysis to calculate the alignment between
    eigenvectors of C1 and left singular vectors of C3.
    """
    C1, _, C3, _, C5, _, _ = selection_mat(mat, V1s_idx, V1t_idx, V2s_idx, V2t_idx)

    eigvals_C1, eigvecs_C1 = np.linalg.eigh(C1)
    U, _, _ = np.linalg.svd(C3)

    idx1 = np.argsort(eigvals_C1)[::-1]
    eigvecs_C1 = eigvecs_C1[:, idx1]

    alignment_v1v1 = np.abs(eigvecs_C1.T @ U)

    U_C5, _, _ = np.linalg.svd(C5)
    alignment_v1v2 = np.abs(eigvecs_C1.T @ U_C5)

    return alignment_v1v1, alignment_v1v2

def Calculate_Alignment(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params, t_span=[0,6]):
    N = model.params['N']
    params = model.params
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    alignment_data = {}

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

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            S = create_S_matrix(updated_model)
            D = S**2
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            Py = correlation(J, L, D, com_params['bw_y1_y2'])

            align_v1_v1_trials = []
            align_v1_v2_trials = []

            for _ in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V2s_idx, V2t_idx = random_permutation(
                    com_params['N1_y_idx'],
                    com_params['N2_y_idx'],
                    com_params['V1_s'],
                    com_params['V1_t'],
                    com_params['V2_s'],
                    com_params['V2_t'],
                    ss
                )
                alignment_v1v1, alignment_v1v2 = analysis_alignment(Py, V1s_idx, V1t_idx, V2s_idx, V2t_idx)
                align_v1_v1_trials.append(alignment_v1v1)
                align_v1_v2_trials.append(alignment_v1v2)

            alignment_data[gamma, contrast] = {
                'V1_V1': {
                    'mean': np.mean(align_v1_v1_trials, axis=0),
                    'std': np.std(align_v1_v1_trials, axis=0),
                },
                'V1_V2': {
                    'mean': np.mean(align_v1_v2_trials, axis=0),
                    'std': np.std(align_v1_v2_trials, axis=0),
                }
            }

    return alignment_data

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

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            S = create_S_matrix(updated_model)
            D = S**2
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            Py = correlation(J, L, D, com_params['bw_y1_y4'])

            perf_V1_V1 = np.zeros((com_params['num_trials'], com_params['V1_t']))
            perf_V1_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))
            perf_V4_V4 = np.zeros((com_params['num_trials'], com_params['V4_t']))

            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4s_idx, V4t_idx = random_permutation(
                    com_params['N1_y_idx'],
                    com_params['N4_y_idx'],
                    com_params['V1_s'],
                    com_params['V1_t'],
                    com_params['V4_s'],
                    com_params['V4_t'],
                    ss
                )
                dims_V1_V1, dims_V1_V4, dims_V4_V4, perf_V1_V1[kl, :], perf_V1_V4[kl, :], perf_V4_V4[kl, :] = analysis_ss(Py, V1s_idx, V1t_idx, V4s_idx, V4t_idx)

            performance_data[gamma, contrast] = {
                'V1_V1': {
                    'mean': np.mean(perf_V1_V1, axis=0),
                    'std': np.std(perf_V1_V1, axis=0),
                    'dims': dims_V1_V1
                },
                'V1_V2': {
                    'mean': np.mean(perf_V1_V4, axis=0),
                    'std': np.std(perf_V1_V4, axis=0),
                    'dims': dims_V1_V4
                },
                'V2_V2': {
                    'mean': np.mean(perf_V4_V4, axis=0),
                    'std': np.std(perf_V4_V4, axis=0),
                    'dims': dims_V4_V4
                }
            }
            covariance_data[gamma, contrast] = Py

    return performance_data, covariance_data

def Calculate_Covariance_mean(model, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, method, com_params, delta_tau, noise_potential, noise_firing_rate, GR_noise, t_span=[0,6]):
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

        J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)

        S = create_S_matrix(updated_model)
        D = S**2
        L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

        Py = correlation(J, L, D, com_params['bw_y1_y4'])

    return Py, ss

###################### For frequency Decomposition of CS analysis ######################


def calculate_pred_performance_freq(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4,
                                     delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params,
                                    freq, t_span=None):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01
    performance_data = {}

    for gamma in tqdm(gamma_vals):
        for contrast in tqdm(contrast_vals, leave=False):
            perf_V1_V1 = np.zeros((com_params['num_trials'], len(freq)))
            perf_V1_V4 = np.zeros((com_params['num_trials'], len(freq)))
            perf_V4_V4 = np.zeros((com_params['num_trials'], len(freq)))

            updated_model = copy.deepcopy(model)
            if fb_gain:
                updated_model.params['gamma1'] = gamma
            elif input_gain_beta1:
                updated_model.params['beta1'] = gamma
            elif input_gain_beta4:
                updated_model.params['beta4'] = gamma

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            S = create_S_matrix(updated_model)
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            mat_model = matrix_solution(J, L, S)
            S_fij = mat_model.spectral_matrix(freq, J)
            S_fij = 2 * torch.real(S_fij)

            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4s_idx, V4t_idx = random_permutation(
                    com_params['N1_y_idx'], com_params['N4_y_idx'],
                    com_params['V1_s'], com_params['V1_t'], com_params['V4_s'], com_params['V4_t'], ss
                )
                for f_idx, _ in enumerate(freq):
                    Py = S_fij[f_idx].cpu().numpy()
                    perf_V1_V1[kl, f_idx], perf_V1_V4[kl, f_idx], perf_V4_V4[kl, f_idx] = analysis_ss_freq(Py, V1s_idx, V1t_idx, V4s_idx, V4t_idx)

            performance_data[gamma, contrast] = {
                'V1_V1': {
                    'mean': np.mean(perf_V1_V1, axis=0),
                    'std': np.std(perf_V1_V1, axis=0),
                },
                'V1_V2': {
                    'mean': np.mean(perf_V1_V4, axis=0),
                    'std': np.std(perf_V1_V4, axis=0),
                },
                'V2_V2': {
                    'mean': np.mean(perf_V4_V4, axis=0),
                    'std': np.std(perf_V4_V4, axis=0),
                }
            }
            print(f"Completed gamma={gamma:.3f}, contrast={contrast:.3f}")

    return performance_data

def performance_all_dimensions(P1, P2, P3):
    """
    This function returns the prediction performance for the highest dimension
    given the required correlation matrices.
    """
    W_opt = np.linalg.inv(P1) @ P3

    U, S, V = np.linalg.svd(W_opt)
    sing_vals = S
    dim = min(S.shape)
    dims = np.arange(0, dim)
    vec = np.zeros(S.shape)
    vec[:dim] = sing_vals[:dim]

    S_new = np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(S_new[:dim, :dim], vec)
    W_re = np.dot(U, np.dot(S_new, V))
    e = np.trace(P2 + W_re.T @ P1 @ W_re - 2 * W_re.T @ P3)

    e_R = np.trace(P2)
    pred_perf = 1 - (e / e_R)

    return pred_perf

def analysis_ss_freq(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    P1, P2, P3, P4, P5, P6, P7 = selection_mat(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx)

    perf_V1_V1 = performance_all_dimensions(P1, P2, P3)
    perf_V1_V4 = performance_all_dimensions(P1, P4, P5)
    perf_V4_V4 = performance_all_dimensions(P6, P4, P7)

    return perf_V1_V1, perf_V1_V4, perf_V4_V4


#################### For Dimensionality vs Frequency analysis ######################

def calculate_dim_vs_freq(model, gamma_vals, contrast_vals, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, method, com_params, freq, thresold, t_span=[0, 6]):
    N = model.params['N']
    initial_conditions = np.ones((model.num_var * N)) * 0.01

    dimension_data = {}

    for gamma in tqdm(gamma_vals):
        for contrast in tqdm(contrast_vals, leave=False):
            dim_V1_V1 = np.zeros((com_params['num_trials'], len(freq)))
            dim_V1_V4 = np.zeros((com_params['num_trials'], len(freq)))
            dim_V4_V4 = np.zeros((com_params['num_trials'], len(freq)))

            updated_model = copy.deepcopy(model)
            if fb_gain:
                updated_model.params['gamma1'] = gamma
            elif input_gain_beta1:
                updated_model.params['beta1'] = gamma
            elif input_gain_beta4:
                updated_model.params['beta4'] = gamma

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            S = create_S_matrix(updated_model)
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            mat_model = matrix_solution(J, L, S)
            S_fij = mat_model.spectral_matrix(freq, J)
            S_fij = 2 * torch.real(S_fij)

            for kl in range(com_params['num_trials']):
                V1s_idx, V1t_idx, V4s_idx, V4t_idx = random_permutation(com_params['N1_y_idx'], com_params['N4_y_idx'], com_params['V1_s'], com_params['V1_t'], com_params['V4_s'], com_params['V4_t'], ss)
                for f_idx, _ in enumerate(freq):
                    Py = S_fij[f_idx].cpu().numpy()
                    dim_V1_V1[kl, f_idx], dim_V1_V4[kl, f_idx], dim_V4_V4[kl, f_idx] = analysis_dim_vs_freq(Py, V1s_idx, V1t_idx, V4s_idx, V4t_idx, thresold)

            dimension_data[gamma, contrast] = {
                'V1_V1': {
                    'mean': np.mean(dim_V1_V1, axis=0),
                    'std': np.std(dim_V1_V1, axis=0),
                },
                'V1_V2': {
                    'mean': np.mean(dim_V1_V4, axis=0),
                    'std': np.std(dim_V1_V4, axis=0),
                },
                'V2_V2': {
                    'mean': np.mean(dim_V4_V4, axis=0),
                    'std': np.std(dim_V4_V4, axis=0),
                }
            }
            print(f"Completed gamma={gamma:.3f}, contrast={contrast:.3f}")

    return dimension_data

def analysis_dim_vs_freq(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx, threshold):
    """
    This function does the analysis to calculate the predictive
    performance as a function of the dimensionality.
    """
    P1, P2, P3, P4, P5, P6, P7 = selection_mat(mat, V1s_idx, V1t_idx, V4s_idx, V4t_idx)

    dim_V1_V1 = Perf_dim_vs_freq(P1, P2, P3, threshold)
    dim_V1_V4 = Perf_dim_vs_freq(P1, P4, P5, threshold)
    dim_V4_V4 = Perf_dim_vs_freq(P6, P4, P7, threshold)

    return dim_V1_V1, dim_V1_V4, dim_V4_V4

def Perf_dim_vs_freq(P1, P2, P3, threshold):
    """
    Wrapper function that returns the dimension when the prediction performance
    reaches 95% of its highest value.
    """
    pred_perf, dims = performance(P1, P2, P3)

    max_perf = np.max(pred_perf)
    threshold_perf = threshold * max_perf

    dim_95 = next((dim for dim, perf in zip(dims, pred_perf) if perf >= threshold_perf), None)

    return dim_95
