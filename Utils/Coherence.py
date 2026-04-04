import numpy as np
import torch
import copy
from tqdm import tqdm
from Utils.matrix_spectrum import matrix_solution
from Models.Model import get_steady_states
from scipy.integrate import solve_ivp
import os
from Utils.matrix_spectrum import noise_power_spectrum


def Calculate_coherence(model, i, j, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, low_pass_add, rho, noise_sigma, noise_tau, contrast_vals, method, gamma_vals, min_freq=1, max_freq=5e2, n_freq_mat=100, t_span=[0, 6]):
    """
    Calculate coherence for all combinations of gamma and contrast values.
    Saves all data in a single file.
    """
    freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    params = model.params
    initial_conditions = np.ones((model.num_var * params['N'])) * 0.01
    S = create_S_matrix(model)

    coherence_data = {}

    for contrast in tqdm(contrast_vals, desc='Contrast'):
        for gamma in tqdm(gamma_vals, desc=f'Gamma (c={contrast})', leave=False):
            updated_params = copy.deepcopy(params)
            if fb_gain:
                updated_params['gamma1'] = gamma
            if input_gain_beta1:
                updated_params['beta1'] = gamma
            if input_gain_beta4:
                updated_params['beta4'] = gamma

            updated_model = copy.deepcopy(model)
            updated_model.params = updated_params

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            mat_model = matrix_solution(J, L, S, noise_sigma, noise_tau, low_pass_add=low_pass_add, rho=rho)
            coh_matrix, _ = mat_model.coherence(i=i, j=j, freq=freq_mat)

            key = (gamma, contrast)
            coherence_data[key] = {
                'freq': freq_mat.numpy(),
                'coh': np.abs(coh_matrix.numpy())
            }

    return coherence_data


def Calculate_power_spectra(model, i, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau,
                          noise_potential, noise_firing_rate, GR_noise, low_pass_add, rho,
                          noise_sigma, noise_tau, contrast_vals, method, gamma_vals,
                          tau_f, sigma_f, min_freq=None, max_freq=None, n_freq_mat=None, t_span=None):
    """
    Calculate power spectra for all combinations of gamma and contrast values.
    Uses frequency-dependent S matrix for filtering.
    """
    freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    params = model.params
    initial_conditions = np.ones((model.num_var * params['N'])) * 0.01

    power_data = {}

    for contrast in tqdm(contrast_vals, desc='Contrast'):
        for gamma in tqdm(gamma_vals, desc=f'Gamma (c={contrast})', leave=False):
            updated_params = copy.deepcopy(params)
            if fb_gain:
                updated_params['gamma1'] = gamma
            if input_gain_beta1:
                updated_params['beta1'] = gamma
            if input_gain_beta4:
                updated_params['beta4'] = gamma

            updated_model = copy.deepcopy(model)
            updated_model.params = updated_params

            J, ss = updated_model.get_Jacobian_augmented(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            S = create_S_matrix(updated_model)
            mat_model = matrix_solution(J, L, S, noise_sigma, noise_tau, low_pass_add=low_pass_add, rho=rho)
            power_matrix, _ = mat_model.auto_spectrum(i=i, freq=freq_mat)

            key = (gamma, contrast)
            power_data[key] = {
                'freq': freq_mat.numpy(),
                'power': np.abs(power_matrix.numpy())
            }

    return power_data


def create_L_matrix(model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise):
    '''
    Create L matrix for the model.
    noise_potential: noise in the potential variables
    noise_firing_rate: noise in the firing rate variables
    GR_noise: Whether to use Gaussian rectified noise for the firing rate variables
    '''
    N = model.params['N']
    params = model.params

    if model.simulate_firing_rates:
        if GR_noise:
            _, y1Plus, _, y4Plus, u1, u1Plus, u4, u4Plus, p1, p1Plus, p4, p4Plus, s1, s1Plus, s4, s4Plus = [
                ss[i*N:(i+1)*N] for i in range(model.jacobian_dimension)]
            y1Plus_var = calculate_noise_variance(noise_firing_rate, y1Plus, delta_tau,
                calculate_effective_tau_y(params['tauY1'], p1Plus))
            y4Plus_var = calculate_noise_variance(noise_firing_rate, y4Plus, delta_tau,
                calculate_effective_tau_y(params['tauY4'], p4Plus))
            u1Plus_var = calculate_noise_variance(noise_firing_rate, u1Plus, delta_tau,
                calculate_effective_tau_u(params['tauU1'], params['b1'], u1, params['sigma1']))
            u4Plus_var = calculate_noise_variance(noise_firing_rate, u4Plus, delta_tau,
                calculate_effective_tau_u(params['tauU4'], params['b4'], u4, params['sigma4']))
            p1Plus_var = calculate_noise_variance(noise_firing_rate, p1Plus, delta_tau,
                calculate_effective_tau_p(params['tauP1'], u1Plus))
            p4Plus_var = calculate_noise_variance(noise_firing_rate, p4Plus, delta_tau,
                calculate_effective_tau_p(params['tauP4'], u4Plus))
            s1Plus_var = calculate_noise_variance(noise_firing_rate, s1Plus, delta_tau, params['tauS1'])
            s4Plus_var = calculate_noise_variance(noise_firing_rate, s4Plus, delta_tau, params['tauS4'])
            noise_vec = np.ones(N) * noise_potential
            var_list = np.concatenate([
                noise_vec, y1Plus_var, noise_vec, y4Plus_var,
                noise_vec, u1Plus_var, noise_vec, u4Plus_var,
                noise_vec, p1Plus_var, noise_vec, p4Plus_var,
                noise_vec, s1Plus_var, noise_vec, s4Plus_var,
            ])
        else:
            noise_vec_potential = np.ones(N) * noise_potential
            noise_vec_firing_rate = np.ones(N) * noise_firing_rate
            var_list = np.concatenate([
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
                noise_vec_potential, noise_vec_firing_rate,
            ])
    else:
        jacobian_dimension = int(N * model.jacobian_dimension)
        var_list = np.ones(jacobian_dimension) * noise_potential

    # Augmented part for filtered noise variables
    num_membrane = 8
    m = num_membrane * N
    f_noise = np.ones(m) * (model.params['sigma_f'])
    var_list_aug = np.concatenate([var_list, f_noise])

    L = np.diag(var_list_aug)
    return torch.tensor(L, dtype=torch.float32)


def calculate_noise_variance(noise, ss_firing_rate, delta_tau, effective_tau):
    ratio = delta_tau / effective_tau
    exp_term = np.exp(ratio) - 1
    inside_sqrt = (ss_firing_rate) * (1 + 2 / exp_term)
    return noise * np.sqrt(np.maximum(0, inside_sqrt))


def calculate_effective_tau_y(tauY, a_ss):
    return (tauY * (1 + a_ss) / a_ss)


def calculate_effective_tau_u(tauU, b_ss, u, sigma):
    return (tauU * u / np.square(b_ss * sigma))


def calculate_effective_tau_p(tauP, uPlus_ss):
    return (tauP / (1 - uPlus_ss))


def create_S_matrix(model):
    N = model.params['N']

    if model.simulate_firing_rates:
        tau_list = [
            'tauY1', 'tauYPlus1', 'tauY4', 'tauYPlus4',
            'tauU1', 'tauUPlus1', 'tauU4', 'tauUPlus4',
            'tauP1', 'tauPPlus1', 'tauP4', 'tauPPlus4',
            'tauS1', 'tauSPlus1', 'tauS4', 'tauSPlus4',
        ]
    else:
        tau_list = [
            'tauY1', 'tauY4', 'tauU1', 'tauU4',
            'tauP1', 'tauP4', 'tauS1', 'tauS4',
        ]

    S_diag = []
    for tau in tau_list:
        S_diag.extend([np.sqrt(1 / model.params[tau])] * N)

    num_membrane = 8
    m = num_membrane * N
    f_entries = [1 / np.sqrt(model.params['tau_f'])] * m
    S_diag.extend(f_entries)
    S = np.diag(S_diag)
    return torch.tensor(S, dtype=torch.float32)
