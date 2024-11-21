import numpy as np
import torch
import copy
from tqdm import tqdm
from matrix_spectrum import matrix_solution



def calculate_coherence(model, i, j,fb_gain,input_gain_beta1,input_gain_beta4, delta_tau, noise,baseline,poisson, contrast, method, gamma_vals, min_freq=1, max_freq=5e2, n_freq_mat=100, t_span=[0, 6]):
    freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    params = model.params
    initial_conditions = np.ones((model.num_var * params['N'])) * 0.01
    S = create_S_matrix(model) # Just the timescales

    coherence_data = {}

    for gamma in tqdm(gamma_vals):
        updated_params = copy.deepcopy(params)
        if fb_gain:
            updated_params['gamma1'] = gamma
        if input_gain_beta1:
            updated_params['beta1'] = gamma
        if input_gain_beta4:
            updated_params['beta4'] = gamma
       

        # Create a copy of the model with updated parameters
        updated_model = copy.deepcopy(model)
        updated_model.params = updated_params 

        # Get Jacobian
        J, ss = updated_model.get_Jacobian(contrast,initial_conditions,  method, t_span)
        J = torch.tensor(J, dtype=torch.float32)

        # Create L matrix
        L = create_L_matrix(updated_model, ss, delta_tau, noise, baseline=baseline, poisson=poisson, contrast=contrast)   

        # Calculate coherence
        mat_model = matrix_solution(J, L, S)
        coh_matrix, _ = mat_model.coherence(i=i, j=j, freq=freq_mat)

        key = f'contrast={contrast}_gamma={gamma}'
        coherence_data[key] = {
            'freq': freq_mat.numpy(),
            'coh': np.abs(coh_matrix.numpy())
        }

    return coherence_data

def Calculate_power_spectra(model, i,fb_gain,input_gain_beta1,input_gain_beta4, delta_tau, noise,baseline,poisson, contrast, method, gamma_vals, min_freq=1, max_freq=5e2, n_freq_mat=100, t_span=[0, 6]):
    freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    params = model.params
    initial_conditions = np.ones((model.num_var * params['N'])) * 0.01
    S = create_S_matrix(model) # Just the timescales

    power_data = {}

    for gamma in tqdm(gamma_vals):
        updated_params = copy.deepcopy(params)
        if fb_gain:
            updated_params['gamma1'] = gamma
        if input_gain_beta1:
            updated_params['beta1'] = gamma
        if input_gain_beta4:
            updated_params['beta4'] = gamma
       

        # Create a copy of the model with updated parameters
        updated_model = copy.deepcopy(model)
        updated_model.params = updated_params 

        # Get Jacobian
        J, ss = updated_model.get_Jacobian(contrast,initial_conditions,  method, t_span)
        J = torch.tensor(J, dtype=torch.float32)

        # Create L matrix
        L = create_L_matrix(updated_model, ss, delta_tau, noise, baseline=baseline, poisson=poisson, contrast=contrast)

        # Calculate coherence
        mat_model = matrix_solution(J, L, S)
        power_matrix, _ = mat_model.auto_spectrum(i=i, freq=freq_mat)

        # key = f'contrast={contrast}_gamma={gamma}'
        key = gamma, contrast
        power_data[key] = {
            'freq': freq_mat.numpy(),
            'power': np.abs(power_matrix.numpy())
        }

    return power_data
    


# def calculate_noise_variance(noise,ss_firing_rate,delta_tau,effective_tau):
#     ratio = delta_tau/effective_tau
#     # print(f'effective_tau: {effective_tau}')
#     exp_term = np.exp(ratio) - 1
#     # print(f"ss_firing_rate: {ss_firing_rate}")
#     # print(f"exp_term: {exp_term}")
#     inside_sqrt = ss_firing_rate * (1 + 2 / exp_term)
#     # print(f"inside_sqrt : {inside_sqrt}")
#     # print(ss_firing_rate[18],ratio[18])
#     return  noise * np.sqrt( inside_sqrt)
def calculate_noise_variance(noise, ss_firing_rate, delta_tau, effective_tau, baseline=None,poisson=False):
    if poisson:
        return noise * np.sqrt(ss_firing_rate)
    else:
        ratio = delta_tau / effective_tau
        exp_term = np.exp(ratio) - 1
        inside_sqrt = (ss_firing_rate ) * (1 + 2 / exp_term) - baseline
        return noise * np.sqrt(np.maximum(inside_sqrt, 0))

    
def calculate_effective_tau_yPlus(tauY, a_ss, contrast):
    # Increase effective tau with contrast
    contrast_factor = 1 + contrast**2  # Linear scaling with contrast
    return (tauY * (1 + a_ss) / a_ss) * contrast_factor

def calculate_effective_tau_uPlus(tauU, b_ss, u, sigma, contrast):
    # Increase effective tau with contrast
    contrast_factor = 1 + contrast**2  # Linear scaling with contrast
    return (tauU * u / np.square(b_ss * sigma)) * contrast_factor

def calculate_effective_tau_pPlus(tauP, uPlus_ss, contrast):
    # Increase effective tau with contrast
    contrast_factor = 1 + contrast**2  # Linear scaling with contrast
    return (tauP/(1-uPlus_ss)) * contrast_factor

########################################################################################
# def calculate_effective_tau_yPlus(tauY, a_ss, contrast):
#     # Increase effective tau with contrast
#     contrast_factor = 1 + contrast  # Linear scaling with contrast
#     return tauY * (1 + a_ss) / a_ss * contrast_factor

# def calculate_effective_tau_uPlus(tauU, b_ss, u, sigma, contrast):
#     # Increase effective tau with contrast
#     contrast_factor = 1 + contrast  # Linear scaling with contrast
#     return tauU * u / np.square(b_ss * sigma) * contrast_factor

# def calculate_effective_tau_pPlus(tauP, uPlus_ss, contrast):
#     # Increase effective tau with contrast
#     contrast_factor = 1 + contrast  # Linear scaling with contrast
#     return tauP/(1-uPlus_ss) * contrast_factor


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
    
    S = np.diag(S_diag)
    return torch.tensor(S, dtype=torch.float32)

def create_L_matrix(model, ss, delta_tau, noise, baseline, poisson, contrast):
    N = model.params['N']
    # add a small constant to the ss
    # ss = ss + 1e-2
    params = model.params
    if model.simulate_firing_rates:
        _, y1Plus, _, y4Plus, u1, u1Plus, u4, u4Plus, p1, p1Plus, p4, p4Plus, s1, s1Plus, s4, s4Plus = [ss[i*N:(i+1)*N] for i in range(model.jacobian_dimension)]
        
        y1Plus_var = calculate_noise_variance(noise, y1Plus, delta_tau, 
            calculate_effective_tau_yPlus(params['tauY1'], p1Plus, contrast),
            baseline=baseline, poisson=poisson)
        y4Plus_var = calculate_noise_variance(noise, y4Plus, delta_tau, 
            calculate_effective_tau_yPlus(params['tauY4'], p4Plus, contrast),
            baseline=baseline, poisson=poisson)
        u1Plus_var = calculate_noise_variance(noise, u1Plus, delta_tau, 
            calculate_effective_tau_uPlus(params['tauU1'], params['b1'], u1, params['sigma1'], contrast),
            baseline=baseline, poisson=poisson)
        u4Plus_var = calculate_noise_variance(noise, u4Plus, delta_tau, 
            calculate_effective_tau_uPlus(params['tauU4'], params['b4'], u4, params['sigma4'], contrast),
            baseline=baseline, poisson=poisson)
        p1Plus_var = calculate_noise_variance(noise, p1Plus, delta_tau, 
            calculate_effective_tau_pPlus(params['tauP1'], u1Plus, contrast),
            baseline=baseline, poisson=poisson)
        p4Plus_var = calculate_noise_variance(noise, p4Plus, delta_tau, 
            calculate_effective_tau_pPlus(params['tauP4'], u4Plus, contrast),
            baseline=baseline, poisson=poisson)
        s1Plus_var = calculate_noise_variance(noise, s1Plus, delta_tau, params['tauS1'],baseline=baseline, poisson=poisson)
        s4Plus_var = calculate_noise_variance(noise, s4Plus, delta_tau, params['tauS4'],baseline=baseline, poisson=poisson)
        
        noise_vec = np.ones(N) * noise
        var_list = np.concatenate([
            noise_vec, y1Plus_var, noise_vec, y4Plus_var, 
            noise_vec, u1Plus_var, noise_vec, u4Plus_var, 
            noise_vec, p1Plus_var, noise_vec, p4Plus_var, 
            noise_vec, s1Plus_var, noise_vec, s4Plus_var,
            
        ])
    else:
        jacobian_dimension = int(N * model.jacobian_dimension)
        var_list = np.ones(jacobian_dimension) * noise
    
    L = np.diag(var_list)
    return torch.tensor(L, dtype=torch.float32)


