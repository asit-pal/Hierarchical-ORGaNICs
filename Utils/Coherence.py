import numpy as np
import torch
import copy
from tqdm import tqdm
from Utils.matrix_spectrum import matrix_solution
from Models.Model import get_steady_states
from scipy.integrate import solve_ivp
import os
from Utils.matrix_spectrum import noise_power_spectrum


def Calculate_coherence(model, i, j, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, low_pass_add, noise_sigma, noise_tau, contrast_vals, method, gamma_vals, min_freq=1, max_freq=5e2, n_freq_mat=100, t_span=[0, 6]):
    """
    Calculate coherence for all combinations of gamma and contrast values.
    Saves all data in a single file.
    """
    freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
    params = model.params
    initial_conditions = np.ones((model.num_var * params['N'])) * 0.01
    S = create_S_matrix(model) # Just the timescales

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
           
            # Create a copy of the model with updated parameters
            updated_model = copy.deepcopy(model)
            updated_model.params = updated_params 

            # Get Jacobian
            J, ss = updated_model.get_Jacobian(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)

            # Create L matrix
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

            # Calculate coherence
            mat_model = matrix_solution(J, L, S, noise_sigma, noise_tau, low_pass_add=low_pass_add)
            coh_matrix, _ = mat_model.coherence(i=i, j=j, freq=freq_mat)

            # Store data with (gamma, contrast) tuple as key
            key = (gamma, contrast)
            coherence_data[key] = {
                'freq': freq_mat.numpy(),
                'coh': np.abs(coh_matrix.numpy())
            }

    return coherence_data

# def Calculate_power_spectra(model, i, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, noise_potential, noise_firing_rate, GR_noise, low_pass_add, noise_sigma, noise_tau, contrast_vals, method, gamma_vals,tau_f, min_freq=0.1, max_freq=200, n_freq_mat=500, t_span=[0, 6]):
#     """
#     Calculate power spectra for all combinations of gamma and contrast values.
#     Saves all data in a single file.
#     """
#     freq_mat = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq_mat)
#     params = model.params
#     initial_conditions = np.ones((model.num_var * params['N'])) * 0.01
#     S = create_S_matrix(model) # Just the timescales

#     power_data = {}

#     for contrast in tqdm(contrast_vals, desc='Contrast'):
#         for gamma in tqdm(gamma_vals, desc=f'Gamma (c={contrast})', leave=False):
#             updated_params = copy.deepcopy(params)
#             if fb_gain:
#                 updated_params['gamma1'] = gamma
#             if input_gain_beta1:
#                 updated_params['beta1'] = gamma
#             if input_gain_beta4:
#                 updated_params['beta4'] = gamma
           
#             # Create a copy of the model with updated parameters
#             updated_model = copy.deepcopy(model)
#             updated_model.params = updated_params 

#             # Get Jacobian
#             J, ss = updated_model.get_Jacobian(contrast, initial_conditions, method, t_span)
#             J = torch.tensor(J, dtype=torch.float32)
#             # Create L matrix
#             L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, GR_noise)

#             # Calculate coherence
#             mat_model = matrix_solution(J, L, S, noise_sigma, noise_tau, low_pass_add=low_pass_add)
#             power_matrix, _ = mat_model.auto_spectrum(i=i, freq=freq_mat)
            
#             # if low_pass:
#             #     noise_power = noise_power_spectrum(freq_mat,sigma=0.05,tau=300*1e-3)
#             #     power_matrix = power_matrix + noise_power
                   

#             # Store data with (gamma, contrast) tuple as key
#             key = (gamma, contrast)
#             power_data[key] = {
#                 'freq': freq_mat.numpy(),
#                 'power': np.abs(power_matrix.numpy())
#             }

#     return power_data

def Calculate_power_spectra(model, i, fb_gain, input_gain_beta1, input_gain_beta4, delta_tau, 
                          noise_potential, noise_firing_rate,sigma_f, GR_noise, low_pass_add, 
                          noise_sigma, noise_tau, contrast_vals, method, gamma_vals, 
                           min_freq=None, max_freq=None, n_freq_mat=None, t_span=None):
    """
    Calculate power spectra for all combinations of gamma and contrast values.
    Now uses frequency-dependent S matrix for filtering.
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
           
            # Create a copy of the model with updated parameters
            updated_model = copy.deepcopy(model)
            updated_model.params = updated_params 

            # Get Jacobian
            J, ss = updated_model.get_Jacobian(contrast, initial_conditions, method, t_span)
            J = torch.tensor(J, dtype=torch.float32)
            
            # Create L matrix with noise only in filtered variables
            L = create_L_matrix(updated_model, ss, delta_tau, noise_potential, noise_firing_rate, sigma_f, GR_noise)
            
            # Create S matrix
            S = create_S_matrix(updated_model)
            
            # Calculate power spectrum
            mat_model = matrix_solution(J, L, S, noise_sigma, noise_tau, low_pass_add=low_pass_add)
            power_matrix = torch.zeros_like(freq_mat)
            
            for f_idx, freq in enumerate(freq_mat):
                power_at_freq, _ = mat_model.auto_spectrum(i=i, freq=freq.unsqueeze(0))
                power_matrix[f_idx] = power_at_freq

            # Store data
            key = (gamma, contrast)
            power_data[key] = {
                'freq': freq_mat.numpy(),
                'power': np.abs(power_matrix.numpy())
            }

    return power_data

def create_L_matrix(model, ss, delta_tau, noise_potential, noise_firing_rate, sigma_f, GR_noise):
    '''
    Create noise covariance matrix L.
    Noise is only added to the filtered variables (fy, fu, fp, fs) and is zero elsewhere.
    
    Args:
    model: RingModel instance
    ss: Steady state of the system
    delta_tau: Time window for noise integration
    noise_potential: Noise amplitude for membrane potentials
    noise_firing_rate: Noise amplitude for firing rates
    GR_noise: Whether to use Gaussian rectified noise for the firing rate variables
    '''
    N = model.params['N']
        
    params = model.params

    # Make sigma_f contrast dependent if contrast is defined in model
    # if hasattr(model, 'contrast'):
    #     k_sigma = 0.5  # constant to scale contrast dependence
    #     effective_sigma_f = sigma_f / (1 + k_sigma * model.contrast)
    # else:
    #     effective_sigma_f = sigma_f

    if model.simulate_firing_rates:
        if GR_noise:    
            # Unpack steady states, including filtered variables
            y1, y1Plus, y4, y4Plus, u1, u1Plus, u4, u4Plus, p1, p1Plus, p4, p4Plus, s1, s1Plus, s4, s4Plus, \
            fy1, fu1, fp1, fs1, fy4, fu4, fp4, fs4 = [ss[i*N:(i+1)*N] for i in range(model.jacobian_dimension)]
            
            # Calculate variances for original variables
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
            
            # Create noise vector for membrane potentials
            noise_vec = np.ones(N) * noise_potential 
            
            # Create noise vector for filtered variables using sigma_f directly
            # filtered_noise_vec = np.ones(N) * sigma_f
            
            # Concatenate all variances
            var_list = np.concatenate([
                noise_vec, y1Plus_var, noise_vec, y4Plus_var, 
                noise_vec, u1Plus_var, noise_vec, u4Plus_var, 
                noise_vec, p1Plus_var, noise_vec, p4Plus_var, 
                noise_vec, s1Plus_var, noise_vec, s4Plus_var,
                y1Plus_var*sigma_f,u1Plus_var*sigma_f,p1Plus_var*sigma_f,s1Plus_var*sigma_f,
                y4Plus_var*sigma_f,u4Plus_var*sigma_f,p4Plus_var*sigma_f,s4Plus_var*sigma_f,
                # filtered_noise_vec, filtered_noise_vec, filtered_noise_vec, filtered_noise_vec,  # Area 1 filtered vars
                # filtered_noise_vec, filtered_noise_vec, filtered_noise_vec, filtered_noise_vec   # Area 4 filtered vars
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
                noise_vec_potential, noise_vec_potential, noise_vec_potential, noise_vec_potential,  # Area 1 filtered vars
                noise_vec_potential, noise_vec_potential, noise_vec_potential, noise_vec_potential   # Area 4 filtered vars
            ])
    else:
        jacobian_dimension = int(N * model.jacobian_dimension)
        var_list = np.ones(jacobian_dimension) * noise_potential
    
    L = np.diag(var_list)
    return torch.tensor(L, dtype=torch.float32)



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
def calculate_noise_variance(noise, ss_firing_rate, delta_tau, effective_tau):
    ratio = delta_tau / effective_tau
    exp_term = np.exp(ratio) - 1
    inside_sqrt = (ss_firing_rate ) * (1 + 2 / exp_term) 
    return noise * np.sqrt(np.maximum(0, inside_sqrt))
    # return noise * np.ones_like(ss_firing_rate)

def calculate_effective_tau_y(tauY, a_ss):
    # Increase effective tau with contrast, but saturate the effect with a constant k
    # k = 0.1  # small constant to prevent tau_effective from dropping too low
    return (tauY * (1 + a_ss) / (a_ss ))

def calculate_effective_tau_u(tauU, b_ss, u, sigma):
    # Increase effective tau with contrast
    return (tauU * u / np.square(b_ss * sigma)) 

def calculate_effective_tau_p(tauP, uPlus_ss):
    # Increase effective tau with contrast
    return (tauP/(1-uPlus_ss)) 

# def get_effective_timescales(model, c=None, initial_conditions=None, ss=None, method='RK45', t_span=[0, 5], decay_time=None, plot=None):
#     # Create a deep copy of the model to avoid modifying the original
    
    
#     # Get steady state if not provided
#     if ss is None:
#         if c is None or initial_conditions is None:
#             raise ValueError("Must provide either ss or both c and initial_conditions")
#         ss = get_steady_states(model, c, initial_conditions, t_span, method, Jacobian=True)
        
#     model_copy = copy.deepcopy(model)
    
#     # Set alpha parameters to zero
#     model_copy.params['alpha1'] = 0
#     model_copy.params['alpha4'] = 0
#     ss = np.array(ss).flatten()
    
#     # Setup for decay measurement
#     t_decay = np.arange(0, decay_time, model_copy.params['dt'])
#     decay_span = [0, decay_time]
    
#     # Turn off input and simulate decay
#     def decay_func(t, y):
#         x = np.zeros(model_copy.M)  # Zero input
#         return model_copy.dynm_func(t, y, x)
    
#     # Simulate decay
#     sol = solve_ivp(decay_func, decay_span, ss, method=method, t_eval=t_decay,
#                     vectorized=True, rtol=1e-10, atol=1e-10)
    
#     # Calculate time constants from decay
#     decay_curves = sol.y.reshape(model_copy.num_var, model_copy.N, -1)
    
#     N = model_copy.params['N']
#     y1, y1Plus, y4, y4Plus, u1, u1Plus, u4, u4Plus, p1, p1Plus, p4, p4Plus, s1, s1Plus, s4, s4Plus = [
#         ss[i*N:(i+1)*N] for i in range(model_copy.jacobian_dimension)
#     ]
    
#     # Calculate analytical taus for each neuron and variable type
#     analytical_taus = np.zeros((8, N))  # 8 variable types, N neurons each
#     for n in range(N):
#         analytical_taus[0, n] = calculate_effective_tau_y(model_copy.params['tauY1'], p1Plus[n])  # y1
#         analytical_taus[1, n] = calculate_effective_tau_y(model_copy.params['tauY4'], p4Plus[n])  # y4
#         analytical_taus[2, n] = calculate_effective_tau_u(model_copy.params['tauU1'], model_copy.params['b1'], u1[n], model_copy.params['sigma1'])  # u1
#         analytical_taus[3, n] = calculate_effective_tau_u(model_copy.params['tauU4'], model_copy.params['b4'], u4[n], model_copy.params['sigma4'])  # u4
#         analytical_taus[4, n] = calculate_effective_tau_p(model_copy.params['tauP1'], u1Plus[n])  # p1
#         analytical_taus[5, n] = calculate_effective_tau_p(model_copy.params['tauP4'], u4Plus[n])  # p4
#         analytical_taus[6, n] = model_copy.params['tauS1']  # s1
#         analytical_taus[7, n] = model_copy.params['tauS4']  # s4

#     # Initialize tau_eff with shape (num_variables, N)
#     tau_eff = np.zeros((model_copy.jacobian_dimension//2, N))  # Only the membrane potential variables
#     membrane_indices = range(0, model_copy.jacobian_dimension, 2)  # Only the membrane potential variables

#     # Fit exponential decay for each neuron
#     for idx, i in enumerate(membrane_indices):
#         for n in range(N):
#             y = decay_curves[i, n, :]
#             # Skip the first 10% of points to avoid initial transients
#             skip_points = int(len(y) * 0.05)  # Skip 10% of initial points
#             y = y[skip_points:]
#             t = t_decay[skip_points:]
            
#             dy = np.abs(np.diff(y))
#             # Use relative threshold based on the maximum change
#             max_change = np.max(dy)
#             relative_threshold = 0.01  # 1% of max change
#             changing_idx = dy > (max_change * relative_threshold)
                
#             if np.sum(changing_idx) > 2:    
#                 # Only fit the decaying part
#                 last_changing_idx = np.where(changing_idx)[0][-1]
#                 t_valid = t[:last_changing_idx+1]
#                 y_valid = np.log(np.maximum(y[:last_changing_idx+1], 1e-10))
                
#                 # Fit only the decaying part
#                 A = np.vstack([t_valid, np.ones_like(t_valid)]).T
#                 slope, intercept = np.linalg.lstsq(A, y_valid, rcond=None)[0]
#                 tau_eff[idx, n] = -1/slope
#             else:
#                 # Use analytical tau as fallback
#                 tau_eff[idx, n] = analytical_taus[idx, n]


#     if plot:
#         var_names = ['y1', 'y4', 'u1', 'u4', 'p1', 'p4', 's1', 's4']
#         # Create list of curves for each membrane variable
#         curves = [decay_curves[i, :, :] for i in membrane_indices]
        
#         decay_data = {
#             'time': t_decay,
#             'curves': curves,  # Add the actual curves here
#             'fits': [],  # Optionally, you could add fitted curves here too
#             'tau_values': tau_eff,
#             'var_names': var_names,
#             'analytical_taus': analytical_taus,
#         }
#         return tau_eff, decay_data
#     else:
#         return tau_eff

########################################################################################
# def calculate_effective_tau_yPlus(tauY, a_ss, contrast):
#     # Increase effective tau with contrast
#     contrast_factor = 1 + contrast  # Linear scaling with contrast
#     return tauY * (1 + a_ss) / a_ss * contrast_factor

# def calculate_effective_tau_uPlus(tauU, b_ss, u, sigma, contrast):
#     # Increase effective tau with contrast
#     contrast_factor = Contrast_factor(contrast)  # Linear scaling with contrast
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
            'tauF1', 'tauF1', 'tauF1', 'tauF1',  # Filtered variables for area 1
            'tauF4', 'tauF4', 'tauF4', 'tauF4'   # Filtered variables for area 4
        ]
    else:
        tau_list = [
            'tauY1', 'tauY4', 'tauU1', 'tauU4',
            'tauP1', 'tauP4', 'tauS1', 'tauS4',
            'tauF1', 'tauF1', 'tauF1', 'tauF1',  # Filtered variables for area 1
            'tauF4', 'tauF4', 'tauF4', 'tauF4'   # Filtered variables for area 4
        ]
    
    S_diag = []
    for tau in tau_list:
        # S_diag.extend([np.sqrt(1 / model.params[tau])] * N)
        S_diag.extend([np.power(1 / model.params[tau],0.50)] * N)
    S = np.diag(S_diag)
    return torch.tensor(S, dtype=torch.float32)

def create_S_matrix_filtered(model, freq, tau_f,sigma_f):
    """
    Create S matrix with frequency-dependent filtering
    Args:
        model: RingModel instance
        freq: frequency for filtering (in Hz)
        tau_f: filtering time scale (in seconds)
    Returns:
        S: Torch tensor containing the frequency-dependent noise spectrum
    """
    N = model.params['N']
    
    # Calculate frequency-dependent filtering
    omega = 2 * np.pi * freq  # Convert to angular frequency
    filter_factor = sigma_f * np.sqrt((1/tau_f) * (1 / (1 + (omega * tau_f)**2)))
    
    if model.simulate_firing_rates:
        # Create base S matrix with filtering
        S_vec = np.ones(N) * filter_factor
        # Repeat for all variables in the model
        S = np.concatenate([S_vec] * model.jacobian_dimension)
    else:
        S = np.ones(N * model.jacobian_dimension) * filter_factor
    
    return torch.tensor(np.diag(S), dtype=torch.float32)




