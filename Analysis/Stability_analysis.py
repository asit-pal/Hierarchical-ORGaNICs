import yaml
import os
import sys
import autograd.numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
# Initialize model parameters
def main(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config from: {config_file}")
    
    # Load config from yaml file
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
    if not results_dir: # If config_file is just a filename without path
        results_dir = os.getcwd() # Save in current working directory
    data_dir = os.path.join(results_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists
    
    params = setup_parameters(
        config=config, # The full config is still passed for other parameters setup_parameters might need
        tau=1e-3, 
        tauPlus=1e-3,
        N=36
    )

    # Initialize model
    # Ensure 'simulate_firing_rates' is True for Jacobian analysis as intended
    Ring_Model = RingModel(params, simulate_firing_rates=True) 
    # Initial conditions: ensure correct size based on Ring_Model.num_var and params['N']
    initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.1

    # Simulation and search parameters
    method = 'RK45'
    t_span_val = [0, 6] # Increased for stability
    tol_eigenvalue_check =  1e-8
    precision_bsearch =  1e-3

    # Define range for contrast values to loop over
    # contrast_min_loop =  0.01 
    # contrast_max_loop =  1.0 
    # contrast_num_points =  40 
    # contrast_loop_values = np.linspace(contrast_min_loop, contrast_max_loop, num=contrast_num_points)
    contrast_loop_values = np.logspace(np.log10(1e-2), np.log10(1), 40)
    # contrast_loop_values = np.array([ 0.01,0.10,0.20,0.30,0.35,0.40,0.45,0.50])

    # Define search range for gamma (this range will be used by the binary search for each fixed contrast)
    gamma_min_bsearch =  0.1 
    gamma_max_bsearch =  5.0

    bifurcation_points = {}

    for current_c_loop in tqdm(contrast_loop_values, desc="Processing contrast values"):
        print(f"Processing fixed contrast c = {current_c_loop:.2f}")
        
        try:
            critical_gamma = find_critical_gamma(
                ring_model=Ring_Model,
                initial_conditions=initial_conditions,
                fixed_contrast=current_c_loop,
                gamma_min_search=gamma_min_bsearch,
                gamma_max_search=gamma_max_bsearch,
                method=method,
                precision_bsearch=precision_bsearch,
                t_span=t_span_val,
                tol_eigenvalue=tol_eigenvalue_check
            )
            bifurcation_points[current_c_loop] = critical_gamma
            print(f"Bifurcation for c = {current_c_loop:.4f} occurs near gamma1 = {critical_gamma:.4f}")
        except ValueError as e:
            print(f"Error during binary search for c = {current_c_loop:.4f}: {e}")
            bifurcation_points[current_c_loop] = np.nan

    output_npy_file = os.path.join(data_dir, 'contrast_vs_gamma_hopf.npy')
    np.save(output_npy_file, bifurcation_points)
    print(f"Bifurcation points saved to {output_npy_file}")


def has_zero_eigenvalue(J, tol):
    # Calculate eigenvalues of the Jacobian
    eigenvalues = np.linalg.eigvals(J)
    # Check if any eigenvalue has a real part that is positive or close to zero (i.e., > -tol).
    # This can indicate an unstable system (positive real part) or a system near a bifurcation (real part close to zero).
    return np.any(np.real(eigenvalues) > -tol)

def find_critical_gamma(ring_model, initial_conditions, fixed_contrast, gamma_min_search, gamma_max_search, method, precision_bsearch, t_span, tol_eigenvalue):

    max_iterations = 20
    iterations = 0
    
    # Local copies for binary search
    g_min = gamma_min_search
    g_max = gamma_max_search

    while (g_max - g_min > precision_bsearch) and (iterations < max_iterations):
        iterations += 1
        gamma_mid = (g_max + g_min) / 2

        ring_model.params['gamma1'] = gamma_mid # Set the current gamma1 for the model

        try:
            J, _ = ring_model.get_Jacobian_augmented(
                c=fixed_contrast, 
                initial_conditions=initial_conditions,
                method=method, 
                t_span=t_span
            )
            
            if has_zero_eigenvalue(J, tol_eigenvalue):
                # Bifurcation found at or below this gamma_mid. Search in the lower half.
                g_max = gamma_mid  
            else:
                # No bifurcation at this gamma_mid. Search in the upper half.
                g_min = gamma_mid  
        
        except ValueError as e:
            # This typically means the system did not reach a steady state for this (fixed_contrast, gamma_mid) pair.
            # How to adjust g_min/g_max depends on the expected behavior.
            # If instability (non-convergence) is expected at higher gammas, then this gamma_mid might be too high.
            print(f"Info: Problem at gamma1={gamma_mid:.4f} for c={fixed_contrast:.4f} (likely non-convergence or Jacobian issue): {e}. Assuming this gamma is too high, adjusting g_max.")
            g_max = gamma_mid 
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error for gamma1={gamma_mid:.4f}, c={fixed_contrast:.4f}: {e}. Adjusting g_max.")
            g_max = gamma_mid

    return (g_max + g_min) / 2


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Stability_analysis.py path/to/config.yaml")
        print(f"Arguments received: {sys.argv}")
        sys.exit(1)
    
    config_file_path = sys.argv[1]
    main(config_file_path)


