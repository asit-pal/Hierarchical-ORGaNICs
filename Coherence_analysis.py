# Essential imports
import yaml
import numpy as np
import os
from Create_weight_matrices import setup_parameters
from Model import RingModel
from Coherence import calculate_coherence

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    
    # Find next available config number
    config_num = 1
    while os.path.exists(os.path.join('Results', f'config{config_num}')):
        config_num += 1
    
    # Create folder with incremented number
    folder_name = f'config{config_num}'
    results_dir = os.path.join('Results', folder_name)
    
    # Create directory structure
    os.makedirs(os.path.join(results_dir, 'Data', 'Coherence_data'), exist_ok=True)

    # Initialize model parameters
    params = setup_parameters(
        config=config,
        tau=1e-3,
        tauPlus=1e-3,
        N=36
    )
    
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Analysis parameters
    N = params['N']
    i = int(N / 2)  # y1 index
    j = int(N / 2) + 2*int(N) if Ring_Model.simulate_firing_rates else int(N / 2) + int(N)  # y4 index

    # Run analyses based on config
    if config['Coherence']['Feedback_gain']['enabled']:
        run_feedback_gain_analysis(Ring_Model, i, j, config, results_dir)
    
    if config['Coherence']['Input_gain_beta1']['enabled']:
        run_input_gain_beta1_analysis(Ring_Model, i, j, config, results_dir)

def run_feedback_gain_analysis(Ring_Model, i, j, config, results_dir):
    fb_config = config['Coherence']['Feedback_gain']
    for contrast in fb_config['c_vals']:
        coherence_data = calculate_coherence(
            Ring_Model,
            i, j,
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            contrast=contrast,
            method='RK45',
            gamma_vals=fb_config['gamma_vals']
        )
        if fb_config['save_data']:
            np.save(os.path.join(results_dir, 'Data', 'Coherence_data', f'coherence_fb_gain_contrast_{contrast}.npy'),
                    coherence_data)

def run_input_gain_beta1_analysis(Ring_Model, i, j, config, results_dir):
    beta1_config = config['Coherence']['Input_gain_beta1']
    for contrast in beta1_config['c_vals']:
        coherence_data = calculate_coherence(
            Ring_Model,
            i, j,
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            contrast=contrast,
            method='RK45',
            beta1_vals=beta1_config['beta1_vals'],
            t_span=beta1_config['t_span']
        )
        if beta1_config['save_data']:
            np.save(os.path.join(results_dir, 'Data', 'Coherence_data', f'coherence_input_gain_beta1_contrast_{contrast}.npy'),
                    coherence_data)

if __name__ == "__main__":
    main()


