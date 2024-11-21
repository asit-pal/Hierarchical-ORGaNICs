# Essential imports
import yaml
import numpy as np
from Create_weight_matrices import setup_parameters
from Model import RingModel
from Communication import Calculate_Pred_perf_Dim
import os
# import sys
config_path = 'configs/config3.yaml'  # specify config path
  
def load_config(config_path=config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    
    # if len(sys.argv) > 1:
    #     config_path = sys.argv[1]
    # folder_name = os.path.basename(os.path.dirname(config_path))
    folder_name = os.path.splitext(os.path.basename(config_path))[0]
    # # Find next available config number
    # config_num = 1
    # while os.path.exists(os.path.join('Results', f'config{config_num}')):
    #     config_num += 1
    
    # Create folder with incremented number
    # folder_name = f'config{config_num}'
    results_dir = os.path.join('Results', folder_name)
    
    # Create directory structure
    os.makedirs(os.path.join(results_dir, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Data', 'Communication_data'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Plots'), exist_ok=True)

    # Save config
    config_save_path = os.path.join(results_dir, 'configs', 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize model parameters - using default tau and N since they're not in config
    params = setup_parameters(
        config=config,
        tau=1e-3,  # default value
        tauPlus=1e-3,
        N=36       # default value
    )
    # Initialize model
    Ring_Model = RingModel(params, simulate_firing_rates=True)
    
    # Initialize initial conditions
    # initial_conditions = np.ones((Ring_Model.num_var * params['N'])) * 0.01

    # Set up indices based on simulation type
    N = params['N']
    if Ring_Model.simulate_firing_rates:
        N1_y = np.arange(N, 2*N)    # index for y1Plus
        N4_y = np.arange(3*N, 4*N)  # index for y4Plus   
        bw_y1_y4 = False
    else:
        N1_y = np.arange(0*N, N)    # index for y1
        N4_y = np.arange(1*N, 2*N)  # index for y4
        bw_y1_y4 = True
    # Analysis parameters using exact config structure
    # Communication subspace parameters
    com_params = {'num_trials':250,
                'tol':5e-2,
                'V1_s':18,
                'V1_t':18,
                'V4_t':18,
                'N1_y_idx':N1_y,
                'N4_y_idx':N4_y,
                'bw_y1_y4':bw_y1_y4}

    # Run analyses based on config
    if config['Communication']['Feedback_gain']['enabled']:
        run_feedback_gain_analysis(Ring_Model, config, com_params, results_dir)
    
    if config['Communication']['Input_gain_beta1']['enabled']:
        run_input_gain_beta1_analysis(Ring_Model, config, com_params, results_dir)

def run_feedback_gain_analysis(Ring_Model, config, com_params, results_dir):
    fb_config = config['Communication']['Feedback_gain']
    for contrast in fb_config['c_vals']:
        Communication_data = Calculate_Pred_perf_Dim(
            Ring_Model,
            fb_config['gamma_vals'],
            contrast,
            g=config['model_params']['g1'],
            fb_gain=True,
            input_gain_beta1=False,
            input_gain_beta4=False,
            method='RK45',
            com_params=com_params,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            t_span=fb_config['t_span']
            )
        if fb_config['save_data']:
            np.save(os.path.join(results_dir, 'Data', 'Communication_data', f'Communication_fb_gain_contrast_{contrast}.npy'),
                    Communication_data)

def run_input_gain_beta1_analysis(Ring_Model, config, com_params, results_dir):
    beta1_config = config['Communication']['Input_gain_beta1']
    for contrast in beta1_config['c_vals']:
        Communication_data = Calculate_Pred_perf_Dim(
            Ring_Model,
            beta1_config['beta1_vals'],  # Use default gamma from model params
            contrast,
            g=config['model_params']['g1'],
            fb_gain=False,
            input_gain_beta1=True,
            input_gain_beta4=False,
            method='RK45',
            com_params=com_params,
            delta_tau=config['noise_params']['delta_tau'] * Ring_Model.params['tau'],
            noise=config['noise_params']['noise'],
            t_span=beta1_config['t_span']
        )
        if beta1_config['save_data']:
            np.save(os.path.join(results_dir, 'Data', 'Communication_data', f'Communication_input_gain_beta1_contrast_{contrast}.npy'),
                    Communication_data)



if __name__ == "__main__":
    main()



