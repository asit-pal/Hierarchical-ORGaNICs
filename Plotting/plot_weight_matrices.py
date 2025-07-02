import matplotlib.pyplot as plt
import yaml
import sys
import os
import numpy as np

# Add the root directory of the project to sys.path
# This allows to import modules from 'Utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.Create_weight_matrices import setup_parameters

def plot_weight_matrices():
    """
    Generates and plots the weight matrices using the setup_parameters function
    from Utils/Create_weight_matrices.py and saves the plot as a PNG file.
    """
    # Load config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get parameters and weight matrices
    params = setup_parameters(config)

    W11 = params['W11']
    W44 = params['W44']
    W14 = params['W14']
    W41 = params['W41']
    Wzx = params['Wzx']
    
    # Save the params object
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results_4')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(os.path.join(results_dir, 'params.npy'), params)
    print(f"Parameters object saved to {os.path.join(results_dir,'Data' ,'params.npy')}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Weight Matrices Visualization', fontsize=16)

    # Plot W11 (Recurrent in Area 1)
    im1 = axes[0, 0].imshow(W11, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('W11 (Recurrent Area 1)')
    fig.colorbar(im1, ax=axes[0, 0])

    # Plot W44 (Recurrent in Area 4)
    im2 = axes[0, 1].imshow(W44, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('W44 (Recurrent Area 4)')
    fig.colorbar(im2, ax=axes[0, 1])

    # Plot W14 (Feedforward 1 -> 4)
    im3 = axes[0, 2].imshow(W14, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('W14 (Feedforward 1 -> 4)')
    fig.colorbar(im3, ax=axes[0, 2])

    # Plot W41 (Feedback 4 -> 1)
    im4 = axes[1, 0].imshow(W41, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('W41 (Feedback 4 -> 1)')
    fig.colorbar(im4, ax=axes[1, 0])

    # Plot Wzx (Input -> Area 1)
    im5 = axes[1, 1].imshow(Wzx, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Wzx (Input -> Area 1)')
    axes[1, 1].set_xlabel('Stimuli (M)')
    axes[1, 1].set_ylabel('Neurons (N)')
    fig.colorbar(im5, ax=axes[1, 1])

    # Hide the empty subplot
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(os.path.join(results_dir,'Plots', 'weight_matrices.png'))
    
    print(f"Plot saved to {os.path.join(results_dir, 'Plots', 'weight_matrices.png')}")
    
    plt.show()


if __name__ == '__main__':
    plot_weight_matrices() 