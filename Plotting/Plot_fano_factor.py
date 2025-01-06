import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 36,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
results_dir = 'Results/config4'  # Adjust this path as needed

def plot_fano_factor_results(results_dir):
    data_dir = os.path.join(results_dir, 'Data', 'Fano_factor_data')
    fano_files = [f for f in os.listdir(data_dir) if f.startswith('fano_factor_neuron')]
    
    # Create figures
    fig_fano, ax_fano = plt.subplots(figsize=(12, 8))
    fig_var, ax_var = plt.subplots(figsize=(12, 8))
    
    # Color map for different gamma values
    gamma_colors = plt.cm.viridis
    
    # Load all data and organize by contrast
    all_data = {}
    for file in fano_files:
        data_path = os.path.join(data_dir, file)
        data = np.load(data_path, allow_pickle=True).item()
        contrast = list(set([key[1] for key in data.keys()]))[0]  # Get contrast from data
        all_data[contrast] = data
    
    # Get unique gamma values from first file (should be same for all files)
    gamma_vals = sorted(list(set([key[0] for key in list(all_data.values())[0].keys()])))
    contrasts = sorted(all_data.keys())
    
    # Plot for each gamma value
    for idx, gamma in enumerate(gamma_vals):
        color = gamma_colors(idx / len(gamma_vals))
        
        fano_factors = []
        means = []
        variances = []
        
        for contrast in contrasts:
            data = all_data[contrast][(gamma, contrast)]
            fano_factors.append(data['fano_factor'])
            means.append(data['mean_rate'])
            variances.append(data['variance'])
        
        # Plot Fano factor vs contrast
        ax_fano.plot(contrasts, fano_factors, '-o', color=color, 
                    label=f'$\gamma = {gamma:.2f}$', linewidth=3, markersize=10)
        
        # Plot variance vs mean
        ax_var.plot(means, variances, '-o', color=color, 
                   label=f'$\gamma = {gamma:.2f}$', linewidth=3, markersize=10)
    
     # Add variance = mean line
    # min_mean = min([all_data[c][(g,c)]['mean_rate'] for g in gamma_vals for c in contrasts])
    # max_mean = max([all_data[c][(g,c)]['mean_rate'] for g in gamma_vals for c in contrasts])
    # ax_var.plot([min_mean, max_mean], [min_mean, max_mean], '--', color='gray', 
    #             label=r'$\textnormal{Variance} = \textnormal{Mean}$', linewidth=2, alpha=0.7)
    
    # Customize Fano factor plot
    ax_fano.set_xlabel(r'$\textnormal{Contrast}$')
    ax_fano.set_ylabel(r'$\textnormal{Fano Factor}$')
    ax_fano.set_xscale('log')
    ax_fano.grid(True, which='both', linestyle='--', alpha=0.7)
    ax_fano.legend(fontsize=24)
    
    # Customize variance vs mean plot
    ax_var.set_xlabel(r'$\textnormal{Mean (spikes/s)}$')
    ax_var.set_ylabel(r'$\textnormal{Variance}\ (\textnormal{spikes}/\textnormal{s})^2$')
    ax_var.grid(True, which='both', linestyle='--', alpha=0.7)
    ax_var.legend(fontsize=24)
    
    # Save figures
    fig_fano.savefig(os.path.join(results_dir, 'Plots', 'Fano_factor', 'fano_factor_vs_contrast.pdf'), 
                     dpi=400, bbox_inches='tight')
    fig_var.savefig(os.path.join(results_dir, 'Plots', 'Fano_factor', 'variance_vs_mean.pdf'), 
                    dpi=400, bbox_inches='tight')
    plt.close('all')

def main():
    # results_dir = 'Results/config4'  # Adjust this path as needed
    
    # Create Plots directory if it doesn't exist
    os.makedirs(os.path.join(results_dir, 'Plots', 'Fano_factor'), exist_ok=True)
    
    # Plot Fano factor results
    plot_fano_factor_results(results_dir)

if __name__ == "__main__":
    main()
