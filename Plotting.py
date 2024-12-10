import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


def plot_steady_states(steady_states, c_vals, gamma_vals, fb_gain, input_gain_beta1, input_gain_beta4, index_y1, index_y4, figsize=(20, 12), line_width=5, labelsize=44, ticksize=44, legendsize=44):
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False, sharex=False)

    for gamma in gamma_vals:
        steady_states_gamma = [steady_states[(gamma, c)] for c in c_vals]
        
        if fb_gain:
            label = rf'$\gamma_1={gamma}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma}$'
        

        # Plot V1 firing rate (y1+)
        axs[0].plot(c_vals*100, [state[index_y1]**2 for state in steady_states_gamma], '-', lw=line_width, label=label)
        # Plot V4 firing rate (y4+)
        axs[1].plot(c_vals*100, [state[index_y4]**2 for state in steady_states_gamma], '-', lw=line_width)

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    axs[0].set_xlabel(r'$\textnormal{Contrast (\%)}$', fontsize=labelsize)
    axs[1].set_xlabel(r'$\textnormal{Contrast (\%)}$', fontsize=labelsize)
    axs[0].set_ylabel(r'$\textnormal{Relative firing rate}$', fontsize=labelsize)
    axs[1].set_ylabel(r'$\textnormal{Relative firing rate}$', fontsize=labelsize)

    # Modify the y-axis ticks density
    axs[0].yaxis.set_major_locator(ticker.MaxNLocator(6))
    axs[1].yaxis.set_major_locator(ticker.MaxNLocator(6))

    # Increase the fontsize of ticks
    axs[0].tick_params(axis='both', which='major', labelsize=ticksize)
    axs[1].tick_params(axis='both', which='major', labelsize=ticksize)

    axs[0].legend(frameon=False, labelspacing=0.2, loc='upper left', handletextpad=0.3, fontsize=legendsize)
    axs[0].set_title('V1', fontsize=labelsize)
    axs[1].set_title('V4', fontsize=labelsize)
    
    if fb_gain:
        xticks = [ 1, 10, 100]
        xticklabels = [r'$\textnormal{1}$', r'$\textnormal{10}$', r'$\textnormal{100}$']
    elif input_gain_beta1:
        xticks = [ 0.1,1,10,100]
        xticklabels = [r'$\textnormal{0.1}$', r'$\textnormal{1}$', r'$\textnormal{10}$', r'$\textnormal{100}$']
    for ax in axs:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    
    try:
        fig.tight_layout()
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    

    return fig, axs



def plot_coherence_data(coherence_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, line_width=5, line_labelsize=42, legendsize=42):
    # Set the backend to 'Agg' for headless environments
    
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a truncated viridis colormap that does not include the last yellow part
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    
    norm = mcolors.Normalize(vmin=min(gamma_vals), vmax=max(gamma_vals))

    for gamma in gamma_vals:
        key = (gamma, contrast)  # Changed to match power_spectra format
        data = coherence_data[key]
        freq = data['freq']
        coh = data['coh']
        
        # Get color from truncated colormap
        color = truncated_viridis(norm(gamma))
        
        # Determine label based on which gain is being varied
        if fb_gain:
            label = rf'$\gamma_1={gamma}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma}$'
        
        # Use solid line style for all cases
        linestyle = '-'
        
        ax.semilogx(freq, coh, lw=line_width, linestyle=linestyle, label=label, color=color)
    
    # Fix the LaTeX formatting in labels
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{V1\text{-}V2\;Coherence}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    ax.legend(fontsize=legendsize, loc='best', frameon=False, handletextpad=0.2, handlelength=1.0, labelspacing=0.2)

    # Use manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

def plot_power_spectra_fixed_gamma(power_data, contrast_vals, gamma, line_width=5, line_labelsize=42, legendsize=42):
    """
    Plot power spectra for a fixed gamma value across different contrasts.
    Expects pre-normalized power data (normalized by max power at contrast=1).
    
    Args:
        power_data (dict): Dictionary containing normalized power spectra data
        contrast_vals (list): List of contrast values to plot
        gamma (float): The gamma value to plot
        line_width (int): Width of plotted lines
        line_labelsize (int): Size of axis labels
        legendsize (int): Size of legend text
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a truncated viridis colormap that does not include the last yellow part
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    
    norm = mcolors.Normalize(vmin=min(contrast_vals), vmax=max(contrast_vals))
    
    for contrast in contrast_vals:
        key = (gamma, contrast)
        if key not in power_data:
            print(f"Warning: No power data found for contrast = {contrast} and gamma = {gamma}")
            continue
            
        data = power_data[key]
        freq = data['freq']
        power = data['power']  # Data is already normalized
        
        # Get color from truncated colormap
        color = truncated_viridis(norm(contrast))
        
        # Label showing contrast value
        label = rf'$c={contrast}$'
        
        ax.plot(freq, power, lw=line_width, linestyle='-', label=label, color=color)
    
    # Fix the LaTeX formatting in labels
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{Normalized\;V1\;Power}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    ax.legend(fontsize=legendsize, loc='best', frameon=False, handletextpad=0.2, handlelength=1.0, labelspacing=0.2)
    # ax.set_xlim(0,100)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # Add title showing gamma value
    ax.set_title(rf'$\gamma_1={gamma}$', fontsize=line_labelsize)

    # Use manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax



def plot_power_spectra(power_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, line_width=5, line_labelsize=42, legendsize=42):
    # Set the backend to 'Agg' for headless environments
    
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a truncated viridis colormap that does not include the last yellow part
    viridis = plt.get_cmap('viridis')
    truncated_viridis = mcolors.ListedColormap(viridis(np.linspace(0, 0.8, 256)))
    
    norm = mcolors.Normalize(vmin=min(gamma_vals), vmax=max(gamma_vals))
    
    for gamma in gamma_vals:
        key = (gamma,contrast) 
        data = power_data[key]
        freq = data['freq']
        power = data['power']
        
        # Get color from truncated colormap
        color = truncated_viridis(norm(gamma))
        
        # Determine label based on which gain is being varied
        if fb_gain:
            label = rf'$\gamma_1={gamma}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma}$'
        
        # Use solid line style for all cases
        linestyle = '-'
        
        ax.loglog(freq, power, lw=line_width, linestyle=linestyle, label=label, color=color)
    
    # Fix the LaTeX formatting in labels
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=line_labelsize)
    ax.set_ylabel(r'$\mathrm{V1\;Power}$', fontsize=line_labelsize)
    ax.tick_params(axis='both', which='major', labelsize=line_labelsize)
    ax.legend(fontsize=legendsize, loc='best', frameon=False, handletextpad=0.2, handlelength=1.0, labelspacing=0.2)

    # Use manual adjustment instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax




# def plot_performance_vs_freq(performance_data, gamma_vals,contrast, fb_gain, input_gain_beta1, input_gain_beta4, frequencies, labelsize=28, legendsize=28):
#     colors = ['#00BFFF', '#DC143C', '#50C878']

#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     custom_handles = []

#     for gamma, color in zip(gamma_vals, colors):
#         data = performance_data[gamma,contrast]

#         # Retrieve V1 and V4 data
#         V1_mean = np.abs(data['V1']['mean'])
#         V1_std = np.abs(data['V1']['std'])
#         V4_mean = np.abs(data['V4']['mean'])
#         V4_std = np.abs(data['V4']['std'])

#         # Fill between for V1 data
#         ax.fill_between(frequencies, V1_mean - V1_std, V1_mean + V1_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V1_mean, marker='x', linestyle='-', markersize=12, markeredgewidth=3, 
#                 markeredgecolor='black', markerfacecolor='white', color=color)

#         # Fill between for V4 data
#         ax.fill_between(frequencies, V4_mean - V4_std, V4_mean + V4_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V4_mean, marker='o', markeredgewidth=3, markeredgecolor='red', 
#                 markerfacecolor='white', linestyle='-', markersize=12, color=color)

#         if fb_gain:
#             label = rf'$\gamma_1={gamma:.2f}$'
#         elif input_gain_beta1:
#             label = rf'$\beta_1={gamma:.2f}$'
#         elif input_gain_beta4:
#             label = rf'$\beta_4={gamma:.2f}$'

#         custom_handles.append(mpatches.Patch(color=color, label=label))

#     # First legend for gamma/beta values
#     legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', 
#                         frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
#     ax.add_artist(legend1)  # Add first legend back to the plot

#     marker_legend = [
#         mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V4}$', markeredgewidth=3, markerfacecolor='white'),
#         mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V1}$', markeredgewidth=3, markerfacecolor='white'),
#     ]
#     legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.48, 1.0), 
#                         fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
#     ax.add_artist(legend2)  # Add second legend to the plot

#     ax.set_xlabel(r'$\textnormal{Frequency (Hz)}$', fontsize=labelsize, fontweight='bold')
#     ax.set_ylabel(r'$\textnormal{Prediction performance}$', fontsize=labelsize, fontweight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=labelsize)
#     ax.set_xscale('log')  # Set x-axis to logarithmic scale
#     ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
#     return fig, ax


# def plot_dimension_vs_freq(dimension_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, frequencies, labelsize=28, legendsize=28):
#     colors = ['#00BFFF', '#DC143C', '#50C878']

#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     custom_handles = []

#     for gamma, color in zip(gamma_vals, colors):
#         data = dimension_data[gamma, contrast]

#         # Retrieve V1 and V4 data
#         V1_mean = data['V1']['mean']
#         V1_std = data['V1']['std']
#         V4_mean = data['V4']['mean']
#         V4_std = data['V4']['std']

#         # Fill between for V1 data
#         ax.fill_between(frequencies, V1_mean - V1_std, V1_mean + V1_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V1_mean, marker='x', linestyle='-', markersize=12, markeredgewidth=3, 
#                 markeredgecolor='black', markerfacecolor='white', color=color)

#         # Fill between for V4 data
#         ax.fill_between(frequencies, V4_mean - V4_std, V4_mean + V4_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V4_mean, marker='o', markeredgewidth=3, markeredgecolor='red', 
#                 markerfacecolor='white', linestyle='-', markersize=12, color=color)

#         if fb_gain:
#             label = rf'$\gamma_1={gamma:.2f}$'
#         elif input_gain_beta1:
#             label = rf'$\beta_1={gamma:.2f}$'
#         elif input_gain_beta4:
#             label = rf'$\beta_4={gamma:.2f}$'

#         custom_handles.append(mpatches.Patch(color=color, label=label))

#     # First legend for gamma/beta values
#     legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', 
#                         frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
#     ax.add_artist(legend1)  # Add first legend back to the plot

#     marker_legend = [
#         mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V4}$', markeredgewidth=3, markerfacecolor='white'),
#         mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V1}$', markeredgewidth=3, markerfacecolor='white'),
#     ]
#     legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.48, 1.0), 
#                         fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
#     ax.add_artist(legend2)  # Add second legend to the plot

#     ax.set_xlabel(r'$\textnormal{Frequency (Hz)}$', fontsize=labelsize, fontweight='bold')
#     ax.set_ylabel(r'$\textnormal{Dimension}$', fontsize=labelsize, fontweight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=labelsize)
#     ax.set_xscale('log')  # Set x-axis to logarithmic scale
#     ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
#     return fig, ax

# def plot_dimension_vs_freq(dimension_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, frequencies, labelsize=28, legendsize=28):
#     colors = ['#00BFFF', '#DC143C', '#50C878']

#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     custom_handles = []

#     for gamma, color in zip(gamma_vals, colors):
#         data = dimension_data[gamma, contrast]

#         # Retrieve V1 and V4 data
#         V1_mean = data['V1']['mean']
#         V1_std = data['V1']['std']
#         V4_mean = data['V4']['mean']
#         V4_std = data['V4']['std']

#         # Fill between for V1 data
#         ax.fill_between(frequencies, V1_mean - V1_std, V1_mean + V1_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V1_mean, marker='x', linestyle='-', markersize=12, markeredgewidth=3, 
#                 markeredgecolor='black', markerfacecolor='white', color=color)

#         # Fill between for V4 data
#         ax.fill_between(frequencies, V4_mean - V4_std, V4_mean + V4_std, color=color, alpha=0.2)
#         ax.plot(frequencies, V4_mean, marker='o', markeredgewidth=3, markeredgecolor='red', 
#                 markerfacecolor='white', linestyle='-', markersize=12, color=color)

#         if fb_gain:
#             label = rf'$\gamma_1={gamma:.2f}$'
#         elif input_gain_beta1:
#             label = rf'$\beta_1={gamma:.2f}$'
#         elif input_gain_beta4:
#             label = rf'$\beta_4={gamma:.2f}$'

#         custom_handles.append(mpatches.Patch(color=color, label=label))

#     # First legend for gamma/beta values
#     legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', 
#                         frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
#     ax.add_artist(legend1)  # Add first legend back to the plot

#     marker_legend = [
#         mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V4}$', markeredgewidth=3, markerfacecolor='white'),
#         mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=10, 
#                       label=r'$\textnormal{V1-V1}$', markeredgewidth=3, markerfacecolor='white'),
#     ]
#     legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.48, 1.0), 
#                         fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
#     ax.add_artist(legend2)  # Add second legend to the plot

#     ax.set_xlabel(r'$\textnormal{Frequency (Hz)}$', fontsize=labelsize, fontweight='bold')
#     ax.set_ylabel(r'$\textnormal{Dimension}$', fontsize=labelsize, fontweight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=labelsize)
#     ax.set_xscale('log')  # Set x-axis to logarithmic scale
#     ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
#     return fig, ax

# Three area model
def plot_pred_perf_vs_dim_3(performance_data, pairs, contrast, labelsize=42, legendsize=42):
    """
    Plot prediction performance vs dimensions for different parameter pairs.
    
    Args:
        performance_data: Dictionary containing performance metrics
        pairs: List of (g, gamma4, gamma5) tuples
        contrast: Contrast value
        labelsize: Size of axis labels
        legendsize: Size of legend text
    """
    colors = ['#DC143C', '#00BFFF', '#32CD32']
    markers = ['o', 's', 'D']
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, pair in enumerate(pairs):
        g, gamma4, gamma5 = pair
        data = performance_data[g, gamma4, gamma5, contrast]
        
        # Plot V4 data
        ax.fill_between(data['V4']['dims'], 
                       data['V4']['mean'] - data['V4']['std'], 
                       data['V4']['mean'] + data['V4']['std'], 
                       color='gray', alpha=0.1)
        ax.plot(data['V4']['dims'], data['V4']['mean'], 
                marker=markers[i], markeredgecolor='black',
                linestyle='None', markersize=18, color=colors[1])
        
        # Plot V5 data
        ax.fill_between(data['V5']['dims'], 
                       data['V5']['mean'] - data['V5']['std'], 
                       data['V5']['mean'] + data['V5']['std'], 
                       color='gray', alpha=0.1)
        ax.plot(data['V5']['dims'], data['V5']['mean'], 
                marker=markers[i], markeredgecolor='black',
                linestyle='None', markersize=18, color=colors[0])
    
    # Add legends and labels
    marker_legend = [
        mlines.Line2D([0], [0], color=colors[0], marker='o', linestyle='None', 
                     markersize=15, label=r'$\textnormal{V1-V4}$', markeredgecolor='black'),
        mlines.Line2D([0], [0], color=colors[1], marker='o', linestyle='None', 
                     markersize=15, label=r'$\textnormal{V1-MT}$', markeredgecolor='black'),
    ]
    
    ax.legend(handles=marker_legend, loc='upper left', bbox_to_anchor=(-0.05, 1.0),
             fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.2)
    
    plt.xlabel(r'$\textnormal{Dimensions}$', fontsize=labelsize, fontweight='bold')
    plt.ylabel(r'$\textnormal{Prediction performance}$', fontsize=labelsize, fontweight='bold')
    
    # Set x-axis ticks
    x_min, x_max = ax.get_xlim()
    x_ticks = range(int(x_min), int(x_max) + 2, 2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([rf'$\textnormal{x}$' for x in x_ticks])
    
    try:
        fig.tight_layout()
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, ax

def plot_dimension_vs_freq(dimension_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, frequencies, labelsize=28, legendsize=28):
    colors = ['#00BFFF', '#DC143C', '#50C878']

    fig, ax = plt.subplots(figsize=(12, 10))
    
    custom_handles = []

    for gamma, color in zip(gamma_vals, colors):
        data = dimension_data[gamma, contrast]

        # Retrieve V1 and V4 data
        V1_mean = data['V1']['mean']
        V1_std = data['V1']['std']
        V4_mean = data['V4']['mean']
        V4_std = data['V4']['std']

        # # Fill between for V1 data
        # ax.fill_between(frequencies, V1_mean - V1_std, V1_mean + V1_std, color=color, alpha=0.2)
        # ax.plot(frequencies, V1_mean, marker='x', linestyle='-', markersize=12, markeredgewidth=3, 
        #         markeredgecolor='black', markerfacecolor='white', color=color)

        # Fill between for V4 data
        ax.fill_between(frequencies, V4_mean - V4_std, V4_mean + V4_std, color=color, alpha=0.2)
        ax.plot(frequencies, V4_mean, marker='o', markeredgewidth=3, markeredgecolor='red', 
                markerfacecolor='white', linestyle='-', markersize=12, color=color)

        if fb_gain:
            label = rf'$\gamma_1={gamma:.2f}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma:.2f}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma:.2f}$'

        custom_handles.append(mpatches.Patch(color=color, label=label))

    # First legend for gamma/beta values
    legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', 
                        frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
    ax.add_artist(legend1)  # Add first legend back to the plot

    marker_legend = [
        mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, 
                      label=r'$\textnormal{V1-V4}$', markeredgewidth=3, markerfacecolor='white'),
        # mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=10, 
        #               label=r'$\textnormal{V1-V1}$', markeredgewidth=3, markerfacecolor='white'),
    ]
    legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.48, 1.0), 
                        fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
    ax.add_artist(legend2)  # Add second legend to the plot

    ax.set_xlabel(r'$\textnormal{Frequency (Hz)}$', fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(r'$\textnormal{Dimension}$', fontsize=labelsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
    try:
        fig.tight_layout()
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, ax

def plot_performance_vs_freq(performance_data, gamma_vals,contrast, fb_gain, input_gain_beta1, input_gain_beta4, frequencies, labelsize=28, legendsize=28):
    colors = ['#00BFFF', '#DC143C', '#50C878']

    fig, ax = plt.subplots(figsize=(12, 10))
    
    custom_handles = []

    for gamma, color in zip(gamma_vals, colors):
        data = performance_data[gamma,contrast]

        # Retrieve V1 and V4 data
        V1_mean = np.abs(data['V1']['mean'])
        V1_std = np.abs(data['V1']['std'])
        V4_mean = np.abs(data['V4']['mean'])
        V4_std = np.abs(data['V4']['std'])

        # # Fill between for V1 data
        # ax.fill_between(frequencies, V1_mean - V1_std, V1_mean + V1_std, color=color, alpha=0.2)
        # ax.plot(frequencies, V1_mean, marker='x', linestyle='-', markersize=12, markeredgewidth=3, 
        #         markeredgecolor='black', markerfacecolor='white', color=color)

        # Fill between for V4 data
        ax.fill_between(frequencies, V4_mean - V4_std, V4_mean + V4_std, color=color, alpha=0.2)
        ax.plot(frequencies, V4_mean, marker='o', markeredgewidth=3, markeredgecolor='red', 
                markerfacecolor='white', linestyle='-', markersize=12, color=color)

        if fb_gain:
            label = rf'$\gamma_1={gamma:.2f}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma:.2f}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma:.2f}$'

        custom_handles.append(mpatches.Patch(color=color, label=label))

    # First legend for gamma/beta values
    legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', 
                        frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
    ax.add_artist(legend1)  # Add first legend back to the plot

    marker_legend = [
        mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, 
                      label=r'$\textnormal{V1-V4}$', markeredgewidth=3, markerfacecolor='white'),
        # mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=10, 
        #               label=r'$\textnormal{V1-V1}$', markeredgewidth=3, markerfacecolor='white'),
    ]
    legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.48, 1.0), 
                        fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
    ax.add_artist(legend2)  # Add second legend to the plot

    ax.set_xlabel(r'$\textnormal{Frequency (Hz)}$', fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(r'$\textnormal{Prediction performance}$', fontsize=labelsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
    try:
        fig.tight_layout()
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, ax

def plot_pred_perf_vs_dim(performance_data, gamma_vals, contrast, fb_gain, input_gain_beta1, input_gain_beta4, labelsize=42, legendsize=42):
    colors = ['#DC143C', '#00BFFF','#32CD32','#000000']  # Red for gamma_vals[0], Blue for gamma_vals[1], Green for gamma_vals[2],
    fig, ax = plt.subplots(figsize=(12, 10))
    
    custom_handles = []

    for i, (gamma, color) in enumerate(zip(gamma_vals, colors)):
        data = performance_data[gamma, contrast]

        # Retrieve V1 and V4 data
        V1_mean = data['V1']['mean']
        V1_std = data['V1']['std']
        V1_dims = data['V1']['dims']
        V4_mean = data['V4']['mean']
        V4_std = data['V4']['std']
        V4_dims = data['V4']['dims']

        # # Fill between for V1 data
        # ax.fill_between(V1_dims, V1_mean - V1_std, V1_mean + V1_std, color='gray', alpha=0.1)
        # ax.plot(V1_dims, V1_mean, marker='s', linestyle='None', markersize=12,  color=color, markeredgecolor='black')

        # Fill between for V4 data
        ax.fill_between(V4_dims, V4_mean - V4_std, V4_mean + V4_std, color='gray', alpha=0.1)
        ax.plot(V4_dims, V4_mean, marker='o', markersize=12,   linestyle='None', color=color,markeredgecolor='black')

        if fb_gain:
            label = rf'$\gamma_1={gamma:.2f}$'
        elif input_gain_beta1:
            label = rf'$\beta_1={gamma:.2f}$'
        elif input_gain_beta4:
            label = rf'$\beta_4={gamma:.2f}$'
        
        custom_handles.append(mpatches.Patch(color=color, label=label))

    # First legend for gamma/beta values
    legend1 = ax.legend(handles=custom_handles, fontsize=legendsize, loc='upper left', frameon=False, handletextpad=0.1, labelspacing=0.15, handlelength=1.0)
    ax.add_artist(legend1)  # Add first legend back to the plot

    marker_legend = [
        mlines.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=10, label=r'$\textnormal{V1-V2}$' ),
        mlines.Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=10, label=r'$\textnormal{V1-V1}$'),
    ]
    legend2 = ax.legend(handles=marker_legend, loc='upper right', bbox_to_anchor=(0.68, 1.0), fontsize=legendsize, frameon=False, handletextpad=-0.2, labelspacing=0.15)
    ax.add_artist(legend2)  # Add second legend to the plot

    ax.set_xlabel(r'$\textnormal{Dimensions}$', fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(r'$\textnormal{Prediction performance}$', fontsize=labelsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax.set_title(rf'$\textnormal{{Contrast}}={contrast}$', fontsize=labelsize, fontweight='bold')
    # Set x-axis ticks to integers with interval of 2
    x_min, x_max = ax.get_xlim()
    x_ticks = range(int(x_min), int(x_max) + 2, 2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([rf'$\textnormal{x}$' for x in x_ticks])

    try:
        fig.tight_layout()
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    return fig, ax

def plot_decay_trajectories(decay_data, center_idx=None, plot_analytical=True, labelsize=18, legendsize=14):
    """
    Plot decay trajectories and their exponential fits for membrane potentials.
    Only plots for the center neuron (N//2).
    
    Parameters:
    -----------
    decay_data : dict
        Contains:
        - time: array of time points
        - curves: list of arrays containing decay curves
        - tau_values: array of fitted time constants
        - var_names: list of variable names
        - analytical_taus: array of analytical time constants
    center_idx : int, optional
        Index of the neuron to plot. If None, uses N//2
    plot_analytical : bool, optional
        Whether to plot the analytical fit curve (default: True)
    """
    # Get data
    var_names = decay_data['var_names']
    time = decay_data['time']
    if center_idx is None:
        center_idx = decay_data['tau_values'].shape[1] // 2
    
    # Create subplots for each variable type
    num_vars = len(var_names)
    fig, axs = plt.subplots(num_vars, 1, figsize=(15, 4*num_vars))
    if num_vars == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        # Get tau values for this variable type and center neuron
        tau_numerical = decay_data['tau_values'][i, center_idx]
        
        # Plot original decay curve - accessing as list
        y = decay_data['curves'][i][center_idx][:]
        ax.plot(time, y, 'b-', label='Data', linewidth=2)
        
        # Plot numerical fit if valid
        if tau_numerical > 0:
            y_fit = y[0] * np.exp(-time/tau_numerical)
            ax.plot(time, y_fit, 'r--', 
                   label=f'Numerical Fit (τ = {tau_numerical:.3f})', 
                   linewidth=2)
        
        # Plot analytical fit if requested
        if plot_analytical:
            tau_analytical = decay_data['analytical_taus'][i, center_idx]
            if tau_analytical > 0:
                y_analytical = y[0] * np.exp(-time/tau_analytical)
                ax.plot(time, y_analytical, 'g:', 
                       label=f'Analytical (τ = {tau_analytical:.3f})',
                       linewidth=2)
        
        ax.set_title(f'{var_names[i]} Decay (Center Neuron)', fontsize=labelsize)
        ax.set_xlabel('Time (s)', fontsize=labelsize)
        ax.set_ylabel('Activity', fontsize=labelsize)
        ax.set_yscale('log')
        ax.legend(fontsize=legendsize)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=labelsize-4)
    
    plt.tight_layout()
    return fig, axs