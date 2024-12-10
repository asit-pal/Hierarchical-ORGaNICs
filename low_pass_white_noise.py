import numpy as np
import matplotlib.pyplot as plt

def analytical_power_spectrum(freq, sigma, tau):
    """
    Calculate the analytical power spectrum of a low-pass filtered white noise.
    
    Args:
        freq (array): Frequency values in Hz
        sigma (float): Noise amplitude
        tau (float): Time constant of the filter in seconds
        
    Returns:
        array: Power spectrum values
    """
    w = 2 * np.pi * freq  # Angular frequency
    dummy = (sigma**2) / (1 + (tau * w)**2)
    return dummy/6.245362917844805e-06

def plot_power_spectrum(freq_min=0.1, freq_max=500, n_points=1000, sigma=1.0, tau=0.001):
    """
    Plot the analytical power spectrum.
    
    Args:
        freq_min (float): Minimum frequency in Hz
        freq_max (float): Maximum frequency in Hz
        n_points (int): Number of frequency points
        sigma (float): Noise amplitude
        tau (float): Time constant in seconds
    """
    # Create frequency array
    freq = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
    
    # Calculate power spectrum
    power = analytical_power_spectrum(freq, sigma, tau)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot power spectrum
    ax.loglog(freq, power, 'b-', linewidth=3, label=rf'$\tau={tau*1000:.1f}$ ms')
    
    # Add labels and title
    ax.set_xlabel(r'$\mathrm{Frequency\;(Hz)}$', fontsize=42)
    ax.set_ylabel(r'$\mathrm{Power\;Spectrum}$', fontsize=42)
    ax.set_title(rf'Low-Pass Filtered White Noise ($\sigma={sigma}$)', fontsize=42)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=42)
    
    # Add legend
    ax.legend(fontsize=42, frameon=False)
    
    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    return fig, ax

if __name__ == "__main__":
    # Example usage
    fig, ax = plot_power_spectrum(
        freq_min=0.1,
        freq_max=100,
        n_points=500,
        sigma=0.05,
        tau=5000 * 1e-3 # 1 ms
    )
    
    # Save the plot
    plt.savefig('Plots/low_pass_white_noise_spectrum.pdf', dpi=400, bbox_inches='tight')
    plt.close()
