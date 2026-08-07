"""Overlay the analytical power spectra on the direct SDE estimates.

Usage:
    python Plotting/Plot_SDE_validation.py <results_dir>

Reads   <results_dir>/Data/sde_validation.npy   (written by Analysis/SDE_validation.py)
Writes  <results_dir>/Plots/sde_validation_*.pdf
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Diagnostic multi-panel figures: the shared journal rcParams are tuned for
# single-panel figures (90 pt labels) and are unreadable here.
DIAG_RC = {
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': False,
    'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
}

SERIES = [
    # key in the result dict,  label,                     colour,    style
    ('sde_linear',        'SDE: linear (independent)',    '#00BFFF', '-'),
    ('sde_linear_paired', 'SDE: linear (noise-matched)',  '#32CD32', '--'),
    ('sde_nonlinear',     'SDE: full nonlinear',          '#DC143C', '-'),
]

PRETTY = {'y1': r'V1 $y_1$ (membrane)', 'y1Plus': r'V1 $y_1^+$ (rate)',
          'y4': r'V4 $y_4$ (membrane)', 'y4Plus': r'V4 $y_4^+$ (rate)'}


def _save(fig, out_path):
    """Write the PDF plus a PNG -- these are diagnostics, meant to be eyeballed."""
    fig.savefig(out_path)
    png_path = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png_path)
    print(f"Saved {out_path} and {os.path.basename(png_path)}")


def plot_condition(result, gamma, contrast, out_path):
    """Two-row figure: PSD overlay on top, sim/analytical ratio underneath."""
    freq = result['freq']
    labels = result['labels']
    n = len(labels)

    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 8.0), sharex=True,
                                 gridspec_kw={'height_ratios': [2.2, 1.0]})
        axes = np.atleast_2d(axes)
        if n == 1:
            axes = axes.reshape(2, 1)

        for k, lab in enumerate(labels):
            top, bot = axes[0, k], axes[1, k]
            ref = result['analytical'][lab]

            top.loglog(freq, ref, color='k', lw=2.5, label='Analytical (linearised)')
            for key, name, colour, style in SERIES:
                if key not in result:
                    continue
                mean = result[key][lab]['mean']
                sem = result[key][lab]['sem']
                top.loglog(freq, mean, color=colour, ls=style, lw=1.6, alpha=0.9, label=name)
                top.fill_between(freq, np.maximum(mean - 2 * sem, 1e-300), mean + 2 * sem,
                                 color=colour, alpha=0.20, lw=0)
                bot.semilogx(freq, mean / ref, color=colour, ls=style, lw=1.6, alpha=0.9)

            top.set_title(PRETTY.get(lab, lab))
            top.grid(True, which='both', ls=':', alpha=0.4)
            if k == 0:
                top.set_ylabel(r'PSD  (units$^2$/Hz)')
                top.legend(loc='lower left')
                bot.set_ylabel('SDE / analytical')
            bot.axhline(1.0, color='k', lw=1.2)
            bot.set_ylim(0.5, 1.5)
            bot.set_xlabel('Frequency (Hz)')
            bot.grid(True, which='both', ls=':', alpha=0.4)

        fig.suptitle(rf'Analytical vs. SDE power spectra   ($c={contrast}$, '
                     rf'$\gamma_1={gamma}$, {result.get("n_trials_note", "")}'
                     rf'shaded band = $\pm 2$ SEM)', fontsize=14)
        _save(fig, out_path)
        plt.close(fig)


def plot_traces(result, gamma, contrast, out_path, seconds=0.2):
    """Nonlinear vs. noise-matched linear trajectories from a single trial."""
    if 'trace_nonlinear' not in result:
        return
    fs = result['fs_rec']
    nl = result['trace_nonlinear']
    lin = result.get('trace_linear')
    labels = result['labels']
    n_show = int(min(nl.shape[-1], seconds * fs))
    t = np.arange(n_show) / fs

    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(len(labels), 1, figsize=(11, 2.4 * len(labels)), sharex=True)
        axes = np.atleast_1d(axes)
        for k, lab in enumerate(labels):
            ax = axes[k]
            ax.plot(t, nl[k, :n_show], color='#DC143C', lw=1.2, label='full nonlinear')
            if lin is not None:
                ax.plot(t, lin[k, :n_show], color='#32CD32', lw=1.2, ls='--',
                        label='linearised, identical noise')
            ax.set_ylabel(PRETTY.get(lab, lab), fontsize=11)
            ax.grid(True, ls=':', alpha=0.4)
            if k == 0:
                ax.legend(loc='upper right', ncol=2)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(rf'Deviation from the fixed point, one shared noise realisation '
                     rf'($c={contrast}$, $\gamma_1={gamma}$)', fontsize=14)
        _save(fig, out_path)
        plt.close(fig)


def main(results_dir):
    data_path = os.path.join(results_dir, 'Data', 'sde_validation.npy')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run Analysis/SDE_validation.py first.")
        sys.exit(1)
    payload = np.load(data_path, allow_pickle=True).item()

    plots_dir = os.path.join(results_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    settings = payload['settings']
    note = (f"{settings['n_trials']} trials x {settings['T']}s, ")
    for (gamma, contrast), result in payload['results'].items():
        result['n_trials_note'] = note
        tag = f"c{contrast}_g{gamma}".replace('.', 'p')
        plot_condition(result, gamma, contrast,
                       os.path.join(plots_dir, f'sde_validation_psd_{tag}.pdf'))
        plot_traces(result, gamma, contrast,
                    os.path.join(plots_dir, f'sde_validation_traces_{tag}.pdf'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plot the SDE validation of the analytical PSD')
    p.add_argument('results_dir', help='Results directory containing Data/sde_validation.npy')
    args = p.parse_args()
    main(args.results_dir)
