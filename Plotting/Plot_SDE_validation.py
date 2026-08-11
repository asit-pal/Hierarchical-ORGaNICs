"""Overlay the analytical LFP spectra and coherence on the direct SDE estimates.

Usage:
    python Plotting/Plot_SDE_validation.py <results_dir> [--show-linear]

Reads   <results_dir>/Data/sde_validation.npy   (written by Analysis/SDE_validation.py)
Writes  <results_dir>/Plots/sde_validation_*.{pdf,png}

By default each panel carries exactly two curves -- the analytical (linearised)
result and the full nonlinear SDE estimate. `--show-linear` adds the two
linear-SDE control series, which are always computed and always appear in the
printed agreement table; they are diagnostics for the machinery, not the
comparison of interest.
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

# Diagnostic figures: the shared journal rcParams are tuned for single-panel
# figures (90 pt labels) and are unreadable here.
DIAG_RC = {
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
}

ANALYTICAL_KW = dict(color='k', lw=2.6, label='Analytical (linearised)')
NONLINEAR_KW = dict(color='#DC143C', lw=1.7, label='SDE (full nonlinear)')

# Extra control series, only drawn with --show-linear.
LINEAR_SERIES = [
    ('sde_linear',        'SDE: linear (independent)',   '#00BFFF', '-'),
    ('sde_linear_paired', 'SDE: linear (noise-matched)', '#32CD32', '--'),
]

# The published power spectra and coherence are LFP quantities, built from the raw
# membrane potentials -- not the firing rates. `Power_spectra_analysis.py` uses
# i = N//2 (y1) and 2N + N//2 (y4); `Coherence_analysis.py` uses that same pair.
LFP_LABELS = ('y1', 'y4')

PRETTY = {'y1': r'V1 LFP  ($y_1$)', 'y1Plus': r'V1 rate  ($y_1^+$)',
          'y4': r'V4 LFP  ($y_4$)', 'y4Plus': r'V4 rate  ($y_4^+$)'}


def _save(fig, out_path):
    """Write the PDF plus a PNG -- these are diagnostics, meant to be eyeballed."""
    fig.savefig(out_path)
    png_path = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png_path)
    print(f"Saved {out_path} and {os.path.basename(png_path)}")


def _present_labels(result, wanted=LFP_LABELS):
    return [lab for lab in wanted if lab in result['labels']]


def plot_psd(result, gamma, contrast, out_path, show_linear=False):
    """Analytical vs. SDE power spectra for the LFP channels."""
    freq = result['freq']
    labels = _present_labels(result)
    if not labels:
        print("No LFP labels in the payload; skipping the PSD figure.")
        return

    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(1, len(labels), figsize=(6.0 * len(labels), 5.0),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, lab in zip(axes, labels):
            ax.loglog(freq, result['analytical'][lab], **ANALYTICAL_KW)
            ax.loglog(freq, result['sde_nonlinear'][lab]['mean'], **NONLINEAR_KW)
            if show_linear:
                for key, name, colour, style in LINEAR_SERIES:
                    if key in result:
                        ax.loglog(freq, result[key][lab]['mean'], color=colour,
                                  ls=style, lw=1.3, alpha=0.9, label=name)
            ax.set_title(PRETTY.get(lab, lab))
            ax.set_xlabel('Frequency (Hz)')
            ax.grid(True, which='both', ls=':', alpha=0.4)
        axes[0].set_ylabel(r'Power spectral density  (units$^2$/Hz)')
        axes[0].legend(loc='lower left')
        fig.suptitle(_suptitle('LFP power spectra', result, gamma, contrast), fontsize=15)
        _save(fig, out_path)
        plt.close(fig)


def plot_coherence(result, gamma, contrast, out_path, show_linear=False, f_max=100.0):
    """Analytical vs. SDE magnitude-squared coherence for each recorded pair."""
    pairs = [tuple(p) for p in result.get('coherence_pairs', [])]
    pairs = [p for p in pairs if p in result.get('analytical_coherence', {})]
    if not pairs:
        print("No coherence pairs in the payload; skipping the coherence figure.")
        return

    freq = result['freq']
    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(1, len(pairs), figsize=(7.5 * len(pairs), 5.2))
        axes = np.atleast_1d(axes)
        for ax, pair in zip(axes, pairs):
            ax.plot(freq, result['analytical_coherence'][pair], **ANALYTICAL_KW)
            ax.plot(freq, result['sde_nonlinear_coherence'][pair], **NONLINEAR_KW)
            if show_linear:
                for key, name, colour, style in LINEAR_SERIES:
                    ckey = key + '_coherence'
                    if ckey in result and pair in result[ckey]:
                        ax.plot(freq, result[ckey][pair], color=colour, ls=style,
                                lw=1.3, alpha=0.9, label=name)
            a, b = pair
            ax.set_title(f'{PRETTY.get(a, a)} — {PRETTY.get(b, b)}')
            ax.set_xlabel('Frequency (Hz)')
            # Coherence is bounded in [0, 1]; a log axis would misrepresent it.
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, f_max)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, ls=':', alpha=0.4)
        axes[0].set_ylabel('Squared coherence')
        axes[0].legend(loc='lower left')
        fig.suptitle(_suptitle('V1–V4 LFP coherence', result, gamma, contrast), fontsize=15)
        _save(fig, out_path)
        plt.close(fig)


def plot_traces(result, gamma, contrast, out_path, seconds=0.2):
    """Nonlinear vs. noise-matched linear trajectories from a single trial."""
    if 'trace_nonlinear' not in result:
        return
    fs = result['fs_rec']
    labels = _present_labels(result)
    rows = [result['labels'].index(lab) for lab in labels]
    nl = result['trace_nonlinear']
    lin = result.get('trace_linear')
    n_show = int(min(nl.shape[-1], seconds * fs))
    t = np.arange(n_show) / fs

    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(len(rows), 1, figsize=(11, 3.0 * len(rows)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, lab, r in zip(axes, labels, rows):
            ax.plot(t, nl[r, :n_show], color='#DC143C', lw=1.2, label='full nonlinear')
            if lin is not None:
                ax.plot(t, lin[r, :n_show], color='#32CD32', lw=1.2, ls='--',
                        label='linearised, identical noise')
            ax.set_ylabel(PRETTY.get(lab, lab), fontsize=12)
            ax.grid(True, ls=':', alpha=0.4)
        axes[0].legend(loc='upper right', ncol=2)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(rf'Deviation from the fixed point, one shared noise realisation '
                     rf'($c={contrast}$, $\gamma_1={gamma}$)', fontsize=15)
        _save(fig, out_path)
        plt.close(fig)


def _suptitle(what, result, gamma, contrast):
    """Two lines, so a single-panel figure does not clip the condition string."""
    lp = 'on' if result.get('low_pass_add') else 'off'
    return (rf'{what}: analytical vs. SDE' '\n'
            rf'$c={contrast}$, $\gamma_1={gamma}$, '
            rf'{result.get("n_trials_note", "")}low-pass term {lp}')


def main(results_dir, show_linear=False):
    data_path = os.path.join(results_dir, 'Data', 'sde_validation.npy')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run Analysis/SDE_validation.py first.")
        sys.exit(1)
    payload = np.load(data_path, allow_pickle=True).item()

    plots_dir = os.path.join(results_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    settings = payload['settings']
    note = f"{settings['n_trials']} trials x {settings['T']}s, "
    for (gamma, contrast), result in payload['results'].items():
        if 'error' in result:
            print(f"Skipping c={contrast}, gamma={gamma}: condition failed "
                  f"({result['error']})")
            continue
        result['n_trials_note'] = note
        tag = f"c{contrast}_g{gamma}".replace('.', 'p')
        plot_psd(result, gamma, contrast,
                 os.path.join(plots_dir, f'sde_validation_psd_{tag}.pdf'), show_linear)
        plot_coherence(result, gamma, contrast,
                       os.path.join(plots_dir, f'sde_validation_coherence_{tag}.pdf'),
                       show_linear)
        plot_traces(result, gamma, contrast,
                    os.path.join(plots_dir, f'sde_validation_traces_{tag}.pdf'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plot the SDE validation of the analytical spectra and coherence')
    p.add_argument('results_dir', help='Results directory containing Data/sde_validation.npy')
    p.add_argument('--show-linear', action='store_true',
                   help='also draw the linear-SDE control series (diagnostic)')
    args = p.parse_args()
    main(args.results_dir, args.show_linear)
