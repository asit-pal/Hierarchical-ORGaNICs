"""Overlay the analytical and simulated spectra under the *published* normalisation.

Usage:
    python Plotting/Plot_SDE_validation_contrast.py <results_dir> [--gamma G] [--n-boot N]

Reads   <results_dir>/Data/sde_validation.npy   (written by Analysis/merge_sde_validation.py)
Writes  <results_dir>/Plots/sde_validation_index_{grid,published}_*.{pdf,png}

`Plot_SDE_validation.py` compares one contrast at a time in absolute PSD units. That
checks the machinery. This script checks the *figure*: it puts both families through the
contrast index `(P - P_bg)/(P + P_bg)` that
`Plot_PS_fixed_gamma_with_power_decay.py:112-115` plots, on the published axes, so the
question being answered is "would the published panel look the same if it had been made
from direct simulation instead of the linearisation?"

Each series is normalised against **its own** background. Crossing them (simulated
against analytical background) would reintroduce the one-sided factor of 2 that
`matrix_solution` and `scipy.signal.welch` disagree about; keeping them separate makes
the comparison convention-free.

Two figures:

*grid*      -- diagnostic. Two columns (V1, V4), overlay on top, residual below, at font
               sizes a human can read. This is the one to look at.
*published* -- the same content in the paper's own rcParams (single panel, 90 pt labels,
               lw 12), so it can be laid beside the real figure for the same config.
               Unreadable on screen, correct on the page.
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from Plotting import setup_plot_params
from Utils.SDE_contrast_index import index_curves

LFP_LABELS = ('y1', 'y4')
PRETTY = {'y1': r'V1 LFP  ($y_1$)', 'y4': r'V4 LFP  ($y_4$)'}

# Readable diagnostic rcParams; the journal ones (90 pt labels, lw 12) are for a single
# panel and cannot carry an 8-contrast overlay plus a legend.
DIAG_RC = {
    'figure.dpi': 150, 'savefig.dpi': 200,
    'axes.linewidth': 1.2, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 10, 'legend.frameon': False,
    'figure.constrained_layout.use': True,
}


def published_reds(c_vals):
    """The published contrast colour map.

    Reproduces `Plot_PS_fixed_gamma_with_power_decay.py:17-24` line for line. Copied
    rather than imported: that module calls `setup_plot_params()` at import time and
    mutates global rcParams as a side effect.
    """
    contrast_values = np.asarray(sorted(c_vals), dtype=float) * 100.0
    span = contrast_values.max() - contrast_values.min()
    positions = ((contrast_values - contrast_values.min()) / span if span > 0
                 else np.full(contrast_values.shape, 0.5))
    positions = 0.2 + positions * 0.6
    cmap = plt.get_cmap('Reds')
    return {c: cmap(p) for c, p in zip(sorted(c_vals), positions)}


def _darker(rgba, factor=0.55):
    """A darker shade of the same hue, so the dashed simulated curve stays legible on
    top of the thick solid analytical one without changing which contrast it reads as."""
    r, g, b = rgba[:3]
    return (r * factor, g * factor, b * factor, 1.0)


def _save(fig, out_path):
    fig.savefig(out_path)
    png = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png)
    plt.close(fig)
    print(f"Saved {out_path} and {os.path.basename(png)}")


def _style_freq_axis(ax, f_max=80.0):
    """The published x-axis: linear 0-80 Hz, majors every 20, minors every 10."""
    ax.set_xlim(0, f_max)
    ticks = np.arange(0, f_max + 1, 20)
    ax.set_xticks(ticks)
    ax.set_xticks(np.arange(0, f_max + 1, 10), minor=True)
    ax.set_xticklabels([str(int(t)) for t in ticks])


def _style_index_axis(ax):
    """The published y-axis for the contrast index."""
    ax.set_ylim(-0.25, 1.0)
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.set_yticklabels(['0.0', '0.4', '0.8'])


def plot_index_panel(ax, curves, colours, lw_ana=2.6, lw_num=1.6, sem_band=True):
    """Analytical (solid) vs simulated (dashed) contrast index, one line per contrast."""
    for c, cur in sorted(curves.items()):
        colour = colours[c]
        ax.plot(cur['freq'], cur['analytical'], '-', color=colour, lw=lw_ana)
        ax.plot(cur['freq'], cur['numerical'], '--', color=_darker(colour), lw=lw_num)
        if sem_band and cur['sem'] is not None:
            ax.fill_between(cur['freq'], cur['numerical'] - 1.96 * cur['sem'],
                            cur['numerical'] + 1.96 * cur['sem'],
                            color=_darker(colour), alpha=0.18, lw=0)
    _style_freq_axis(ax)
    _style_index_axis(ax)


def plot_residual_panel(ax, curves, colours):
    """Simulated minus analytical index, with the propagated 95% estimator band.

    Anything inside the grey band is estimator noise. Anything outside it, and
    *consistent in sign across contrasts*, is a background artefact rather than a
    linearisation error -- every curve divides by the same P_bg realisation.
    """
    worst = 0.0
    sems = []
    for c, cur in sorted(curves.items()):
        resid = cur['numerical'] - cur['analytical']
        ax.plot(cur['freq'], resid, '-', color=colours[c], lw=1.4)
        worst = max(worst, float(np.max(np.abs(resid))))
        if cur['sem'] is not None:
            sems.append(cur['sem'])
    if sems:
        band = 1.96 * np.median(np.asarray(sems), axis=0)
        ax.fill_between(curves[sorted(curves)[0]]['freq'], -band, band,
                        color='0.6', alpha=0.30, lw=0, zorder=0)
    ax.axhline(0.0, color='k', lw=0.9, zorder=1)
    _style_freq_axis(ax)
    lim = max(0.02, 1.2 * worst)
    ax.set_ylim(-lim, lim)


def add_global_loglog_inset(ax, results, gamma, label, background_c):
    """Raw power on log-log, normalised by the global maximum over all conditions.

    This is the published inset's normaliser (`:146-154`) -- max over *all* (gamma, c)
    keys -- not `Plot_SDE_validation._add_loglog_inset`'s per-condition analytical max.
    Each series is scaled by its own global maximum, matching how the published figure
    would treat whichever data it was given.
    """
    keys = [k for k in results if k[0] == gamma]
    ana_scale = max(np.max(results[k]['analytical'][label]) for k in keys)
    num_scale = max(np.max(results[k]['sde_nonlinear'][label]['mean']) for k in keys)

    inset = ax.inset_axes([0.60, 0.58, 0.38, 0.40])
    freq = np.asarray(results[keys[0]]['freq'])
    for k in sorted(keys, key=lambda kk: kk[1]):
        if k[1] == background_c:
            continue
        inset.loglog(freq, results[k]['analytical'][label] / ana_scale, '-',
                     color='k', lw=0.8, alpha=0.7)
        inset.loglog(freq, results[k]['sde_nonlinear'][label]['mean'] / num_scale, '--',
                     color='#DC143C', lw=0.8, alpha=0.7)

    # 1/f^4 guide, anchored and offset as in the published inset.
    ref = results[max(keys, key=lambda kk: kk[1])]['analytical'][label] / ana_scale
    f_anchor = min(300.0, 0.5 * freq.max())
    anchor = ref[np.abs(freq - f_anchor).argmin()]
    guide_f = np.array([f_anchor, freq.max()])
    inset.loglog(guide_f, anchor * (f_anchor / guide_f) ** 4.0 / 5.0,
                 dashes=[4, 2], color='red', lw=1.2)
    inset.text(f_anchor * 0.8, anchor * 0.02, r'$1/f^{4}$', color='red',
               fontsize=8, ha='center', va='bottom')
    inset.set_xlim(1, freq.max())
    inset.set_xlabel('Frequency', labelpad=1, fontsize=8)
    inset.set_ylabel('Power', labelpad=1, fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_box_aspect(1)
    return inset


def _legend_handles(curves, colours):
    handles = [mlines.Line2D([], [], color=colours[c], lw=2.4, label=f'{c * 100:.1f}')
               for c in sorted(curves)]
    style = [mlines.Line2D([], [], color='k', lw=2.4, ls='-', label='analytical'),
             mlines.Line2D([], [], color='k', lw=1.6, ls='--', label='SDE (full nonlinear)')]
    return handles, style


def plot_index_grid(payload, gamma, out_path, n_boot=0, labels=LFP_LABELS):
    results, settings = payload['results'], payload['settings']
    background_c = float(min(settings['c_vals']))
    labels = [l for l in labels if l in next(iter(results.values()))['labels']]

    with plt.rc_context(DIAG_RC):
        fig, axes = plt.subplots(2, len(labels), figsize=(7.4 * len(labels), 9.0),
                                 squeeze=False,
                                 gridspec_kw={'height_ratios': [2.1, 1.0]})
        method = 'none'
        for col, label in enumerate(labels):
            curves = index_curves(results, gamma, background_c, label, n_boot=n_boot)
            colours = published_reds(list(curves))
            method = next(iter(curves.values()))['sem_method']

            top, bot = axes[0][col], axes[1][col]
            plot_index_panel(top, curves, colours)
            top.set_title(PRETTY.get(label, label))
            add_global_loglog_inset(top, results, gamma, label, background_c)

            plot_residual_panel(bot, curves, colours)
            bot.set_xlabel('Frequency (Hz)')

            if col == 0:
                top.set_ylabel(r'Power normalized  $(P-P_{bg})/(P+P_{bg})$')
                bot.set_ylabel('SDE $-$ analytical')
                handles, style = _legend_handles(curves, colours)
                top.legend(handles=handles + style, title='contrast (%)', ncol=2,
                           loc='upper left', fontsize=9, title_fontsize=9)

        fig.suptitle('Published normalisation: analytical vs. full nonlinear SDE\n'
                     rf'$\gamma_1={gamma}$, background $c={background_c}$, '
                     f"{settings['n_trials']} trials x {settings['T']}s, "
                     f'band from {method}', fontsize=14)
        _save(fig, out_path)


def plot_index_published(payload, gamma, label, out_path, n_boot=0):
    """The same comparison in the paper's own rcParams, for a side-by-side check."""
    results, settings = payload['results'], payload['settings']
    background_c = float(min(settings['c_vals']))
    curves = index_curves(results, gamma, background_c, label, n_boot=n_boot)
    colours = published_reds(list(curves))

    with plt.rc_context():
        setup_plot_params()
        fig, ax = plt.subplots()
        plot_index_panel(ax, curves, colours, lw_ana=12, lw_num=4, sem_band=False)
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel(f'{"V1" if label == "y1" else "V4"} Power Normalized')
        ax.tick_params(axis='both', which='both', pad=10)
        handles = [mlines.Line2D([], [], color=colours[c], lw=12, label=f'{c * 100:.1f}')
                   for c in sorted(curves)]
        ax.legend(handles=handles, loc='upper right')
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        _save(fig, out_path)


def main(results_dir, gamma=None, n_boot=0):
    data_path = os.path.join(results_dir, 'Data', 'sde_validation.npy')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run Analysis/merge_sde_validation.py first.")
        sys.exit(1)
    payload = np.load(data_path, allow_pickle=True).item()

    settings = payload['settings']
    contrasts = sorted(float(c) for c in settings['c_vals'])
    if len(contrasts) < 2:
        print(f"Only one contrast in the payload ({contrasts}). The published "
              f"normalisation is a contrast index and needs a lower-contrast background; "
              f"run the full c_vals sweep first.")
        sys.exit(1)

    plots_dir = os.path.join(results_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    gammas = [gamma] if gamma is not None else sorted({g for g, _ in payload['results']})
    for g in gammas:
        tag = f"g{g}".replace('.', 'p')
        plot_index_grid(payload, g,
                        os.path.join(plots_dir, f'sde_validation_index_grid_{tag}.pdf'),
                        n_boot=n_boot)
        for label in LFP_LABELS:
            if label not in next(iter(payload['results'].values()))['labels']:
                continue
            plot_index_published(
                payload, g, label,
                os.path.join(plots_dir,
                             f'sde_validation_index_published_{label}_{tag}.pdf'),
                n_boot=n_boot)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compare analytical and SDE spectra under the published normalisation')
    p.add_argument('results_dir', help='Results directory containing Data/sde_validation.npy')
    p.add_argument('--gamma', type=float, default=None, help='plot only this gamma1')
    p.add_argument('--n-boot', dest='n_boot', type=int, default=0,
                   help='paired bootstrap draws for the error band; 0 uses the delta method')
    args = p.parse_args()
    main(args.results_dir, args.gamma, args.n_boot)
