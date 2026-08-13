"""Per-sigma_f overlays of V1 PSD and V1-V4 coherence across all contrasts.

Usage:
    python Plotting/Plot_SDE_validation_sigmaf.py --out-dir DIR RUN1 [RUN2 ...]

Each RUN is a results directory holding Data/sde_validation.npy (one sigma_f value,
all contrasts, analytical + full-nonlinear SDE, low_pass_add=True). One figure pair is
written per run:

  * PSD      -- V1 LFP (y1) vs frequency, one coloured line per contrast, SDE solid and
                analytical dashed. Both families are divided by the SAME curve: the
                analytical c=0 PSD of that run. Dividing both by one baseline keeps the
                one-sided x2 convention intact and shows the scale relationship.
  * Coherence-- V1-V4 magnitude-squared coherence vs frequency, one line per contrast,
                SDE solid and analytical dashed. Plotted raw in [0, 1] (no normalisation).

Colour encodes contrast; line style encodes method (solid = SDE, dashed = analytical).
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RC = {
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'lines.linewidth': 1.8,
    'figure.constrained_layout.use': True,
}

PSD_LABEL = 'y1'                 # V1 LFP (raw membrane potential)
COH_PAIR = ('y1', 'y4')         # V1-V4 pair
F_MAX = 80.0
CMAP = plt.cm.viridis


def _load(run_dir):
    path = os.path.join(run_dir, 'Data', 'sde_validation.npy')
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run/merge the validation for {run_dir} first.")
    return np.load(path, allow_pickle=True).item()


def _sigma_f(payload):
    return float(payload['noise_params']['sigma_f'])


def _sigma_tag(sigma_f):
    return f"{sigma_f:g}".replace('.', 'p').replace('-', 'm')


def _style_freq_axis(ax):
    ax.set_xlim(0, F_MAX)
    ticks = np.arange(0, F_MAX + 1, 20)
    ax.set_xticks(ticks)
    ax.set_xticks(np.arange(0, F_MAX + 1, 10), minor=True)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True, ls=':', alpha=0.35)


def _contrast_colors(contrasts):
    """One colour per contrast, evenly spaced along the colormap (contrasts are geometric)."""
    return {c: CMAP(x) for c, x in zip(contrasts, np.linspace(0.08, 0.92, len(contrasts)))}


def _method_handles():
    return [Line2D([0], [0], color='k', ls='-', lw=1.8, label='SDE (full nonlinear)'),
            Line2D([0], [0], color='k', ls='--', lw=1.8, label='Analytical (linearised)')]


def _contrast_handles(contrasts, colors):
    return [Line2D([0], [0], color=colors[c], lw=2.4, label=f'c = {c:g}') for c in contrasts]


def plot_psd(payload, gamma, out_path, draw_background=True, title_suffix=''):
    """V1 PSD across contrasts, both families divided by the analytical c=0 PSD.

    `draw_background=False` keeps the lowest contrast as the normalisation divisor but does not
    draw its line (used when the c=0 background is not a curve of interest).
    """
    results = payload['results']
    contrasts = sorted({float(c) for (g, c) in results if g == gamma})
    background_c = min(contrasts)
    bg_key = (gamma, background_c)
    if bg_key not in results:
        raise SystemExit(f"background c={background_c} missing for gamma={gamma}; "
                         f"cannot normalise by the analytical baseline.")
    p_ana_bg = np.asarray(results[bg_key]['analytical'][PSD_LABEL], dtype=float)
    freq = np.asarray(results[bg_key]['freq'], dtype=float)
    colors = _contrast_colors(contrasts)
    drawn = contrasts if draw_background else [c for c in contrasts if c != background_c]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        for c in drawn:
            r = results[(gamma, c)]
            ana = np.asarray(r['analytical'][PSD_LABEL], dtype=float) / p_ana_bg
            sde = np.asarray(r['sde_nonlinear'][PSD_LABEL]['mean'], dtype=float) / p_ana_bg
            ax.plot(freq, sde, color=colors[c], ls='-', lw=1.8)
            ax.plot(freq, ana, color=colors[c], ls='--', lw=1.6, alpha=0.9)
        _style_freq_axis(ax)
        ax.set_yscale('log')
        ax.set_ylabel(r'V1 power  $P(c)\,/\,P_{\mathrm{analytical}}(c{=}0)$')
        sigma_f = _sigma_f(payload)
        ax.set_title(rf'V1 LFP power spectra ($\sigma_f = {sigma_f:g}$, low-pass on){title_suffix}')
        leg1 = ax.legend(handles=_contrast_handles(drawn, colors),
                         loc='upper right', ncol=2, title='contrast')
        ax.add_artist(leg1)
        ax.legend(handles=_method_handles(), loc='lower left')
        _save(fig, out_path)
        plt.close(fig)


def plot_coherence(payload, gamma, out_path, draw_background=True, title_suffix=''):
    """V1-V4 coherence across contrasts, raw (no normalisation).

    `draw_background=False` omits the lowest-contrast (c=0) line.
    """
    results = payload['results']
    contrasts = sorted({float(c) for (g, c) in results if g == gamma})
    background_c = min(contrasts)
    colors = _contrast_colors(contrasts)
    freq = np.asarray(next(iter(results.values()))['freq'], dtype=float)
    drawn = contrasts if draw_background else [c for c in contrasts if c != background_c]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        used = []
        for c in drawn:
            r = results[(gamma, c)]
            ac = r.get('analytical_coherence', {})
            sc = r.get('sde_nonlinear_coherence', {})
            if COH_PAIR not in ac or COH_PAIR not in sc:
                continue
            ax.plot(freq, np.asarray(sc[COH_PAIR], dtype=float), color=colors[c], ls='-', lw=1.8)
            ax.plot(freq, np.asarray(ac[COH_PAIR], dtype=float), color=colors[c], ls='--',
                    lw=1.6, alpha=0.9)
            used.append(c)
        if not used:
            print(f"No {COH_PAIR} coherence in {out_path}; skipping.")
            plt.close(fig)
            return
        _style_freq_axis(ax)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel('V1-V4 coherence')
        sigma_f = _sigma_f(payload)
        ax.set_title(rf'V1-V4 coherence ($\sigma_f = {sigma_f:g}$, low-pass on){title_suffix}')
        leg1 = ax.legend(handles=_contrast_handles(used, colors),
                         loc='upper right', ncol=2, title='contrast')
        ax.add_artist(leg1)
        ax.legend(handles=_method_handles(), loc='lower left')
        _save(fig, out_path)
        plt.close(fig)


def _save(fig, out_path):
    fig.savefig(out_path)
    png = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png)
    print(f"Saved {out_path} and {os.path.basename(png)}")


def main(run_dirs, out_dir, draw_background=True):
    os.makedirs(out_dir, exist_ok=True)
    for run_dir in run_dirs:
        payload = _load(run_dir)
        sigma_f = _sigma_f(payload)
        tag = _sigma_tag(sigma_f)
        gammas = sorted({float(g) for (g, c) in payload['results']})
        for gamma in gammas:
            gtag = f"g{gamma:g}".replace('.', 'p')
            print(f"\n{run_dir}: sigma_f={sigma_f:g}, gamma={gamma:g}")
            plot_psd(payload, gamma,
                     os.path.join(out_dir, f'sigmaf_{tag}_{gtag}_v1_psd.pdf'),
                     draw_background=draw_background)
            plot_coherence(payload, gamma,
                           os.path.join(out_dir, f'sigmaf_{tag}_{gtag}_v1v4_coherence.pdf'),
                           draw_background=draw_background)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Per-sigma_f V1 PSD and V1-V4 coherence overlays across contrasts')
    p.add_argument('run_dirs', nargs='+',
                   help='results dirs, each with Data/sde_validation.npy (one sigma_f each)')
    p.add_argument('--out-dir', required=True, help='directory to write the figures into')
    p.add_argument('--drop-background', action='store_true',
                   help='keep the lowest contrast (c=0) as the PSD divisor but do not draw it')
    args = p.parse_args()
    main(args.run_dirs, args.out_dir, draw_background=not args.drop_background)
