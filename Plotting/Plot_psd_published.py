"""Published-style V1 PSD figures (contrast index) from the SDE-validation noise_sweep.

Usage:
    python Plotting/Plot_psd_published.py [--pgf] --out-dir DIR RUN1 [RUN2 ...]

The paper's PSD panel plots the contrast index (P - P_bg)/(P + P_bg) against the lowest-contrast
background, on a linear [-0.25, 1] axis -- this is that panel WITHOUT the log-log inset, restyled
to match the coherence figures: Reds by contrast, SDE solid + analytical dashed-with-circles,
compact top-right legend.

Each series is normalised against its OWN c=0 background (the index convention that makes the
one-sided x2 cancel): SDE index uses the SDE c=0 PSD, analytical index uses the analytical c=0 PSD.
c=0 is the background and is not drawn.

Fonts: --pgf renders true LaTeX via pgf+lualatex+Latin Modern (PDF); default is Computer Modern
mathtext (SVG+PNG). See Plot_coherence_published.py for the toolchain notes.
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

_USE_PGF = '--pgf' in sys.argv

import matplotlib
if _USE_PGF:
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'lualatex',
        'pgf.rcfonts': False,
        'pgf.preamble': '\n'.join([
            r'\usepackage{fontspec}',
            r'\setmainfont{Latin Modern Roman}',
            r'\usepackage{amsmath}',
        ]),
    })
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Plotting import setup_plot_params
from Utils.SDE_contrast_index import contrast_index

PSD_LABEL = 'y1'                 # V1 LFP
CONTRAST_VALUES = np.array([2.5, 3.7, 6.1, 9.7, 16.3, 35.9, 50.3, 72.0])


def _reds_colors(n=None):
    if n is None:
        positions = (CONTRAST_VALUES - CONTRAST_VALUES.min()) / (CONTRAST_VALUES.max() - CONTRAST_VALUES.min())
        positions = 0.2 + positions * 0.6
    else:
        positions = [0.2 + 0.6 * i / max(1, n - 1) for i in range(n)]
    cmap = matplotlib.colormaps.get_cmap('Reds')
    return [cmap(p) for p in positions]


def _apply_style():
    setup_plot_params()
    rc = {
        "lines.linewidth": 12,
        "lines.markersize": 10,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.2,
        "legend.labelspacing": 0.3,
        "axes.prop_cycle": plt.cycler(color=_reds_colors()),
        "font.family": "serif",
    }
    if _USE_PGF:
        rc["figure.constrained_layout.use"] = False
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def _labels():
    if _USE_PGF:
        return ('Frequency (Hz)', 'V1 Power Normalized', lambda x: str(x), 'SDE', 'Analytical')
    return (r'$\mathrm{Frequency\ (Hz)}$', r'$\mathrm{V1\ Power\ Normalized}$',
            lambda x: r'$\mathrm{' + str(x) + '}$', r'$\mathrm{SDE}$', r'$\mathrm{Analytical}$')


def _sigma_f(payload):
    return float(payload['noise_params']['sigma_f'])


def _sigma_tag(sigma_f):
    return f"{sigma_f:g}".replace('.', 'p').replace('-', 'm')


def _save(fig, out_base):
    if _USE_PGF:
        fig.subplots_adjust(left=0.20, right=0.96, top=0.96, bottom=0.18)
        path = f"{out_base}.pdf"
        fig.savefig(path, format='pdf')
        print(f"Saved {path}")
        return
    for ext in ('svg', 'png'):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=400, bbox_inches='tight', format=ext)
        print(f"Saved {path}")


def _plot_run(payload, gamma, out_base, bg_mode='own'):
    results = payload['results']
    contrasts = sorted(float(c) for (g, c) in results if g == gamma)
    background_c = min(contrasts)                       # c = 0
    drawn = [c for c in contrasts if c > background_c]  # stimulus contrasts
    colors = _reds_colors() if len(drawn) == len(CONTRAST_VALUES) else _reds_colors(len(drawn))

    bg = results.get((gamma, background_c))
    if bg is None:
        raise SystemExit(f"background c={background_c} missing for gamma={gamma}")
    p_sde_bg = np.asarray(bg['sde_nonlinear'][PSD_LABEL]['mean'], dtype=float)
    p_ana_bg = np.asarray(bg['analytical'][PSD_LABEL], dtype=float)
    # bg_mode='own': each series indexed against its own c=0. bg_mode='ana_c0': BOTH series use the
    # analytical c=0 as the background.
    sde_bg = p_ana_bg if bg_mode == 'ana_c0' else p_sde_bg

    xlab, ylab, tick_fmt, sde_lab, ana_lab = _labels()
    fig, ax = plt.subplots()
    for c, color in zip(drawn, colors):
        r = results[(gamma, c)]
        freq = np.asarray(r['freq'], dtype=float)
        sde = contrast_index(np.asarray(r['sde_nonlinear'][PSD_LABEL]['mean'], dtype=float), sde_bg)
        ana = contrast_index(np.asarray(r['analytical'][PSD_LABEL], dtype=float), p_ana_bg)
        ax.plot(freq, sde, ls='-', lw=10, color=color)
        ax.plot(freq, ana, ls='--', lw=4, color=color, marker='o', markevery=28,
                markersize=26, markerfacecolor='white', markeredgecolor=color, markeredgewidth=5)

    ax.set_xlabel(xlab)
    # "V1 Power Normalized" is long; at the full 90pt the rotated label overruns the axis height,
    # so shrink just the ylabel (xlabel stays at the shared size).
    ax.set_ylabel(ylab, fontsize=68)
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)
    ax.set_xticklabels([tick_fmt(x) for x in xticks])
    ax.set_ylim(-0.25, 1.0)
    yticks = np.array([0.0, 0.4, 0.8])
    ax.set_yticks(yticks)
    ax.set_yticklabels([tick_fmt(f'{y:.1f}') for y in yticks])
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

    handles = [Line2D([0], [0], color='0.25', ls='-', lw=9, label=sde_lab),
               Line2D([0], [0], color='0.25', ls='--', lw=4, marker='o', markersize=18,
                      markerfacecolor='white', markeredgecolor='0.25', markeredgewidth=4,
                      label=ana_lab)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0),
              fontsize=58, handlelength=2.6, handletextpad=0.4, labelspacing=0.3, borderpad=0.4)

    _save(fig, out_base)
    plt.close(fig)


def main(run_dirs, out_dir, bg_mode='own'):
    os.makedirs(out_dir, exist_ok=True)
    _apply_style()
    prefix = 'psd_sharedbg_published' if bg_mode == 'ana_c0' else 'psd_published'
    for run_dir in run_dirs:
        data_path = os.path.join(run_dir, 'Data', 'sde_validation.npy')
        if not os.path.exists(data_path):
            raise SystemExit(f"{data_path} not found")
        payload = np.load(data_path, allow_pickle=True).item()
        sigma_f = _sigma_f(payload)
        tag = _sigma_tag(sigma_f)
        gammas = sorted({float(g) for (g, c) in payload['results']})
        for gamma in gammas:
            gtag = f"g{gamma:g}".replace('.', 'p')
            out_base = os.path.join(out_dir, f'{prefix}_sf{tag}_{gtag}')
            print(f"\n{run_dir}: sigma_f={sigma_f:g}, gamma={gamma:g}, bg={bg_mode}")
            _plot_run(payload, gamma, out_base, bg_mode=bg_mode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Published-style V1 PSD contrast-index overlays (SDE vs analytical)')
    p.add_argument('run_dirs', nargs='+', help='results dirs with Data/sde_validation.npy')
    p.add_argument('--out-dir', required=True, help='directory to write the figures into')
    p.add_argument('--pgf', action='store_true',
                   help='render true LaTeX via pgf+lualatex (PDF); default is mathtext CM (SVG+PNG)')
    p.add_argument('--bg', choices=['own', 'ana_c0'], default='own',
                   help="contrast-index background: 'own' (each series vs its own c=0) or "
                        "'ana_c0' (both series vs the analytical c=0)")
    args = p.parse_args()
    main(args.run_dirs, args.out_dir, bg_mode=args.bg)
