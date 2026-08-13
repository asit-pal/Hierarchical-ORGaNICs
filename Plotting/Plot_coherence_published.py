"""Published-style V1-V4 coherence figures from the SDE-validation noise_sweep.

Usage:
    python Plotting/Plot_coherence_published.py [--pgf] --out-dir DIR RUN1 [RUN2 ...]

Renders the noise_sweep coherence in the paper's aesthetic (matching the supplied
Plot_coherence_fixed_gamma / power-spectra scripts): Reds colormap by contrast, the shared journal
rcParams from Plotting.setup_plot_params, LaTeX `\\textnormal{}` labels, coherence normalised so the
peak sits at 1, x in 0-80 Hz, no inset.

Fonts:
  --pgf : true LaTeX via the pgf backend + lualatex + fontspec/Latin Modern (== the paper's
          Computer Modern). Outputs PDF (paper format) and a PNG preview. Use this when a LaTeX
          toolchain with lualatex is available (the container's `latex` engine cannot be used --
          it needs cm-super/type1ec.sty, which is absent).
  default: mathtext with Computer Modern (`mathtext.fontset='cm'`) -- no LaTeX needed, visually
           close to the paper. Outputs SVG + PNG.

Each RUN is a results dir with Data/sde_validation.npy (one sigma_f). Both curves are drawn per
contrast: SDE (full nonlinear) solid, analytical (linearised) dashed, same Reds colour. c=0 is
dropped. Both families share one normaliser (the joint max over contrasts and methods).
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Backend must be chosen before pyplot is imported.
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

COH_PAIR = ('y1', 'y4')
# The paper's contrast set (percent) — matches the 8 stimulus contrasts (x100), c=0 excluded.
CONTRAST_VALUES = np.array([2.5, 3.7, 6.1, 9.7, 16.3, 35.9, 50.3, 72.0])


def _reds_colors(n=None):
    """Reds colours spaced 0.2..0.8 by contrast (as in the paper scripts)."""
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
        # Under pgf, layout engines/tight-bbox would measure text via the mathtext parser (which
        # does not know \textnormal). Use fixed margins instead (as the published scripts do), so
        # the only rendering is lualatex.
        rc["figure.constrained_layout.use"] = False
    else:
        # No LaTeX toolchain path: use Computer Modern via mathtext so the look matches the
        # paper's LaTeX CM rather than the DejaVu default.
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def _labels():
    """Paper labels. Under pgf, use PLAIN strings: lualatex typesets them in Latin Modern (== CM)
    without invoking the mathtext parser (which only runs on `$...$` and rejects \\textnormal).
    Under mathtext, use \\mathrm so the CM math font is used."""
    if _USE_PGF:
        return ('Frequency (Hz)', 'V1-V4 Coherence', lambda x: str(x), 'SDE', 'Analytical')
    return (r'$\mathrm{Frequency\ (Hz)}$', r'$\mathrm{V1\text{-}V4\ Coherence}$',
            lambda x: r'$\mathrm{' + str(x) + '}$',
            r'$\mathrm{SDE}$', r'$\mathrm{Analytical}$')


def _sigma_f(payload):
    return float(payload['noise_params']['sigma_f'])


def _sigma_tag(sigma_f):
    return f"{sigma_f:g}".replace('.', 'p').replace('-', 'm')


def _save(fig, out_base):
    if _USE_PGF:
        # Fixed margins (no tight bbox) so nothing measures text via mathtext; PDF only (lualatex).
        fig.subplots_adjust(left=0.20, right=0.96, top=0.96, bottom=0.18)
        path = f"{out_base}.pdf"
        fig.savefig(path, format='pdf')
        print(f"Saved {path}")
        return
    for ext in ('svg', 'png'):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=400, bbox_inches='tight', format=ext)
        print(f"Saved {path}")


def _plot_run(payload, gamma, out_base):
    results = payload['results']
    contrasts = sorted(c for (g, c) in results if g == gamma and float(c) > 0.0)
    colors = _reds_colors() if len(contrasts) == len(CONTRAST_VALUES) else _reds_colors(len(contrasts))

    # Joint normaliser: peak of SDE and analytical coherence over all plotted contrasts.
    norm = 0.0
    for c in contrasts:
        r = results[(gamma, c)]
        for key in ('sde_nonlinear_coherence', 'analytical_coherence'):
            arr = r.get(key, {}).get(COH_PAIR)
            if arr is not None:
                norm = max(norm, float(np.max(np.asarray(arr, dtype=float))))
    if norm <= 0:
        print(f"No {COH_PAIR} coherence for gamma={gamma}; skipping.")
        return

    xlab, ylab, tick_fmt, sde_lab, ana_lab = _labels()
    fig, ax = plt.subplots()
    # SDE is the solid line; analytical is a dashed line carrying sparse open circles so the two
    # remain distinguishable where they overlap.
    for c, color in zip(contrasts, colors):
        r = results[(gamma, c)]
        freq = np.asarray(r['freq'], dtype=float)
        sde = np.asarray(r['sde_nonlinear_coherence'][COH_PAIR], dtype=float) / norm
        ana = np.asarray(r['analytical_coherence'][COH_PAIR], dtype=float) / norm
        ax.plot(freq, sde, ls='-', lw=10, color=color)
        ax.plot(freq, ana, ls='--', lw=4, color=color, marker='o', markevery=28,
                markersize=26, markerfacecolor='white', markeredgecolor=color, markeredgewidth=5)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 80)
    xticks = np.array([0, 20, 40, 60, 80])
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0, 81, 10), minor=True)
    ax.set_xticklabels([tick_fmt(x) for x in xticks])
    ax.set_ylim(0, 1.05)
    yticks = np.array([0.0, 0.5, 1.0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([tick_fmt(f'{y:.1f}') for y in yticks])
    ax.tick_params(axis='both', which='minor', pad=10)
    ax.tick_params(axis='both', which='major', pad=10)

    # Minimal method legend (contrast is encoded by the Reds gradient). Placed in the empty
    # top-right corner, compact, so it does not overlap the gamma-band peaks.
    handles = [Line2D([0], [0], color='0.25', ls='-', lw=9, label=sde_lab),
               Line2D([0], [0], color='0.25', ls='--', lw=4, marker='o', markersize=18,
                      markerfacecolor='white', markeredgecolor='0.25', markeredgewidth=4,
                      label=ana_lab)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0),
              fontsize=58, handlelength=2.6, handletextpad=0.4, labelspacing=0.3, borderpad=0.4)

    _save(fig, out_base)
    plt.close(fig)


def main(run_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _apply_style()
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
            out_base = os.path.join(out_dir, f'coherence_published_sf{tag}_{gtag}')
            print(f"\n{run_dir}: sigma_f={sigma_f:g}, gamma={gamma:g}")
            _plot_run(payload, gamma, out_base)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Published-style V1-V4 coherence overlays (SDE vs analytical)')
    p.add_argument('run_dirs', nargs='+', help='results dirs with Data/sde_validation.npy')
    p.add_argument('--out-dir', required=True, help='directory to write the figures into')
    p.add_argument('--pgf', action='store_true',
                   help='render true LaTeX via pgf+lualatex (PDF+PNG); default is mathtext CM (SVG+PNG)')
    args = p.parse_args()
    main(args.run_dirs, args.out_dir)
