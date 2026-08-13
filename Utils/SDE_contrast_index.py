"""The published power-spectrum normalisation, and the machinery to compare two
families of spectra under it.

`Plot_PS_fixed_gamma_with_power_decay.py` does not plot power. It plots a *contrast
index* against the lowest contrast in the sweep:

    I(f) = (P(f) - P_bg(f)) / (P(f) + P_bg(f))          P_bg = PSD at c = min(c_vals)

That is the quantity the paper reports, so it is the quantity the SDE validation has to
agree on. Three things follow, and they are why this module exists rather than an inline
subtraction:

1. **The index is a difference in [-1, 1], not a ratio.** Every discrepancy metric here is
   an absolute difference. A relative error is meaningless where `I` crosses zero, which it
   does at every frequency where the stimulus adds no power.

2. **The one-sided/two-sided factor cancels.** `matrix_solution` returns the two-sided PSD
   in angular frequency and `scipy.signal.welch` the one-sided PSD in Hz, so the analytical
   curve carries a factor of 2 the numerical one does not (see `SDE_simulation.to_onesided`).
   The index divides two spectra from the *same* series, so the factor drops out and the
   comparison is convention-free -- as long as each series is normalised against its own
   background, never against the other's.

3. **The additive low-pass term sits only in the denominator.** Writing `P = D_c + LP` with
   `LP` the deterministic `low_pass_add` measurement term, the index is

       I = (D_c - D_0) / (D_c + D_0 + 2*LP)

   so a background dominated by that exactly-known closed form cannot be moved much by
   whatever the simulation does to `D_0`. It also means raw PSD agreement flatters itself
   wherever `LP` dominates -- at c = 0.5 the term is 93% of the y1 PSD at 20 Hz. Report
   `lp_fraction` next to any agreement number so the reader can discount it.

Used by `Analysis/merge_sde_validation.py` (metrics) and
`Plotting/Plot_SDE_validation_contrast.py` (curves), and by
`Plotting/Plot_SDE_validation.py` for the single definition of `contrast_index`.
"""

import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.SDE_simulation import low_pass_matrix


DEFAULT_LABELS = ('y1', 'y4')


# --------------------------------------------------------------------------- #
# The normalisation itself
# --------------------------------------------------------------------------- #
def contrast_index(P, P_bg):
    """(P - P_bg) / (P + P_bg), the published normalisation.

    Mirrors `Plot_PS_fixed_gamma_with_power_decay.py:112-115`. Both arguments must come
    from the *same* series (analytical with analytical, simulated with simulated);
    crossing them would reintroduce the one-sided factor of 2 this cancels.
    """
    P = np.asarray(P, dtype=float)
    P_bg = np.asarray(P_bg, dtype=float)
    return (P - P_bg) / (P + P_bg)


def contrast_index_sem(P, sem_P, P_bg, sem_bg, rho=0.0):
    """Delta-method standard error of the contrast index.

    dI/dP = 2 P_bg / (P + P_bg)^2 and dI/dP_bg = -2 P / (P + P_bg)^2, so

        var(I) = a^2 var(P) + b^2 var(P_bg) + 2 rho a b sd(P) sd(P_bg)

    with a, b the two partials (opposite in sign). `rho` is the correlation between the
    two Welch estimates. It is genuinely positive here -- `SDE_validation` seeds trial t
    with `seed + t` for *every* condition, so `P` and `P_bg` share their Wiener paths --
    which makes `rho = 0` a conservative upper bound. Use
    `contrast_index_sem_bootstrap` when the per-trial spectra are available; it captures
    that correlation instead of bounding it away.
    """
    P, P_bg = np.asarray(P, float), np.asarray(P_bg, float)
    sem_P, sem_bg = np.asarray(sem_P, float), np.asarray(sem_bg, float)
    denom = (P + P_bg) ** 2
    a = 2.0 * P_bg / denom
    b = -2.0 * P / denom
    var = (a * sem_P) ** 2 + (b * sem_bg) ** 2 + 2.0 * rho * a * b * sem_P * sem_bg
    return np.sqrt(np.maximum(var, 0.0))


def contrast_index_sem_bootstrap(P_trials, Bg_trials, n_boot=2000, seed=0):
    """Paired bootstrap SEM of the contrast index.

    Resamples ONE set of trial indices and applies it to both conditions, so the
    common-random-number correlation induced by the shared seeds survives the
    resampling. Falls back to independent resampling (and a wider interval) when the two
    conditions were run with different trial counts, since the pairing is then undefined.

    P_trials, Bg_trials: (n_trials, n_freq) per-trial PSDs, as stored in
    `result['sde_nonlinear_psd_trials'][channel]`.
    """
    P_trials = np.asarray(P_trials, float)
    Bg_trials = np.asarray(Bg_trials, float)
    n_p, n_b = P_trials.shape[0], Bg_trials.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, P_trials.shape[1]), dtype=float)
    paired = (n_p == n_b)
    for k in range(n_boot):
        idx = rng.integers(0, n_p, n_p)
        jdx = idx if paired else rng.integers(0, n_b, n_b)
        draws[k] = contrast_index(P_trials[idx].mean(axis=0), Bg_trials[jdx].mean(axis=0))
    return draws.std(axis=0, ddof=1), paired


def lp_fraction(freq, analytical_psd, noise_params):
    """Fraction of a one-sided analytical PSD that is the deterministic low-pass term.

    Reuses `SDE_simulation.low_pass_matrix` so this can never drift from what
    `SDE_validation.run_condition` actually added. With k = 1 the shape factor is
    `1 + rho` (the diagonal entry); the factor of 2 converts two-sided to one-sided,
    exactly as `to_onesided` does for the rest of the spectrum.

    A value near 1 means the "agreement" at that frequency is largely a closed-form
    constant compared with itself.
    """
    lp = 2.0 * low_pass_matrix(np.asarray(freq, float), noise_params['noise_sigma'],
                               noise_params['noise_tau'], noise_params['rho'], 1)[:, 0, 0]
    return lp / np.asarray(analytical_psd, float)


# --------------------------------------------------------------------------- #
# Peak location
# --------------------------------------------------------------------------- #
def peak(freq, y):
    """(f_peak, y_peak, at_edge) with 3-point parabolic interpolation.

    The Welch grid is 0.25 Hz, so a bare argmax quantises any peak-shift comparison to
    that step. `at_edge` is True when the maximum lands on the first or last bin, which
    means the spectrum is monotone in-band and the peak location is an artefact of where
    the band was cut -- at tau = 0.01 the peak pinned to 20.0 Hz for exactly this reason.
    A shift computed from an edge peak must not be reported.
    """
    freq = np.asarray(freq, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(freq[i]), float(y[i]), True
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    df = freq[i + 1] - freq[i]
    return float(freq[i] + delta * df), float(y1), False


# --------------------------------------------------------------------------- #
# Curves and metrics over a whole sweep
# --------------------------------------------------------------------------- #
def _series_psd(result, label, series_key):
    return result[series_key][label]['mean'], result[series_key][label].get('sem')


def _trials(result, label, series_key):
    """Per-trial PSDs for `label`, or None if the run did not store them."""
    key = series_key + '_psd_trials'
    if key not in result:
        return None
    return np.asarray(result[key])[:, result['labels'].index(label), :]


def index_curves(results, gamma, background_c, label, series_key='sde_nonlinear',
                 n_boot=0, seed=0):
    """Analytical and numerical contrast indices for every non-background contrast.

    Returns {contrast: {'freq', 'analytical', 'numerical', 'sem', 'sem_method'}}, sorted
    by contrast. Raises if the background condition is missing -- silently promoting the
    next contrast up would rescale every curve with no visible error.

    `n_boot > 0` uses the paired bootstrap when the per-trial spectra were saved; the
    delta method with rho = 0 is the fallback.
    """
    bg = results.get((gamma, background_c))
    if bg is None:
        raise KeyError(f"background condition (gamma={gamma}, c={background_c}) is missing "
                       f"from the payload; the contrast index is undefined without it")
    bg_mean, bg_sem = _series_psd(bg, label, series_key)
    bg_trials = _trials(bg, label, series_key) if n_boot else None

    curves = {}
    for (g, c), result in sorted(results.items(), key=lambda kv: kv[0][1]):
        if g != gamma or c == background_c:
            continue
        mean, sem = _series_psd(result, label, series_key)
        entry = {'freq': np.asarray(result['freq'], float),
                 'analytical': contrast_index(result['analytical'][label],
                                              bg['analytical'][label]),
                 'numerical': contrast_index(mean, bg_mean)}
        trials = _trials(result, label, series_key) if n_boot else None
        if trials is not None and bg_trials is not None:
            entry['sem'], was_paired = contrast_index_sem_bootstrap(
                trials, bg_trials, n_boot=n_boot, seed=seed)
            entry['sem_method'] = 'bootstrap_paired' if was_paired else 'bootstrap_unpaired'
        elif sem is not None and bg_sem is not None:
            entry['sem'] = contrast_index_sem(mean, sem, bg_mean, bg_sem)
            entry['sem_method'] = 'delta_rho0'
        else:
            entry['sem'] = None
            entry['sem_method'] = 'none'
        curves[c] = entry
    return curves


def default_bands(settings):
    """The three bands worth reporting, as {name: (lo, hi)}."""
    lo, hi = float(settings['min_freq']), float(settings['max_freq'])
    band = settings.get('report_band', [20.0, 100.0])
    return {'full': (lo, hi),
            'report': (float(band[0]), float(band[1])),
            'published': (lo, min(80.0, hi))}     # the window the paper actually plots


def index_metrics(results, settings, labels=DEFAULT_LABELS, series_key='sde_nonlinear',
                  background_c=None, bands=None, noise_params=None, n_boot=0):
    """Agreement between the analytical and numerical contrast indices.

    Returns {(gamma, contrast, label, band_name): entry}. Because the index is a
    difference, every discrepancy here is absolute.
    """
    c_vals = sorted(float(c) for c in settings['c_vals'])
    if background_c is None:
        background_c = c_vals[0]
    gammas = sorted({g for g, _ in results})
    bands = bands or default_bands(settings)

    metrics = {}
    for gamma in gammas:
        for label in labels:
            curves = index_curves(results, gamma, background_c, label, series_key,
                                  n_boot=n_boot)
            for contrast, cur in curves.items():
                f = cur['freq']
                for name, (lo, hi) in bands.items():
                    keep = (f >= lo) & (f <= hi)
                    if not keep.any():
                        continue
                    entry = _band_entry(f[keep], cur['analytical'][keep],
                                        cur['numerical'][keep],
                                        None if cur['sem'] is None else cur['sem'][keep])
                    entry['sem_method'] = cur['sem_method']
                    if noise_params is not None:
                        entry['lp_fraction_median'] = float(np.median(lp_fraction(
                            f[keep], results[(gamma, contrast)]['analytical'][label][keep],
                            noise_params)))
                    metrics[(gamma, contrast, label, name)] = entry
    return metrics


def _band_entry(f, ana, num, sem):
    diff = num - ana
    i_max = int(np.argmax(np.abs(diff)))
    f_ana, i_ana, edge_ana = peak(f, ana)
    f_num, i_num, edge_num = peak(f, num)
    entry = {
        'n_bins': int(f.size),
        'median_abs_diff': float(np.median(np.abs(diff))),
        'mean_signed_diff': float(np.mean(diff)),
        'rms_diff': float(np.sqrt(np.mean(diff ** 2))),
        'max_abs_diff': float(np.abs(diff[i_max])),
        'max_abs_diff_freq': float(f[i_max]),
        'band_mean_analytical': float(np.mean(ana)),
        'band_mean_numerical': float(np.mean(num)),
        'band_mean_diff': float(np.mean(num) - np.mean(ana)),
        'peak_freq_analytical': f_ana,
        'peak_index_analytical': i_ana,
        'peak_at_edge_analytical': bool(edge_ana),
        'peak_freq_numerical': f_num,
        'peak_index_numerical': i_num,
        'peak_at_edge_numerical': bool(edge_num),
        'peak_index_diff': float(i_num - i_ana),
        # A shift read off a monotone-in-band spectrum is an artefact of the band edge.
        'peak_freq_shift': float('nan') if (edge_ana or edge_num) else float(f_num - f_ana),
    }
    if sem is not None:
        good = sem > 0
        entry['sem_median'] = float(np.median(sem))
        entry['n_sigma_max'] = (float(np.max(np.abs(diff[good]) / sem[good]))
                                if good.any() else float('nan'))
    else:
        entry['sem_median'] = float('nan')
        entry['n_sigma_max'] = float('nan')
    return entry
