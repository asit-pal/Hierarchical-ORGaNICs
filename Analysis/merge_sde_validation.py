"""Merge the per-contrast SDE validation shards and report the published comparison.

Usage:
    python Analysis/merge_sde_validation.py <results_dir> [--prefix sde_validation]
                                            [--out sde_validation.npy] [--n-boot 1000]

`Analysis/SDE_validation.py --c-index i` writes one shard per contrast so a SLURM array
can run them in parallel. This collects `Data/<prefix>_c*.npy` into the single
`Data/sde_validation.npy` the plotting scripts expect, and prints the comparison the
whole exercise is for: the analytical and simulated spectra under the *published*
normalisation, the contrast index `(P - P_bg)/(P + P_bg)`.

The consistency checks here are not defensive boilerplate. If one array task fails and
its shard is missing, `Plot_SDE_validation` would take the next contrast up as the
background and rescale every curve in the figure with no error message at all -- so a
missing shard has to be fatal here, at merge time, where it is still obvious.
"""

import os
import sys
import glob
import argparse
from itertools import product

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np

from Utils.SDE_contrast_index import (
    DEFAULT_LABELS, default_bands, index_metrics, lp_fraction,
)

# Settings a shard is *allowed* to differ on. `mode`/`paired`/`n_trials` vary by design:
# the c = 0 background keeps the noise-matched linear shadow (it is the only measurement
# of clipping bias in the condition every other curve divides by), and may be run with
# more trials. `n_jobs`/`linear_batch` only affect scheduling, and the per-trial seeds are
# `seed + trial` regardless. Everything else -- dt, T, burn_in, the Welch settings, the
# frequency band, low_pass_add, seed -- changes what is being compared, so a mismatch is
# fatal.
MAY_VARY = {'c_vals', 'mode', 'paired', 'n_trials', 'n_jobs', 'linear_batch',
            'save_traces', 'save_csd', 'save_psd_trials'}

LP_REPORT_FREQS = (1.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)

CLIP_WARN, CLIP_SEVERE = 1e-3, 1e-2


def load_shards(data_dir, prefix):
    paths = sorted(glob.glob(os.path.join(data_dir, f'{prefix}_c*.npy')))
    if not paths:
        raise SystemExit(f"No shards matching {prefix}_c*.npy in {data_dir}")
    shards = []
    for path in paths:
        payload = np.load(path, allow_pickle=True).item()
        payload['_path'] = path
        shards.append(payload)
    shards.sort(key=lambda p: (p.get('c_index') if p.get('c_index') is not None else -1))
    return shards


def check_consistency(shards, allow_missing=False):
    """Fail loudly on anything that would make the merged payload a lie."""
    problems = []
    ref = shards[0]

    for s in shards[1:]:
        if s['record_indices'] != ref['record_indices']:
            problems.append(f"{os.path.basename(s['_path'])}: record_indices differ")
        if s['noise_params'] != ref['noise_params']:
            problems.append(f"{os.path.basename(s['_path'])}: noise_params differ")
        if list(s.get('c_vals_full', [])) != list(ref.get('c_vals_full', [])):
            problems.append(f"{os.path.basename(s['_path'])}: c_vals_full differs "
                            f"({s.get('c_vals_full')} vs {ref.get('c_vals_full')})")
        for k, v in s['settings'].items():
            if k in MAY_VARY:
                continue
            if k in ref['settings'] and ref['settings'][k] != v:
                problems.append(f"{os.path.basename(s['_path'])}: settings['{k}'] = {v!r}, "
                                f"expected {ref['settings'][k]!r}")

    # One frequency grid, or the overlay figure is comparing different abscissae.
    ref_freq = None
    for s in shards:
        for key, result in s['results'].items():
            f = np.asarray(result['freq'])
            if ref_freq is None:
                ref_freq = f
            elif f.shape != ref_freq.shape or not np.allclose(f, ref_freq):
                problems.append(f"{os.path.basename(s['_path'])}: freq grid differs at {key}")

    seen = {}
    for s in shards:
        for key in s['results']:
            if key in seen:
                problems.append(f"duplicate condition {key} in "
                                f"{os.path.basename(seen[key])} and "
                                f"{os.path.basename(s['_path'])}")
            seen[key] = s['_path']

    c_full = list(ref.get('c_vals_full', ref['settings']['c_vals']))
    expected = set(product([float(g) for g in ref['settings']['gamma_vals']],
                           [float(c) for c in c_full]))
    missing = sorted(expected - {(float(g), float(c)) for g, c in seen})
    if missing:
        msg = f"missing conditions (gamma, c): {missing}"
        if allow_missing:
            print(f"  WARNING: {msg}")
        else:
            problems.append(msg + " -- the background would be silently substituted")

    if problems:
        raise SystemExit("Shards are not mergeable:\n  " + "\n  ".join(problems))
    return c_full


def merge(shards, c_full):
    results = {}
    for s in shards:
        results.update(s['results'])
    settings = dict(shards[0]['settings'])
    settings['c_vals'] = c_full
    # `mode`/`paired`/`n_trials` legitimately vary by shard; record what each one used
    # rather than pretending the first shard's value holds everywhere.
    settings['per_contrast'] = {
        float(c): {k: s['settings'][k] for k in ('mode', 'paired', 'n_trials')
                   if k in s['settings']}
        for s in shards for _, c in s['results']}
    return {'results': results,
            'settings': settings,
            'record_indices': shards[0]['record_indices'],
            'noise_params': shards[0]['noise_params'],
            'c_vals_full': c_full,
            'shards': [{'path': os.path.basename(s['_path']),
                        'c_index': s.get('c_index'),
                        'mtime': os.path.getmtime(s['_path'])} for s in shards]}


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def raw_table(results, labels, band_name='report'):
    """Absolute-scale agreement, straight out of the per-condition metrics.

    This is the pre-existing check (`mean_ratio` ~ 1). It is reprinted here with every
    contrast side by side, because a *trend* across contrast is what reveals where the
    linearisation starts to give -- something no single-contrast run could show.
    """
    key = 'metrics_subband' if band_name == 'report' else 'metrics'
    print(f"\n  absolute PSD agreement (sim / analytical), {band_name} band")
    print(f"    {'c':>7} {'trace':<7} {'source':<14} {'mean ratio':>11} {'band power':>11} "
          f"{'median |err|':>13}")
    for (gamma, c) in sorted(results, key=lambda k: (k[0], k[1])):
        m = results[(gamma, c)].get(key) or {}
        for lab in labels:
            if lab not in m:
                continue
            for tag in ('nonlinear', 'linear_paired', 'linear'):
                if f'mean_ratio_{tag}' not in m[lab]:
                    continue
                print(f"    {c:>7.3f} {lab:<7} {tag:<14} "
                      f"{m[lab][f'mean_ratio_{tag}']:>11.4f} "
                      f"{m[lab][f'band_power_ratio_{tag}']:>11.4f} "
                      f"{m[lab][f'median_rel_err_{tag}']:>13.4f}")


def clip_table(results, labels, background_c):
    """Rectification clipping per condition, plus the clip-bias bound at the background.

    At c = 0 the rectified variables sit essentially *on* their floor (y1Plus steady
    state ~ 9e-11 against a noise sd of ~3e-8), so a clip fraction of 0.1-0.5 there is
    expected, not a bug. What matters is whether that moves the y1/y4 spectra the figure
    actually plots -- and the noise-matched linear shadow, which is the same system on
    the same Wiener path with no clipping at all, measures exactly that.
    """
    print(f"\n  rectification clipping (background c = {background_c} is expected to be high)")
    print(f"    {'c':>7} {'clip frac':>11} {'slowest tau':>12} {'verdict':<10} "
          f"{'clip bias on y1/y4':>20}")
    report = {}
    for (gamma, c) in sorted(results, key=lambda k: (k[0], k[1])):
        r = results[(gamma, c)]
        frac = r.get('clip_fraction')
        if frac is None:
            continue
        if c == background_c:
            verdict = 'background'
        elif frac > CLIP_SEVERE:
            verdict = 'SEVERE'
        elif frac > CLIP_WARN:
            verdict = 'WARNING'
        else:
            verdict = 'ok'

        # |mean_ratio_nonlinear - mean_ratio_linear_paired|: the clipped and unclipped
        # runs of the same trajectories, so their difference is the clipping bias.
        m = r.get('metrics_subband') or {}
        bias = []
        for lab in labels:
            e = m.get(lab, {})
            if 'mean_ratio_nonlinear' in e and 'mean_ratio_linear_paired' in e:
                bias.append(abs(e['mean_ratio_nonlinear'] - e['mean_ratio_linear_paired']))
        bias_str = f"{max(bias):.2e}" if bias else "-- (unpaired)"
        report[(gamma, c)] = {'clip_fraction': frac, 'verdict': verdict,
                              'clip_bias': max(bias) if bias else None}
        print(f"    {c:>7.3f} {frac:>11.3e} {1e3 * r['slowest_tau']:>10.1f} ms "
              f"{verdict:<10} {bias_str:>20}")
    return report


def lp_table(results, labels, noise_params):
    """How much of the analytical PSD is the deterministic low_pass_add term.

    Where this is near 1, the analytical and simulated curves share a closed-form
    constant and their agreement says little. It is also why the contrast index is
    robust at the background: the shared term cancels out of the numerator.
    """
    print(f"\n  low_pass_add share of the analytical PSD (%), by frequency")
    header = ' '.join(f"{f:>7.0f}" for f in LP_REPORT_FREQS)
    print(f"    {'c':>7} {'trace':<7} {header}")
    report = {}
    for (gamma, c) in sorted(results, key=lambda k: (k[0], k[1])):
        r = results[(gamma, c)]
        f = np.asarray(r['freq'])
        for lab in labels:
            if lab not in r['analytical']:
                continue
            frac = lp_fraction(f, r['analytical'][lab], noise_params)
            picks = [frac[np.abs(f - target).argmin()] for target in LP_REPORT_FREQS]
            report[(gamma, c, lab)] = dict(zip(LP_REPORT_FREQS, [float(p) for p in picks]))
            print(f"    {c:>7.3f} {lab:<7} " + ' '.join(f"{100 * p:>7.1f}" for p in picks))
    return report


def index_table(metrics, background_c, band_name, labels):
    print(f"\n  contrast index (P-P_bg)/(P+P_bg) vs background c = {background_c}, "
          f"{band_name} band")
    print(f"    {'c':>7} {'trace':<7} {'med|dI|':>9} {'max|dI|':>9} {'@Hz':>7} "
          f"{'signed':>9} {'ana pk':>8} {'num pk':>8} {'shift':>7} {'nsig':>6} {'lp%':>6}")
    rows = sorted((k for k in metrics if k[3] == band_name and k[2] in labels),
                  key=lambda k: (k[1], k[2]))
    for key in rows:
        gamma, c, lab, _ = key
        e = metrics[key]
        shift = e['peak_freq_shift']
        shift_s = 'edge' if np.isnan(shift) else f"{shift:>7.2f}"
        lp = e.get('lp_fraction_median')
        lp_s = '--' if lp is None else f"{100 * lp:.0f}"
        print(f"    {c:>7.3f} {lab:<7} {e['median_abs_diff']:>9.4f} "
              f"{e['max_abs_diff']:>9.4f} {e['max_abs_diff_freq']:>7.1f} "
              f"{e['mean_signed_diff']:>9.4f} {e['peak_freq_analytical']:>8.2f} "
              f"{e['peak_freq_numerical']:>8.2f} {shift_s:>7} "
              f"{e['n_sigma_max']:>6.1f} {lp_s:>6}")
    # A sign shared by every contrast is a background artefact, not linearisation error:
    # all eight curves divide by the same P_bg realisation.
    for lab in labels:
        lab_rows = [k for k in rows if k[2] == lab]
        signed = np.array([metrics[k]['mean_signed_diff'] for k in lab_rows])
        sems = np.array([metrics[k]['sem_median'] for k in lab_rows])
        if signed.size < 3 or not np.all(np.sign(signed) == np.sign(signed[0])):
            continue
        # A shared sign alone is not evidence -- with eight contrasts it happens by
        # chance often enough, and near-zero offsets always have *some* sign. Only flag
        # it when the common offset is an appreciable fraction of the estimator noise.
        floor = 0.5 * np.nanmedian(sems)
        if not np.isfinite(floor) or abs(signed.mean()) <= floor:
            continue
        print(f"    NOTE: {lab} mean_signed_diff is {'positive' if signed[0] > 0 else 'negative'} "
              f"at every contrast, mean {signed.mean():+.4f} against a typical SEM of "
              f"{np.nanmedian(sems):.4f}. All curves divide by one P_bg realisation, so a "
              f"common offset points at the background, not the linearisation.")


def main(results_dir, prefix, out_name, n_boot, allow_missing, labels):
    data_dir = os.path.join(results_dir, 'Data')
    shards = load_shards(data_dir, prefix)
    print(f"Merging {len(shards)} shard(s) from {data_dir}")
    for s in shards:
        conds = sorted(c for _, c in s['results'])
        print(f"  {os.path.basename(s['_path']):<32} c_index={s.get('c_index')} "
              f"c={conds} mode={s['settings'].get('mode')} "
              f"paired={s['settings'].get('paired')} "
              f"n_trials={s['settings'].get('n_trials')}")

    c_full = check_consistency(shards, allow_missing=allow_missing)
    payload = merge(shards, c_full)
    results, settings = payload['results'], payload['settings']
    noise_params = payload['noise_params']
    background_c = float(min(c_full))
    recorded = next(iter(results.values()))['labels']
    labels = [l for l in labels if l in recorded]
    if not labels:
        raise SystemExit(f"None of the requested labels are in the payload ({recorded})")

    print(f"\nBackground contrast: {background_c}   "
          f"contrasts: {sorted(float(c) for c in c_full)}   "
          f"gammas: {settings['gamma_vals']}")

    raw_table(results, labels)
    clip_report = clip_table(results, labels, background_c)
    lp_report = lp_table(results, labels, noise_params)

    metrics = index_metrics(results, settings, labels=tuple(labels),
                            background_c=background_c, noise_params=noise_params,
                            n_boot=n_boot)
    for band_name in ('report', 'published'):
        index_table(metrics, background_c, band_name, labels)
    methods = {e['sem_method'] for e in metrics.values()}
    print(f"\n  index error bars: {', '.join(sorted(methods))}"
          f"{' (n_boot=%d)' % n_boot if n_boot else ''}")

    payload['index_metrics'] = metrics
    payload['clip_report'] = clip_report
    payload['lp_report'] = lp_report
    payload['bands'] = default_bands(settings)

    out_path = os.path.join(data_dir, out_name)
    np.save(out_path, payload)
    print(f"\nSaved merged payload to: {out_path}")
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Merge per-contrast SDE validation shards and report the comparison')
    p.add_argument('results_dir', help='Results directory containing Data/')
    p.add_argument('--prefix', default='sde_validation',
                   help='shard basename prefix (shards are <prefix>_cNN.npy)')
    p.add_argument('--out', dest='out_name', default='sde_validation.npy',
                   help='merged filename written into <results_dir>/Data')
    p.add_argument('--n-boot', dest='n_boot', type=int, default=1000,
                   help='paired bootstrap draws for the index SEM; 0 uses the delta method')
    p.add_argument('--allow-missing', action='store_true',
                   help='warn instead of failing when a condition has no shard')
    p.add_argument('--labels', nargs='+', default=list(DEFAULT_LABELS),
                   help='recorded channels to report (default: the plotted LFPs)')
    args = p.parse_args()
    main(args.results_dir, args.prefix, args.out_name, args.n_boot,
         args.allow_missing, args.labels)
