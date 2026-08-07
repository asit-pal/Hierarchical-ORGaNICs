# Essential imports
import os
import sys
import copy
import time
import argparse
import multiprocessing as mp

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import numpy as np
import torch

from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Coherence import create_S_matrix, create_L_matrix
from Utils.SDE_simulation import (
    SPLUS_BLOCKS, YPLUS_BLOCKS,
    analytical_psd, average_psd, default_record_indices, noise_gain,
    observation_noise, resolve_record_every, selftest_ou,
    simulate_linear_batch, simulate_paired_trial, welch_psd,
)

torch.set_default_dtype(torch.float64)

DEFAULTS = {
    'gamma_vals': [1.0],
    'c_vals': [0.5],
    't_span': [0, 5],          # window for the deterministic fixed-point solve
    'mode': 'both',            # 'linear' | 'nonlinear' | 'both'
    'integrator': 'exact',     # linear standalone run: 'exact' (Van Loan) or 'euler'
    'dt': 2.0e-5,              # SDE step (s) for the nonlinear (Euler) integration
    'dt_linear': 1.0e-4,       # step for the standalone exact linear run (unbiased at any dt)
    'T': 16.0,                 # simulated duration per trial after burn-in (s)
    'burn_in': 3.0,            # discarded transient (s); must exceed the slowest mode
    'n_trials': 32,
    'linear_batch': 8,         # trials advanced together in the linear run (memory)
    'record_fs': 20000.0,      # target sampling rate of the stored traces (Hz)
    'welch_nperseg_sec': 4.0,  # segment length (s); must outlast the slowest mode
    'welch_overlap': 0.5,
    'min_freq': 1.0,
    'max_freq': 200.0,
    'report_band': [20.0, 100.0],   # extra sub-band summarised in the report
    'seed': 0,
    'n_jobs': 16,
    'paired': True,            # drive the linear system with the nonlinear run's noise
    'clip_nonneg': True,
    'save_traces': True,       # keep a short trace snippet for the time-domain panel
}

# Filled in by the pool initializer so the big arrays are not re-pickled per task.
_CTX = {}

# BLAS threading env vars. Each worker's per-step work is one 864x864 matrix-vector
# product; letting every worker spin up a full thread pool oversubscribes the node
# badly. These must be set *before* numpy/torch are imported in the child, which is
# why the pool uses 'spawn' rather than 'fork'.
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def _init_worker(ctx):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _CTX.update(ctx)


def _run_trial(trial):
    """Pool worker: one nonlinear (optionally noise-matched) SDE trial."""
    c = _CTX
    t_start = time.time()
    out = simulate_paired_trial(
        model=c['model'], contrast=c['contrast'], ss_full=c['ss_full'],
        J_aug=c['J_aug'], B=c['B'], dt=c['dt'], n_steps=c['n_steps'],
        n_burn=c['n_burn'], record_idx=c['record_idx'],
        record_every=c['record_every'], seed=c['seed'] + trial,
        paired=c['paired'], clip_nonneg=c['clip_nonneg'],
    )
    res = {'n_clipped': out['n_clipped'], 'wall_seconds': time.time() - t_start}
    _, psd = welch_psd(out['nonlinear'], c['fs_rec'], c['nperseg'], c['overlap'])
    res['psd_nonlinear'] = psd
    if c['paired']:
        _, psd_lin = welch_psd(out['linear'], c['fs_rec'], c['nperseg'], c['overlap'])
        res['psd_linear'] = psd_lin
        # Trajectory-level linearisation error, same noise realisation.
        res['rms_nl'] = np.sqrt(np.mean(out['nonlinear'] ** 2, axis=-1))
        res['rms_lin'] = np.sqrt(np.mean(out['linear'] ** 2, axis=-1))
        res['rms_diff'] = np.sqrt(np.mean((out['nonlinear'] - out['linear']) ** 2, axis=-1))
    if trial == 0 and c['save_traces']:
        n_keep = min(out['nonlinear'].shape[-1], int(0.5 * c['fs_rec']))
        res['trace_nonlinear'] = out['nonlinear'][:, :n_keep]
        if c['paired']:
            res['trace_linear'] = out['linear'][:, :n_keep]
    return res


def resolve_settings(config, cli):
    """Merge DEFAULTS < config['SDE_validation'] < non-None CLI arguments."""
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in config.get('SDE_validation', {}).items()
                     if k != 'enabled'})
    settings.update({k: v for k, v in cli.items() if v is not None})
    return settings


def welch_frequency_grid(fs_rec, nperseg, min_freq, max_freq):
    """The Welch bin centres that fall inside [min_freq, max_freq]."""
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs_rec)
    return freqs[(freqs >= min_freq) & (freqs <= max_freq)]


def run_condition(model, contrast, gamma, settings, noise_cfg, record_indices, verbose=True):
    """Analytical vs. simulated PSD for one (gamma, contrast) condition."""
    N = model.params['N']
    initial_conditions = np.ones(model.num_var * N) * 0.01

    updated_params = copy.deepcopy(model.params)
    updated_params['gamma1'] = gamma
    cond_model = copy.deepcopy(model)
    cond_model.params = updated_params

    t0 = time.time()
    J_aug, ss_aug = cond_model.get_Jacobian_augmented(
        contrast, initial_conditions, 'RK45', settings['t_span'])
    J_aug = torch.tensor(J_aug, dtype=torch.float64)
    L = create_L_matrix(cond_model, ss_aug,
                        noise_cfg['delta_tau'] * cond_model.params['tau'],
                        noise_cfg['noise_potential'], noise_cfg['noise_firing_rate'],
                        noise_cfg['GR_noise']).to(torch.float64)
    S = create_S_matrix(cond_model).to(torch.float64)
    B = noise_gain(L, S)

    eig = np.linalg.eigvals(J_aug.numpy())
    slowest = -1.0 / eig.real.max()
    if verbose:
        print(f"  fixed point + Jacobian in {time.time() - t0:.1f}s; "
              f"slowest mode tau = {slowest * 1e3:.2f} ms, "
              f"fastest = {1e3 / np.abs(eig.real).max():.3f} ms", flush=True)

    # --- timing / sampling ------------------------------------------------- #
    dt = float(settings['dt'])
    record_every, fs_rec = resolve_record_every(dt, settings['record_fs'])
    n_steps = int(round(settings['T'] / dt))
    n_burn = int(round(settings['burn_in'] / dt))
    n_samples = n_steps // record_every
    # Clamp so the frequency grid we build here is the one welch actually uses.
    nperseg = int(min(round(settings['welch_nperseg_sec'] * fs_rec), n_samples))
    labels = list(record_indices.keys())
    idx = [record_indices[k] for k in labels]

    fastest_tau = 1.0 / np.abs(eig.real).max()
    if dt > 0.05 * fastest_tau:
        print(f"  WARNING: dt={dt:.2e}s is large relative to the fastest mode "
              f"({fastest_tau:.2e}s); Euler-Maruyama may be biased.", flush=True)
    # The simulation starts at the fixed point, i.e. off the stationary
    # distribution, so the burn-in has to outlast the slowest relaxation.
    if settings['burn_in'] < 5 * slowest:
        print(f"  WARNING: burn_in={settings['burn_in']}s is short for the slowest mode "
              f"({slowest * 1e3:.0f} ms); the estimate may retain a transient.", flush=True)
    if settings['welch_nperseg_sec'] < 3 * slowest:
        print(f"  WARNING: Welch segments ({settings['welch_nperseg_sec']}s) are short "
              f"relative to the slowest mode ({slowest * 1e3:.0f} ms); expect a low-frequency "
              f"bias below ~{1 / settings['welch_nperseg_sec']:.1f} Hz.", flush=True)

    freqs = welch_frequency_grid(fs_rec, nperseg, settings['min_freq'], settings['max_freq'])
    band = None  # filled after the first welch call

    # --- analytical reference ---------------------------------------------- #
    t0 = time.time()
    ana = analytical_psd(J_aug, L, S, freqs, record_indices,
                         noise_sigma=noise_cfg['noise_sigma'],
                         noise_tau=noise_cfg['noise_tau'],
                         low_pass_add=noise_cfg['low_pass_add'],
                         rho=noise_cfg['rho'], verbose=verbose)
    if verbose:
        print(f"  analytical PSD at {len(freqs)} frequencies in {time.time() - t0:.1f}s", flush=True)

    result = {'freq': freqs, 'labels': labels, 'analytical': ana,
              'fs_rec': fs_rec, 'dt': dt, 'nperseg': nperseg,
              'slowest_tau': slowest}

    rng = np.random.default_rng(settings['seed'] + 10007)

    def _add_observation_noise(traces):
        """Add the `low_pass_add` measurement noise to simulated traces."""
        if not noise_cfg['low_pass_add']:
            return traces
        return traces + observation_noise(traces.shape, 1.0 / fs_rec,
                                          noise_cfg['noise_sigma'], noise_cfg['noise_tau'],
                                          noise_cfg['rho'], rng)

    # --- standalone linear SDE (batched, unbiased) ------------------------- #
    if settings['mode'] in ('linear', 'both'):
        t0 = time.time()
        # The exact stepper is unbiased at any step size, so this run only needs
        # a step fine enough to sample at `record_fs` -- typically several times
        # coarser than the step the nonlinear Euler integration requires.
        dt_lin = float(settings['dt_linear']) if settings['dt_linear'] else 1.0 / fs_rec
        fs_lin = 1.0 / dt_lin
        n_steps_lin = int(round(settings['T'] / dt_lin))
        n_burn_lin = int(round(settings['burn_in'] / dt_lin))
        # Welch bin centres are k / nperseg_sec regardless of fs, so the linear run
        # may sample at its own rate as long as the segment spans the same duration.
        nperseg_lin = int(min(round(settings['welch_nperseg_sec'] * fs_lin), n_steps_lin))
        if fs_lin < 4 * settings['max_freq']:
            print(f"  WARNING: linear run samples at {fs_lin:.0f} Hz, close to the "
                  f"{settings['max_freq']:.0f} Hz analysis limit; aliasing may bias it.",
                  flush=True)

        rng_lin = np.random.default_rng(settings['seed'])
        psd_chunks = []
        w_freqs = None
        for start in range(0, settings['n_trials'], int(settings['linear_batch'])):
            n_this = min(int(settings['linear_batch']), settings['n_trials'] - start)
            traces = simulate_linear_batch(
                J_aug, B, dt_lin, n_steps_lin, n_burn_lin, n_this, idx,
                1, rng_lin, integrator=settings['integrator'])
            traces = _add_observation_noise(traces)
            w_freqs, psd = welch_psd(traces, fs_lin, nperseg_lin, settings['welch_overlap'])
            band_lin = (w_freqs >= settings['min_freq']) & (w_freqs <= settings['max_freq'])
            if not np.allclose(w_freqs[band_lin], freqs):
                raise ValueError("linear-run frequency grid does not match the analytical "
                                 "grid; welch_nperseg_sec * fs must be an integer for both")
            psd_chunks.append(psd[..., band_lin])
            if start == 0 and settings['save_traces']:
                result['trace_linear_batch'] = traces[0, :, :int(min(traces.shape[-1],
                                                                    0.5 * fs_lin))]
            del traces
        mean, sem = average_psd(np.concatenate(psd_chunks, axis=0))
        result['sde_linear'] = {lab: {'mean': mean[i], 'sem': sem[i]}
                                for i, lab in enumerate(labels)}
        result['sde_linear_meta'] = {'integrator': settings['integrator'],
                                     'n_trials': settings['n_trials'], 'dt': dt_lin}
        if verbose:
            print(f"  linear SDE ({settings['integrator']}, {settings['n_trials']} trials, "
                  f"dt={dt_lin:.1e}s) in {time.time() - t0:.1f}s", flush=True)

    # --- nonlinear SDE, optionally noise-matched --------------------------- #
    if settings['mode'] in ('nonlinear', 'both'):
        t0 = time.time()
        ctx = {'model': cond_model, 'contrast': contrast, 'ss_full': np.asarray(ss_aug).flatten(),
               'J_aug': J_aug.numpy(), 'B': B, 'dt': dt, 'n_steps': n_steps, 'n_burn': n_burn,
               'record_idx': idx, 'record_every': record_every, 'seed': settings['seed'],
               'paired': settings['paired'], 'clip_nonneg': settings['clip_nonneg'],
               'fs_rec': fs_rec, 'nperseg': nperseg, 'overlap': settings['welch_overlap'],
               'save_traces': settings['save_traces']}

        n_jobs = max(1, min(int(settings['n_jobs']), settings['n_trials']))
        trials = list(range(settings['n_trials']))
        if n_jobs == 1:
            _init_worker(ctx)
            out = [_run_trial(t) for t in trials]
        else:
            saved = {k: os.environ.get(k) for k in _THREAD_ENV}
            os.environ.update({k: '1' for k in _THREAD_ENV})
            try:
                # 'spawn' so the children import numpy with the single-thread env
                # above in force; under 'fork' the vars would arrive too late.
                with mp.get_context('spawn').Pool(n_jobs, initializer=_init_worker,
                                                  initargs=(ctx,)) as pool:
                    out = pool.map(_run_trial, trials)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        if band is None:
            w_freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs_rec)
            band = (w_freqs >= settings['min_freq']) & (w_freqs <= settings['max_freq'])

        psd_nl = np.stack([o['psd_nonlinear'][..., band] for o in out])
        mean, sem = average_psd(psd_nl)
        result['sde_nonlinear'] = {lab: {'mean': mean[i], 'sem': sem[i]}
                                   for i, lab in enumerate(labels)}
        n_clipped = int(sum(o['n_clipped'] for o in out))
        result['n_clipped'] = n_clipped
        # y1Plus, y4Plus, s1Plus, s4Plus are the four clippable blocks (N entries each)
        n_clippable = (len(YPLUS_BLOCKS) + len(SPLUS_BLOCKS)) * N
        total_updates = settings['n_trials'] * (n_steps + n_burn) * n_clippable
        result['clip_fraction'] = n_clipped / max(1, total_updates)

        if settings['paired']:
            psd_lin = np.stack([o['psd_linear'][..., band] for o in out])
            mean_l, sem_l = average_psd(psd_lin)
            result['sde_linear_paired'] = {lab: {'mean': mean_l[i], 'sem': sem_l[i]}
                                           for i, lab in enumerate(labels)}
            rms_nl = np.mean([o['rms_nl'] for o in out], axis=0)
            rms_lin = np.mean([o['rms_lin'] for o in out], axis=0)
            rms_diff = np.mean([o['rms_diff'] for o in out], axis=0)
            result['paired_error'] = {
                lab: {'rms_nonlinear': float(rms_nl[i]), 'rms_linear': float(rms_lin[i]),
                      'rms_diff': float(rms_diff[i]),
                      'relative': float(rms_diff[i] / rms_nl[i]) if rms_nl[i] > 0 else np.nan}
                for i, lab in enumerate(labels)}

        if settings['save_traces']:
            for key in ('trace_nonlinear', 'trace_linear'):
                if key in out[0]:
                    result[key] = out[0][key]
        if verbose:
            per_trial = np.array([o['wall_seconds'] for o in out])
            us_per_step = 1e6 * per_trial.mean() / (n_steps + n_burn)
            print(f"  nonlinear SDE ({settings['n_trials']} trials, {n_jobs} workers) "
                  f"in {time.time() - t0:.1f}s "
                  f"({per_trial.mean():.1f}s/trial, {us_per_step:.0f} us/step); "
                  f"clipped {n_clipped} state updates "
                  f"({100 * result['clip_fraction']:.2e}% of all updates)", flush=True)

    result['metrics'] = compute_metrics(result, labels)
    result['metrics_subband'] = compute_metrics(result, labels, settings['report_band'])
    result['report_band'] = settings['report_band']
    return result


def compute_metrics(result, labels, sub_band=None):
    """Per-trace agreement between each simulation and the analytical curve.

    `sub_band` restricts the comparison to [lo, hi] Hz. The full band is
    dominated by the low-frequency peak, where Welch resolves the slowest mode
    poorly; the sub-band isolates the gamma range the model is actually about.
    """
    freqs = result['freq']
    keep = np.ones(freqs.shape, dtype=bool) if sub_band is None else \
        (freqs >= sub_band[0]) & (freqs <= sub_band[1])
    if not keep.any():
        return {}
    f = freqs[keep]
    df = np.gradient(f)
    metrics = {}
    for i, lab in enumerate(labels):
        ref = result['analytical'][lab][keep]
        entry = {'peak_freq_analytical': float(f[np.argmax(ref)]),
                 'n_bins': int(keep.sum())}
        for key, tag in (('sde_linear', 'linear'),
                         ('sde_linear_paired', 'linear_paired'),
                         ('sde_nonlinear', 'nonlinear')):
            if key not in result:
                continue
            sim = result[key][lab]['mean'][keep]
            ratio = sim / ref
            entry[f'mean_ratio_{tag}'] = float(np.mean(ratio))
            entry[f'median_ratio_{tag}'] = float(np.median(ratio))
            entry[f'median_rel_err_{tag}'] = float(np.median(np.abs(ratio - 1.0)))
            entry[f'max_rel_err_{tag}'] = float(np.max(np.abs(ratio - 1.0)))
            entry[f'peak_freq_{tag}'] = float(f[np.argmax(sim)])
            # Total power in band -- insensitive to per-bin estimator scatter.
            entry[f'band_power_ratio_{tag}'] = float(np.sum(sim * df) / np.sum(ref * df))
        metrics[lab] = entry
    return metrics


def main(config_file, cli):
    print(f"SDE validation of the analytical power spectra")
    print(f"Attempting to load config from: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
        print(f"Absolute path: {os.path.abspath(config_file)}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    settings = resolve_settings(config, cli)
    results_dir = os.path.dirname(os.path.abspath(config_file))
    data_dir = os.path.join(results_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    print("Settings:")
    for k in sorted(settings):
        print(f"  {k}: {settings[k]}")

    # Sanity-check the PSD normalisation before trusting anything downstream.
    ratio, _ = selftest_ou(dt=1e-4, T=100.0, verbose=True)
    if not (0.95 < ratio < 1.05):
        raise RuntimeError(f"OU self-test failed (mean PSD ratio {ratio:.3f}); "
                           "the welch/analytical conventions do not match.")

    params = setup_parameters(config=config, N=36)
    model = RingModel(params, simulate_firing_rates=True)
    record_indices = default_record_indices(params['N'])
    noise_cfg = config['noise_params']

    results = {}
    for contrast in settings['c_vals']:
        for gamma in settings['gamma_vals']:
            print(f"\n--- contrast = {contrast}, gamma1 = {gamma} ---", flush=True)
            results[(gamma, contrast)] = run_condition(
                model, contrast, gamma, settings, noise_cfg, record_indices)
            report(results[(gamma, contrast)])

    payload = {'results': results,
               'settings': settings,
               'record_indices': record_indices,
               'noise_params': noise_cfg}
    filepath = os.path.join(data_dir, 'sde_validation.npy')
    np.save(filepath, payload)
    print(f"\nSaved SDE validation data to: {filepath}")


def _agreement_table(metrics, title):
    print(f"\n  agreement with the analytical PSD, {title}:")
    print(f"    {'trace':<8} {'source':<14} {'mean ratio':>11} {'band power':>11} "
          f"{'median |err|':>13} {'peak Hz':>9}")
    for lab, m in metrics.items():
        for tag in ('linear', 'linear_paired', 'nonlinear'):
            if f'mean_ratio_{tag}' not in m:
                continue
            print(f"    {lab:<8} {tag:<14} {m[f'mean_ratio_{tag}']:>11.4f} "
                  f"{m[f'band_power_ratio_{tag}']:>11.4f} "
                  f"{m[f'median_rel_err_{tag}']:>13.4f} "
                  f"{m[f'peak_freq_{tag}']:>9.1f}")
        print(f"    {'':<8} {'analytical':<14} {'':>11} {'':>11} {'':>13} "
              f"{m['peak_freq_analytical']:>9.1f}")


def report(result):
    """Print the agreement tables for one condition."""
    _agreement_table(result['metrics'],
                     f"{result['freq'][0]:.0f}-{result['freq'][-1]:.0f} Hz (full band)")
    if result.get('metrics_subband'):
        lo, hi = result['report_band']
        _agreement_table(result['metrics_subband'], f"{lo:.0f}-{hi:.0f} Hz")
    if 'paired_error' in result:
        print("\n  noise-matched trajectory error (nonlinear vs. linear, same Wiener path):")
        for lab, e in result['paired_error'].items():
            print(f"    {lab:<8} rms(nonlin)={e['rms_nonlinear']:.3e}  "
                  f"rms(nonlin - lin)={e['rms_diff']:.3e}  "
                  f"relative={100 * e['relative']:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='Validate the analytical power spectra against direct SDE simulation')
    p.add_argument('config_file', help='Path to config file')
    p.add_argument('--mode', choices=['linear', 'nonlinear', 'both'], default=None)
    p.add_argument('--integrator', choices=['exact', 'euler'], default=None)
    p.add_argument('--dt', type=float, default=None)
    p.add_argument('--T', type=float, default=None, help='simulated seconds per trial')
    p.add_argument('--burn-in', dest='burn_in', type=float, default=None)
    p.add_argument('--n-trials', dest='n_trials', type=int, default=None)
    p.add_argument('--n-jobs', dest='n_jobs', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--min-freq', dest='min_freq', type=float, default=None)
    p.add_argument('--max-freq', dest='max_freq', type=float, default=None)
    p.add_argument('--record-fs', dest='record_fs', type=float, default=None)
    p.add_argument('--dt-linear', dest='dt_linear', type=float, default=None)
    p.add_argument('--nperseg-sec', dest='welch_nperseg_sec', type=float, default=None)
    p.add_argument('--no-paired', dest='paired', action='store_false', default=None,
                   help='skip the noise-matched linear run inside the nonlinear trials')
    p.add_argument('--c-vals', dest='c_vals', type=float, nargs='+', default=None)
    p.add_argument('--gamma-vals', dest='gamma_vals', type=float, nargs='+', default=None)
    args = p.parse_args()

    cli = {k: v for k, v in vars(args).items() if k != 'config_file'}
    main(args.config_file, cli)
