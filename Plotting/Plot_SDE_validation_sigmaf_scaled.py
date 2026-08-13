"""V1 PSD + V1-V4 coherence per sigma_f, with low_pass_add scaled to track sigma_f.

Usage:
    python Plotting/Plot_SDE_validation_sigmaf_scaled.py --out-dir DIR RUN1 [RUN2 ...]

Motivation. Reducing sigma_f (the private OU membrane noise) puts the full SDE back in the linear
regime where it matches the analytical PSD, but it also flattens the V1-V4 coherence, because
sigma_f is the *private* noise and low_pass_add is the *shared* noise -- lowering sigma_f alone
raises the shared/private ratio. To keep the curves' shape, scale the shared low_pass amplitude
`noise_sigma` by the SAME factor sigma_f was reduced. Reference sigma_f=0.01, noise_sigma=0.03, so
    noise_sigma_new = 0.03 * (sigma_f / 0.01) = 3 * sigma_f.

No new simulation is needed: low_pass_add is a deterministic closed-form additive term.
  * SDE PSD / coherence: rescaled from the saved auto-PSD and the saved complex cross-spectral
    block (`sde_nonlinear_csd`) -- subtract the base low_pass term, add the scaled one.
  * Analytical PSD / coherence: the payload stores only the real analytical coherence, not the
    complex cross-spectrum, so the analytical block is recomputed at the scaled noise_sigma via
    `analytical_spectra` (closed form, no Monte-Carlo). Its recurrent part is deterministic and
    matches the stored run; only the low_pass differs.

Each RUN is a results dir with Data/sde_validation.npy and its config_*.yaml. The c=0 background is
kept as the PSD normalisation divisor but not drawn. Reuses the plot helpers in
Plotting/Plot_SDE_validation_sigmaf.py.
"""

import os
import sys
import glob
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
matplotlib.use('Agg')
import numpy as np
import yaml
import torch

from Utils.Create_weight_matrices import setup_parameters
from Models.Model import RingModel
from Utils.Coherence import create_L_matrix, create_S_matrix
from Utils.SDE_simulation import (low_pass_matrix, coherence_from_block, analytical_spectra,
                                  default_record_indices)

# Sibling module (this file's own directory is on sys.path when run as a script).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Plot_SDE_validation_sigmaf as P

SIGMA_F_REF = 0.01       # reference private-noise amplitude
NOISE_SIGMA_REF = 0.03   # reference shared low_pass amplitude
N_GRID = 36              # the runs used setup_parameters(..., N=36)


def lp_term(freq, sigma, tau, rho, k):
    """One-sided additive low_pass term, exactly as added in run_condition (factor of 2)."""
    return 2.0 * low_pass_matrix(np.asarray(freq, dtype=float), sigma, tau, rho, k)


def load_config(run_dir):
    matches = sorted(glob.glob(os.path.join(run_dir, 'config_*.yaml')))
    if not matches:
        raise SystemExit(f"no config_*.yaml in {run_dir}")
    with open(matches[0]) as f:
        return yaml.safe_load(f)


def build_model(config, gamma):
    params = setup_parameters(config=config, N=N_GRID)
    params['gamma1'] = gamma
    return RingModel(params, simulate_firing_rates=True)


def analytical_at(model, contrast, freq, record_indices, pairs, noise_cfg, noise_sigma, t_span):
    """Recompute the analytical block at a given noise_sigma (closed form, no simulation)."""
    N = model.params['N']
    ic = np.ones(model.num_var * N) * 0.01
    J_aug_np, ss_aug = model.get_Jacobian_augmented(contrast, ic, 'RK45', list(t_span))
    J_aug = torch.tensor(J_aug_np, dtype=torch.float64)
    L = create_L_matrix(model, ss_aug, noise_cfg['delta_tau'] * model.params['tau'],
                        noise_cfg['noise_potential'], noise_cfg['noise_firing_rate'],
                        noise_cfg['GR_noise']).to(torch.float64)
    S = create_S_matrix(model).to(torch.float64)
    return analytical_spectra(J_aug, L, S, np.asarray(freq, dtype=float), record_indices,
                              pairs=pairs, noise_sigma=noise_sigma, noise_tau=noise_cfg['noise_tau'],
                              low_pass_add=True, rho=noise_cfg['rho'], verbose=False)


def process_run(run_dir):
    config = load_config(run_dir)
    payload = np.load(os.path.join(run_dir, 'Data', 'sde_validation.npy'),
                      allow_pickle=True).item()
    noise_cfg = payload['noise_params']
    base_sigma = float(noise_cfg['noise_sigma'])
    sigma_f = float(noise_cfg['sigma_f'])
    new_sigma = NOISE_SIGMA_REF * (sigma_f / SIGMA_F_REF)
    tau, rho = noise_cfg['noise_tau'], noise_cfg['rho']
    settings = payload['settings']
    t_span = settings.get('t_span', [0, 5])
    pairs = [tuple(p) for p in settings.get('coherence_pairs', [['y1', 'y4']])]
    record_indices = default_record_indices(N_GRID)
    results = payload['results']

    print(f"\n{run_dir}: sigma_f={sigma_f:g}  ->  noise_sigma {base_sigma:g} -> {new_sigma:g} "
          f"(low_pass power x {(new_sigma / base_sigma) ** 2:.3g})")

    models = {}  # one RingModel per gamma, reused across contrasts

    # --- verification (before overwriting anything) --------------------------- #
    (g0, c0), r0 = next(iter(results.items()))
    models[g0] = build_model(config, g0)
    chk = analytical_at(models[g0], c0, r0['freq'], record_indices, pairs, noise_cfg,
                        base_sigma, t_span)
    for lab in ('y1', 'y4'):
        rel = np.max(np.abs(chk['psd'][lab] - np.asarray(r0['analytical'][lab], float))
                     / np.abs(np.asarray(r0['analytical'][lab], float)))
        print(f"  rebuild check @ (g={g0:g}, c={c0:g}) {lab}: max rel diff vs stored "
              f"analytical = {rel:.2e}")

    # --- rescale + recompute -------------------------------------------------- #
    for (gamma, c), r in results.items():
        labels = list(r['labels'])
        k = len(labels)
        freq = r['freq']
        lp_base = lp_term(freq, base_sigma, tau, rho, k)
        lp_new = lp_term(freq, new_sigma, tau, rho, k)

        # SDE auto-PSD: swap the additive low_pass term (deterministic; SEM untouched)
        for lab in labels:
            i = labels.index(lab)
            m = np.asarray(r['sde_nonlinear'][lab]['mean'], dtype=float)
            r['sde_nonlinear'][lab]['mean'] = m - lp_base[:, i, i] + lp_new[:, i, i]

        # SDE coherence: rescale the saved complex cross-spectral block, then re-form coherence
        if 'sde_nonlinear_csd' in r:
            pooled_new = np.asarray(r['sde_nonlinear_csd']) - lp_base + lp_new
            r['sde_nonlinear_coherence'] = {
                pair: coherence_from_block(pooled_new, labels.index(pair[0]), labels.index(pair[1]))
                for pair in pairs}

        # Analytical PSD + coherence: recompute at the scaled noise_sigma
        if gamma not in models:
            models[gamma] = build_model(config, gamma)
        ana = analytical_at(models[gamma], c, freq, record_indices, pairs, noise_cfg,
                            new_sigma, t_span)
        r['analytical'] = ana['psd']
        r['analytical_coherence'] = ana['coherence']

    return payload, sigma_f, new_sigma


def main(run_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for run_dir in run_dirs:
        payload, sigma_f, new_sigma = process_run(run_dir)
        tag = P._sigma_tag(sigma_f)
        suffix = rf'  [$\sigma_{{lp}}={new_sigma:g}$]'
        for gamma in sorted({float(g) for (g, c) in payload['results']}):
            gtag = f"g{gamma:g}".replace('.', 'p')
            P.plot_psd(payload, gamma,
                       os.path.join(out_dir, f'sigmaf_{tag}_{gtag}_scaledLP_v1_psd.pdf'),
                       draw_background=False, title_suffix=suffix)
            P.plot_coherence(payload, gamma,
                             os.path.join(out_dir, f'sigmaf_{tag}_{gtag}_scaledLP_v1v4_coherence.pdf'),
                             draw_background=False, title_suffix=suffix)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Per-sigma_f PSD + coherence with low_pass_add scaled to track sigma_f')
    p.add_argument('run_dirs', nargs='+',
                   help='results dirs, each with Data/sde_validation.npy and config_*.yaml')
    p.add_argument('--out-dir', required=True, help='directory to write the figures into')
    args = p.parse_args()
    main(args.run_dirs, args.out_dir)
