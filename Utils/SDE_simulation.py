"""Stochastic (SDE) simulation of the ORGaNICs network.

This module exists to *validate* the analytical power spectra produced by
`Utils.matrix_spectrum.matrix_solution`. That analytical result rests on two
approximations:

  (A1) the dynamics are linearised about the deterministic fixed point, and
  (A2) the stationary spectrum of the resulting linear SDE is
         S(w) = (J + i w I)^-1  L D L^T  (J - i w I)^-T .

Two simulations are provided so the two approximations can be separated:

  `simulate_linear_batch`   integrates the *linearised* augmented system
                            dX = J_aug X dt + L S dW.
                            Disagreement here means (A2) / the numerics are
                            wrong; it says nothing about (A1).

  `simulate_paired_trial`   integrates the *full nonlinear* model with the same
                            noise sources injected, and (optionally) the linear
                            system driven by the *identical* Wiener increments.
                            Disagreement here, when the linear check passes,
                            isolates the linearisation error (A1).

Noise model
-----------
The augmented state is [x; f] where x is the 16N-dimensional reduced state
(y1, y1Plus, y4, y4Plus, u1, ..., s4Plus) and f is an 8N-dimensional bank of
Ornstein-Uhlenbeck processes (one per membrane potential). The SDE is

    dx = F(x) dt + A f dt + B_x dW_x        A = I/tau_x on the membrane rows
    df = -f/tau_f dt   + B_f dW_f

with B = L @ S exactly as assembled by `Utils.Coherence.create_L_matrix` and
`Utils.Coherence.create_S_matrix`, so that B B^T = L D L^T is the noise
intensity used by `matrix_solution`.

PSD conventions
---------------
`matrix_solution.auto_spectrum` returns the two-sided PSD in the angular
frequency convention,

    S(w) = int C(tau) e^{-i w tau} dtau ,    var(x) = int S(w) dw / (2 pi) .

`scipy.signal.welch` returns the one-sided PSD in ordinary frequency,

    var(x) = int_0^{fs/2} P(f) df .

The two are related by  P(f) = 2 * S(2 pi f)  for f > 0. `to_onesided` applies
that factor so the analytical and Welch curves can be overlaid directly.
`selftest_ou()` verifies the factor numerically on a scalar OU process.
"""

import numpy as np
import torch
from scipy.linalg import expm
from scipy.signal import csd, welch

from Utils.matrix_spectrum import matrix_solution, noise_power_spectrum

# Block layout of the reduced (Jacobian) state vector, in units of N:
#   0 y1     1 y1Plus   2 y4     3 y4Plus
#   4 u1     5 u1Plus   6 u4     7 u4Plus
#   8 p1     9 p1Plus  10 p4    11 p4Plus
#  12 s1    13 s1Plus  14 s4    15 s4Plus
MEMBRANE_BLOCKS = (0, 2, 4, 6, 8, 10, 12, 14)  # driven by the filtered f noise
YPLUS_BLOCKS = (1, 3)                          # appear inside sqrt() in dy
SPLUS_BLOCKS = (13, 15)                        # appear in a denominator in dp

# Coarse stride used to walk off the transient with the exact stepper. Kept well
# below the slowest mode but far above dt; a longer stride would push the matrix
# exponential of the stiff Van Loan block into the regime where scaling-and-
# squaring loses accuracy.
BURN_STRIDE_SECONDS = 0.02


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def as_numpy(x):
    """Return a float64 numpy view of a torch tensor or array-like."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def noise_gain(L, S):
    """Return B = L @ S, the matrix multiplying the Wiener increments.

    Also checks that B B^T reproduces the noise intensity L D L^T that
    `matrix_solution` builds internally (D = S**2, elementwise). The two agree
    only because `create_S_matrix` returns a diagonal S; the assertion makes
    that assumption explicit rather than silent.
    """
    L = as_numpy(L)
    S = as_numpy(S)
    B = L @ S
    Q_matrix_solution = L @ (S ** 2) @ L.T
    if not np.allclose(B @ B.T, Q_matrix_solution, atol=1e-12, rtol=1e-8):
        raise ValueError(
            "B B^T != L D L^T -- create_S_matrix is no longer diagonal, so the "
            "SDE noise gain and the analytical noise intensity have diverged."
        )
    return B


def default_record_indices(N):
    """Indices (into the reduced state) of the traces we compare by default.

    `y1` and `y4` are the same centre-neuron membrane potentials that
    `Analysis/Power_spectra_analysis.py` feeds to `auto_spectrum`.
    """
    return {
        'y1':     0 * N + N // 2,
        'y1Plus': 1 * N + N // 2,
        'y4':     2 * N + N // 2,
        'y4Plus': 3 * N + N // 2,
    }


def to_onesided(S_two_sided_angular):
    """Convert a two-sided angular-frequency PSD to scipy-welch convention."""
    return 2.0 * np.asarray(S_two_sided_angular)


def resolve_record_every(dt, target_fs):
    """Largest decimation factor whose output rate is still >= `target_fs`."""
    every = max(1, int(np.floor((1.0 / dt) / float(target_fs))))
    return every, 1.0 / (dt * every)


# --------------------------------------------------------------------------- #
# the additive `low_pass_add` measurement term
# --------------------------------------------------------------------------- #
def low_pass_matrix(freqs, sigma, tau, rho, k):
    """Two-sided additive measurement term implied by `low_pass_add`.

    `matrix_spectrum.spectral_matrix` adds

        ones(n, n) * P(w)  +  eye(n) * rho * P(w) ,
        P(w) = sigma^2 / (1 + (tau w)^2)^2

    to the spectral matrix: a *shared* low-pass process seen by every channel --
    which is what creates cross-channel power, and hence coherence -- plus an
    independent per-channel one scaled by rho. Note the off-diagonal entries get
    P, not P*(1 + rho).

    The term is deterministic and known in closed form, so the analytical and the
    simulated spectra both take it from here rather than the simulation paying
    estimator noise for a quantity that has an exact expression. The two call
    sites differ only by the one-sided factor:

        analytical:  block += low_pass_matrix(...)      then to_onesided(...)
        numerical:   C     += 2 * low_pass_matrix(...)  (welch is already one-sided)

    Returns:
        (n_freq, k, k) real array.
    """
    om = 2 * np.pi * np.asarray(freqs, dtype=np.float64)
    P = noise_power_spectrum(om, sigma, tau)
    shape = np.ones((k, k)) + rho * np.eye(k)
    return P[:, None, None] * shape[None, :, :]


def coherence_from_block(block, i, j):
    """Magnitude-squared coherence |Sij|^2 / (Sii Sjj) from a spectral block.

    The one-sided factor of 2 cancels between numerator and denominator, so this
    is convention-free: it gives the same answer for the two-sided analytical
    block and for the one-sided welch estimate. `block` is (..., n_freq, k, k).
    """
    Sij = block[..., i, j]
    Sii = np.real(block[..., i, i])
    Sjj = np.real(block[..., j, j])
    return np.abs(Sij) ** 2 / (Sii * Sjj)


# --------------------------------------------------------------------------- #
# analytical reference
# --------------------------------------------------------------------------- #
def _spectral_block_reference(mat, freqs, indices, chunk=8):
    """The (k, k) sub-block of `matrix_solution.spectral_matrix`, chunked over freq.

    This is the untouched reference implementation. It materialises an
    (n_freq, n, n) complex array (~6 GB at n=864 for 500 frequencies) and
    inverts two n x n complex matrices per frequency, so it is only practical
    for spot checks -- `analytical_spectra` uses it to validate the fast path.
    """
    idx = list(indices.values())
    k = len(idx)
    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.empty((freqs.size, k, k), dtype=np.complex128)
    for start in range(0, freqs.size, chunk):
        stop = min(start + chunk, freqs.size)
        spec = mat.spectral_matrix(freq=torch.as_tensor(freqs[start:stop]))
        out[start:stop] = spec[:, idx][:, :, idx].cpu().numpy()
        del spec
    return out


def _spectral_block_fast(mat, freqs, indices):
    """Same sub-block, without forming the full spectral matrix.

    `spectral_matrix` computes  S(w) = M^-1 Q M^-H  with  M = J + i w I. Only a
    few entries are needed, and for real J

        S_kl = e_k^T M^-1 Q M^-H e_l = a_k^T Q conj(a_l),   M^T a_k = e_k ,

    because M^-H e_l = conj(M^-T e_l). So one LU factorisation of M^T per
    frequency plus one back-substitution per requested index replaces two full
    n x n complex inversions -- roughly an order of magnitude less work, and
    O(n k) instead of O(n_freq * n^2) memory. The same factorisation yields the
    off-diagonal entries for free, which is what makes coherence cheap.
    """
    n = mat.N
    Q = mat.noise_mat
    eye = torch.eye(n, dtype=torch.cdouble)
    idx = list(indices.values())
    k = len(idx)
    E = torch.zeros((n, k), dtype=torch.cdouble)
    for col, i in enumerate(idx):
        E[i, col] = 1.0
    Jc = mat.J.to(torch.cdouble)

    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.empty((freqs.size, k, k), dtype=np.complex128)
    with torch.no_grad():
        for f_i, f in enumerate(freqs):
            om = 2 * np.pi * f
            A = torch.linalg.solve((Jc + 1j * om * eye).transpose(0, 1), E)  # (n, k)
            blk = torch.einsum('ak,ab,bl->kl', A, Q, torch.conj(A))
            # Q is Hermitian, so the block is too; symmetrise to kill round-off.
            blk = 0.5 * (blk + blk.conj().transpose(0, 1))
            out[f_i] = blk.cpu().numpy()

    if mat.low_pass_add:
        out = out + low_pass_matrix(freqs, mat.noise_sigma, mat.noise_tau, mat.rho, k)
    return out


def analytical_spectra(J, L, S, freqs, indices, pairs=(), noise_sigma=None,
                       noise_tau=None, low_pass_add=False, rho=None,
                       n_check=3, verbose=False):
    """Analytical spectra for the recorded channels, in welch (one-sided) units.

    Values come from the fast path, cross-checked against the untouched
    `matrix_solution` at `n_check` frequencies spread over the grid -- so what
    gets compared against the SDE is the quantity the published pipeline
    produces, just computed without the 6 GB intermediate.

    Args:
        J, L, S: augmented Jacobian and noise matrices (torch tensors).
        freqs: 1-D array of frequencies in Hz.
        indices: dict {label: reduced-state index}.
        pairs: iterable of (label_a, label_b) for which to return coherence.
    Returns:
        dict with
          'block'     : (n_freq, k, k) complex one-sided cross-spectral matrix
          'psd'       : {label: one-sided auto-spectrum}
          'coherence' : {(a, b): magnitude-squared coherence}
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    mat = matrix_solution(J, L, S, noise_sigma, noise_tau,
                          low_pass_add=low_pass_add, rho=rho)
    labels = list(indices.keys())
    fast = _spectral_block_fast(mat, freqs, indices)

    if n_check:
        probe = np.unique(np.linspace(0, freqs.size - 1, n_check).astype(int))
        ref = _spectral_block_reference(mat, freqs[probe], indices)
        # Block-relative error, not per-entry: off-diagonal entries pass through
        # near-cancellations where a per-entry ratio blows up spuriously.
        scale = np.maximum(np.abs(ref).max(axis=(1, 2)), 1e-300)
        rel = np.abs(fast[probe] - ref).max(axis=(1, 2)) / scale
        if rel.max() > 1e-6:
            raise ValueError("fast analytical spectral block disagrees with "
                             f"matrix_solution (max relative difference {rel.max():.3e})")
        herm = np.abs(fast - np.conj(np.swapaxes(fast, 1, 2))).max()
        if herm > 1e-10 * np.abs(fast).max():
            raise ValueError(f"analytical spectral block is not Hermitian ({herm:.3e})")
        if verbose:
            print(f"  [analytical] block matches matrix_solution to {rel.max():.2e} "
                  f"at {probe.size} probe frequencies")

    block = to_onesided(fast)
    psd = {lab: np.real(block[:, i, i]).copy() for i, lab in enumerate(labels)}

    coh = {}
    for a, b in pairs:
        ia, ib = labels.index(a), labels.index(b)
        coh[(a, b)] = coherence_from_block(block, ia, ib)
        if n_check:
            # Compare against the *published* entry point, so the validated
            # object is literally matrix_solution.coherence's output.
            probe = np.unique(np.linspace(0, freqs.size - 1, n_check).astype(int))
            ref_coh, _ = mat.coherence(i=indices[a], j=indices[b],
                                       freq=torch.as_tensor(freqs[probe]))
            ref_coh = np.abs(ref_coh.cpu().numpy())   # Utils/Coherence.py takes abs()
            diff = np.abs(coh[(a, b)][probe] - ref_coh).max()
            if diff > 1e-8:
                raise ValueError(f"coherence({a},{b}) disagrees with "
                                 f"matrix_solution.coherence (max abs diff {diff:.3e})")
            if verbose:
                print(f"  [analytical] coherence({a},{b}) matches "
                      f"matrix_solution.coherence to {diff:.2e}")

    return {'block': block, 'psd': psd, 'coherence': coh}


def analytical_psd(J, L, S, freqs, indices, noise_sigma=None, noise_tau=None,
                   low_pass_add=False, rho=None, n_check=3, verbose=False):
    """One-sided analytical PSD (welch convention) at `freqs`, for `indices`.

    Thin wrapper over `analytical_spectra` for callers that only want
    auto-spectra and shouldn't have to know about the cross-spectral block.
    """
    return analytical_spectra(J, L, S, freqs, indices, pairs=(),
                              noise_sigma=noise_sigma, noise_tau=noise_tau,
                              low_pass_add=low_pass_add, rho=rho,
                              n_check=n_check, verbose=verbose)['psd']


# --------------------------------------------------------------------------- #
# linear SDE
# --------------------------------------------------------------------------- #
def _exact_discretization(J, Q, dt):
    """Van Loan (1978) exact discretisation of dX = J X dt + dW, cov(dW)=Q dt.

    Returns (A, G) with A = expm(J dt) and G G^T = the exact one-step noise
    covariance, so that X_{k+1} = A X_k + G z_k (z ~ N(0, I)) samples the
    continuous process *exactly* at spacing dt -- no Euler bias.
    """
    n = J.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -J
    C[:n, n:] = Q
    C[n:, n:] = J.T
    E = expm(C * dt)
    A = E[n:, n:].T
    Sigma = A @ E[:n, n:]
    Sigma = 0.5 * (Sigma + Sigma.T)
    evals, evecs = np.linalg.eigh(Sigma)
    # Q is rank-deficient whenever a noise std is zero (e.g. noise_potential=0),
    # so tiny negative eigenvalues are round-off, not a modelling error.
    neg = evals[evals < 0]
    if neg.size and np.abs(neg).max() > 1e-10 * max(1.0, evals.max()):
        raise ValueError(f"one-step noise covariance is not PSD (min eig {evals.min():.3e})")
    G = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))
    return A, G


def simulate_linear_batch(J, B, dt, n_steps, n_burn, n_trials, record_idx,
                          record_every, rng, integrator='exact'):
    """Integrate dX = J X dt + B dW for `n_trials` independent realisations.

    All trials are advanced together as columns of a matrix, which turns the
    per-step work into a single GEMM.

    Args:
        J: (n, n) augmented Jacobian.
        B: (n, n) noise gain, so the intensity is Q = B B^T.
        integrator: 'exact' (Van Loan, unbiased at any dt) or 'euler'
            (Euler-Maruyama, biased at O(dt) -- kept so the same stepper can be
            used for the noise-matched comparison against the nonlinear model).
    Returns:
        (n_trials, len(record_idx), n_samples) array of recorded deviations.
    """
    J = as_numpy(J)
    B = as_numpy(B)
    n = J.shape[0]
    record_idx = np.asarray(record_idx, dtype=int)

    Q = B @ B.T
    if integrator == 'exact':
        A, G = _exact_discretization(J, Q, dt)
        diag_noise = None
    elif integrator == 'euler':
        A = np.eye(n) + J * dt
        diag_noise = np.diag(B) * np.sqrt(dt) if _is_diagonal(B) else None
        G = None if diag_noise is not None else B * np.sqrt(dt)
    else:
        raise ValueError(f"unknown integrator '{integrator}'")

    X = np.zeros((n, n_trials))

    # Burn-in. The exact stepper is unbiased at any step size, so the transient
    # can be walked off in coarse strides (one extra matrix exponential) instead
    # of ~1e5 fine steps. Euler has to take the same dt it will use afterwards.
    if n_burn > 0 and integrator == 'exact':
        dt_burn = min(BURN_STRIDE_SECONDS, n_burn * dt)
        n_chunk = max(1, int(round(n_burn * dt / dt_burn)))
        A_b, G_b = _exact_discretization(J, Q, n_burn * dt / n_chunk)
        for _ in range(n_chunk):
            X = A_b @ X + G_b @ rng.standard_normal((n, n_trials))
    else:
        for _ in range(n_burn):
            z = rng.standard_normal((n, n_trials))
            X = A @ X + (diag_noise[:, None] * z if diag_noise is not None else G @ z)

    n_samples = n_steps // record_every
    rec = np.empty((len(record_idx), n_samples, n_trials))
    for j in range(n_steps):
        z = rng.standard_normal((n, n_trials))
        X = A @ X + (diag_noise[:, None] * z if diag_noise is not None else G @ z)
        if j % record_every == 0 and j // record_every < n_samples:
            rec[:, j // record_every, :] = X[record_idx, :]

    if not np.all(np.isfinite(rec)):
        raise FloatingPointError("linear SDE diverged -- reduce dt or use integrator='exact'")
    return np.transpose(rec, (2, 0, 1))


def _is_diagonal(M, tol=1e-14):
    return np.abs(M - np.diag(np.diag(M))).max() <= tol * max(1.0, np.abs(M).max())


# --------------------------------------------------------------------------- #
# nonlinear SDE (optionally noise-matched against the linear one)
# --------------------------------------------------------------------------- #
def simulate_paired_trial(model, contrast, ss_full, J_aug, B, dt, n_steps, n_burn,
                          record_idx, record_every, seed, paired=True,
                          clip_nonneg=True):
    """One trial of the full nonlinear SDE, optionally noise-matched to the linear one.

    Both systems are advanced inside a single loop consuming the *same* Wiener
    increments, so their trajectories differ only through the nonlinearity.
    Both use Euler-Maruyama with the same dt, so discretisation bias is common
    to the pair and cancels in the trajectory-level comparison.

    Args:
        model: RingModel (already carrying the parameters for this condition).
        ss_full: deterministic fixed point of the full state (num_var * N).
        J_aug: augmented Jacobian at that fixed point.
        B: augmented noise gain L @ S.
        paired: also integrate the linear system with the identical noise.
        clip_nonneg: hold y1Plus/y4Plus at >= 0 and s1Plus/s4Plus at >= rectify.
            Only matters if noise pushes a state through a sqrt() branch point;
            the returned `n_clipped` counts how often it fired, and a run with
            n_clipped == 0 is the unmodified dynamics.
    Returns:
        dict with 'nonlinear' (n_rec, n_samples), optional 'linear' (same shape),
        'n_clipped', and 'record_every'.
    """
    N = model.N
    n_red = model.jacobian_dimension * N
    n_aug = J_aug.shape[0]
    m = n_aug - n_red
    tau_f = model.params['tau_f']
    tau_x = model.params['tau_x']

    J_aug = as_numpy(J_aug)
    B = as_numpy(B)
    if not _is_diagonal(B):
        raise ValueError("nonlinear SDE assumes a diagonal noise gain (L and S are diagonal)")
    sigma = np.diag(B)
    sigma_x, sigma_f = sigma[:n_red], sigma[n_red:]

    x_input = np.zeros(model.params['M'])
    x_input[model.target_angle] = contrast
    model.contrast = contrast

    # `get_Jacobian_augmented` returns [ss; zeros(m)]; the nonlinear integrator
    # needs only the num_var*N physical state, so drop any appended f block.
    ss_full = np.asarray(ss_full, dtype=np.float64).flatten()[:model.num_var * N]
    Y = ss_full.copy()
    f = np.zeros(m)
    Xlin = np.zeros(n_aug) if paired else None

    record_idx = np.asarray(record_idx, dtype=int)
    n_samples = n_steps // record_every
    rec_nl = np.empty((len(record_idx), n_samples))
    rec_lin = np.empty((len(record_idx), n_samples)) if paired else None

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    n_clipped = 0

    # slices of the reduced state that must stay in the physical branch
    yplus_slices = [slice(b * N, (b + 1) * N) for b in YPLUS_BLOCKS]
    splus_slices = [slice(b * N, (b + 1) * N) for b in SPLUS_BLOCKS]
    mem_slices = [(slice(b * N, (b + 1) * N), slice(i * N, (i + 1) * N))
                  for i, b in enumerate(MEMBRANE_BLOCKS)]

    for k in range(n_burn + n_steps):
        zx = rng.standard_normal(n_red)
        zf = rng.standard_normal(m)

        # ---- nonlinear system -------------------------------------------- #
        dY = np.asarray(model.dynm_func(0.0, Y, x_input), dtype=np.float64)
        for row, col in mem_slices:            # A f: filtered noise -> membranes
            dY[row] += f[col] / tau_x
        Y = Y + dt * dY
        Y[:n_red] += sqrt_dt * sigma_x * zx

        if clip_nonneg:
            for sl in yplus_slices:
                bad = Y[sl] < 0.0
                if bad.any():
                    n_clipped += int(bad.sum())
                    Y[sl] = np.maximum(Y[sl], 0.0)
            for sl in splus_slices:
                bad = Y[sl] < model.rectify
                if bad.any():
                    n_clipped += int(bad.sum())
                    Y[sl] = np.maximum(Y[sl], model.rectify)

        # ---- filtered-noise bank (shared by both systems) ----------------- #
        f = f + dt * (-f / tau_f) + sqrt_dt * sigma_f * zf

        # ---- linear system, identical increments -------------------------- #
        if paired:
            Xlin = Xlin + dt * (J_aug @ Xlin)
            Xlin[:n_red] += sqrt_dt * sigma_x * zx
            Xlin[n_red:] += sqrt_dt * sigma_f * zf

        if k >= n_burn:
            j = k - n_burn
            if j % record_every == 0 and j // record_every < n_samples:
                s = j // record_every
                rec_nl[:, s] = Y[record_idx] - ss_full[record_idx]
                if paired:
                    rec_lin[:, s] = Xlin[record_idx]

    if not np.all(np.isfinite(rec_nl)):
        raise FloatingPointError("nonlinear SDE diverged -- reduce dt")

    out = {'nonlinear': rec_nl, 'n_clipped': n_clipped, 'record_every': record_every}
    if paired:
        out['linear'] = rec_lin
    return out


# --------------------------------------------------------------------------- #
# spectral estimation
# --------------------------------------------------------------------------- #
def welch_psd(traces, fs, nperseg, overlap=0.5):
    """Welch PSD of `traces` (..., n_samples). Returns (freqs, psd)."""
    nperseg = int(min(nperseg, traces.shape[-1]))
    freqs, psd = welch(traces, fs=fs, nperseg=nperseg,
                       noverlap=int(overlap * nperseg),
                       detrend='constant', scaling='density',
                       window='hann', axis=-1)
    return freqs, psd


def average_psd(psd_per_trial):
    """Mean and standard error across the leading (trial) axis."""
    psd_per_trial = np.asarray(psd_per_trial)
    mean = psd_per_trial.mean(axis=0)
    n = psd_per_trial.shape[0]
    sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


def cross_spectral_matrix(traces, fs, nperseg, overlap=0.5):
    """Welch cross-spectral matrix of `traces` (..., k, n_samples).

    Returns (freqs, C) with C of shape (..., n_freq, k, k), complex. Uses
    `scipy.signal.csd` with kwargs identical to `welch_psd`, so the two
    normalisations cancel and `real(C[..., i, i]) == welch_psd(traces_i)`.

    Two conventions worth stating, because they are easy to get wrong:

      * scipy computes  Pxy = <conj(X) Y>,  whereas matrix_spectrum's
        spectral_matrix[i, j] = <X_i conj(X_j)>. The two are transposes of one
        another. Irrelevant for magnitude-squared coherence; load-bearing if the
        cross-spectrum *phase* is ever compared.
      * scipy applies the one-sided factor of 2 to auto- and cross-spectra
        alike, and that factor cancels in coherence -- so a coherence comparison
        needs no convention correction on either side.
    """
    traces = np.asarray(traces)
    k = traces.shape[-2]
    nperseg = int(min(nperseg, traces.shape[-1]))
    kw = dict(fs=fs, nperseg=nperseg, noverlap=int(overlap * nperseg),
              detrend='constant', scaling='density', window='hann', axis=-1)
    C = None
    freqs = None
    for i in range(k):
        for j in range(i, k):
            freqs, cij = csd(traces[..., i, :], traces[..., j, :], **kw)
            if C is None:
                C = np.empty(traces.shape[:-2] + (freqs.size, k, k), dtype=np.complex128)
            if i == j:
                C[..., i, i] = np.real(cij)   # exactly real by construction
            else:
                C[..., i, j] = cij
                C[..., j, i] = np.conj(cij)
    return freqs, C


def pool_cross_spectra(blocks):
    """Mean cross-spectral matrix over the leading (trial) axis.

    Coherence must be formed from pooled *spectra*, never by averaging per-trial
    coherences. Magnitude-squared coherence is a ratio functional, so its upward
    bias (~ (1 - C)^2 / n_segments, and exactly 1 in the single-segment limit)
    survives averaging over trials untouched -- averaging shrinks the variance
    but not the bias. `scipy.signal.csd` already averages over a trial's
    segments, and every trial contributes the same number of segments, so the
    mean of the per-trial matrices *is* the grand mean over all segments.
    """
    return np.asarray(blocks).mean(axis=0)


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def selftest_ou(tau=0.02, sigma=1.0, dt=1e-4, T=200.0, seed=0, verbose=True):
    """Check the PSD convention on a scalar OU process with a known spectrum.

    dx = -x/tau dt + sigma dW  has two-sided angular PSD
        S(w) = sigma^2 / (w^2 + 1/tau^2),
    so the one-sided welch PSD should be 2 S(2 pi f). The per-bin scatter is
    ~1/sqrt(n_segments); what matters is that the *mean* ratio is 1, since a
    wrong convention would show up as a constant factor. Returns (bias, median
    relative error) over 1-200 Hz.
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    a = np.exp(-dt / tau)
    s = sigma * np.sqrt(tau / 2.0 * (1 - a ** 2))   # exact one-step std
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma * np.sqrt(tau / 2.0)
    z = rng.standard_normal(n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + s * z[k]

    fs = 1.0 / dt
    freqs, psd = welch_psd(x, fs, nperseg=int(2.0 * fs))
    band = (freqs >= 1) & (freqs <= 200)
    w = 2 * np.pi * freqs[band]
    ref = to_onesided(sigma ** 2 / (w ** 2 + 1.0 / tau ** 2))
    ratio = psd[band] / ref
    err = np.abs(ratio - 1.0)
    n_seg = 2 * int(T / 2.0) - 1
    if verbose:
        print(f"[selftest_ou] mean PSD ratio (sim/analytic) = {ratio.mean():.4f} "
              f"(expect 1.000 +/- {1/np.sqrt(n_seg * band.sum()):.4f})")
        print(f"[selftest_ou] median per-bin relative error = {np.median(err):.4f} "
              f"(expect ~{1/np.sqrt(n_seg):.4f} from estimator scatter)")
    return float(ratio.mean()), float(np.median(err))


def _toy_linear_system():
    """A 3-D linear SDE with a resonance, shared by the self-tests.

    Damped oscillator (~30 Hz) coupled to a slow leak; nothing special about the
    numbers beyond giving a spectrum with a peak and a roll-off. Note x1 is
    exactly dx0/dt, which the coherence self-test exploits.
    """
    w0, zeta = 2 * np.pi * 30.0, 0.15
    J = np.array([[0.0, 1.0, 0.0],
                  [-w0 ** 2, -2 * zeta * w0, 20.0],
                  [0.0, 0.0, -50.0]])
    L = np.diag([0.0, 1.0, 0.7])
    S = np.diag([1.0, 3.0, 2.0])
    return torch.as_tensor(J), torch.as_tensor(L), torch.as_tensor(S)


def selftest_linear_system(dt=2e-4, T=60.0, n_trials=8, seed=1, verbose=True):
    """End-to-end check on a 3-D linear SDE with a resonance.

    Exercises exactly the path used for the real model -- `matrix_solution`
    through `analytical_psd`, `simulate_linear_batch` (both integrators),
    `welch_psd` and `average_psd` -- on a system small enough that the whole
    thing runs in seconds. Returns {integrator: mean PSD ratio}.
    """
    Jt, Lt, St = _toy_linear_system()
    B = noise_gain(Lt, St)

    fs = 1.0 / dt
    nperseg = int(2.0 * fs)
    n_steps = int(T / dt)
    n_burn = int(1.0 / dt)
    indices = {'x0': 0, 'x1': 1, 'x2': 2}

    freqs = np.fft.rfftfreq(nperseg, d=dt)
    band = (freqs >= 2) & (freqs <= 150)
    ana = analytical_psd(Jt, Lt, St, freqs[band], indices)

    out = {}
    for integrator in ('exact', 'euler'):
        traces = simulate_linear_batch(Jt, B, dt, n_steps, n_burn, n_trials,
                                       list(indices.values()), 1,
                                       np.random.default_rng(seed),
                                       integrator=integrator)
        _, psd = welch_psd(traces, fs, nperseg)
        mean, _ = average_psd(psd[..., band])
        ratios = {lab: float(np.mean(mean[i] / ana[lab]))
                  for i, lab in enumerate(indices)}
        out[integrator] = ratios
        if verbose:
            txt = '  '.join(f"{k}={v:.4f}" for k, v in ratios.items())
            print(f"[selftest_linear] {integrator:>5}: mean PSD ratio (sim/analytic)  {txt}")
    return out


def selftest_coherence(dt=2e-4, T=60.0, n_trials=8, seed=2, verbose=True):
    """Check the empirical cross-spectral estimator against analytical coherence.

    Uses the same 3-D toy system as `selftest_linear_system`. Three assertions:

      1. `real(csd(x_i, x_i))` reproduces `welch(x_i)` -- the two normalisations
         match, which is what makes the coherence ratio meaningful at all.
      2. x1 is exactly dx0/dt, so S_01 = i w S_00 and the coherence between them
         is identically 1. Any plumbing error shows up here first.
      3. For a genuinely partial pair (x0, x2) the *pooled* coherence tracks the
         analytical value, while averaging per-trial coherences is visibly biased
         high -- which is the whole reason pooling is done spectra-first.

    Returns a dict of the measured quantities.
    """
    Jt, Lt, St = _toy_linear_system()
    B = noise_gain(Lt, St)

    fs = 1.0 / dt
    nperseg = int(2.0 * fs)
    n_steps = int(T / dt)
    n_burn = int(1.0 / dt)
    indices = {'x0': 0, 'x1': 1, 'x2': 2}
    pairs = (('x0', 'x1'), ('x0', 'x2'))

    freqs = np.fft.rfftfreq(nperseg, d=dt)
    band = (freqs >= 2) & (freqs <= 150)
    ana = analytical_spectra(Jt, Lt, St, freqs[band], indices, pairs=pairs)

    traces = simulate_linear_batch(Jt, B, dt, n_steps, n_burn, n_trials,
                                   list(indices.values()), 1,
                                   np.random.default_rng(seed), integrator='exact')

    # (1) csd diagonal vs welch
    _, psd = welch_psd(traces, fs, nperseg)
    _, blocks = cross_spectral_matrix(traces, fs, nperseg)
    diag = np.real(np.einsum('tfkk->tkf', blocks))
    diag_err = np.abs(diag - psd).max() / np.abs(psd).max()
    if diag_err > 1e-12:
        raise ValueError(f"csd diagonal does not match welch (rel {diag_err:.2e})")

    pooled = pool_cross_spectra(blocks)[band]
    out = {'diag_rel_err': float(diag_err)}

    # (2) exact-unity pair, and (3) partial pair with the bias comparison
    for a, b in pairs:
        ia, ib = list(indices).index(a), list(indices).index(b)
        pooled_coh = coherence_from_block(pooled, ia, ib)
        per_trial = coherence_from_block(blocks[:, band], ia, ib).mean(axis=0)
        ref = ana['coherence'][(a, b)]
        out[f'{a}{b}'] = {'analytical': float(ref.mean()),
                          'pooled': float(pooled_coh.mean()),
                          'per_trial_mean': float(per_trial.mean())}
        if verbose:
            print(f"[selftest_coherence] {a}-{b}: analytical={ref.mean():.4f}  "
                  f"pooled={pooled_coh.mean():.4f}  "
                  f"per-trial-averaged={per_trial.mean():.4f} (biased high)")
        if abs(pooled_coh.mean() - ref.mean()) > 0.05:
            raise ValueError(f"pooled coherence({a},{b}) = {pooled_coh.mean():.4f} "
                             f"does not match analytical {ref.mean():.4f}")

    # The bias must be visible on the partial pair, otherwise the pooling scheme
    # is untested by this self-test and could silently regress.
    partial = out['x0x2']
    if partial['per_trial_mean'] - partial['pooled'] < 0.01:
        raise ValueError("per-trial-averaged coherence is not measurably biased "
                         "high; the pooling comparison is not exercising anything")
    if verbose:
        print(f"[selftest_coherence] csd/welch diagonal agreement: {diag_err:.2e}")
    return out


def selftest_low_pass(sigma=0.03, tau=0.05, rho=0.1, verbose=True):
    """Pin the one-sided factor on the additive `low_pass_add` term.

    With the dynamic noise zeroed (L = 0) the entire spectrum *is* the additive
    term, so the analytical block must equal exactly the one-sided
    2 P(w) (ones + rho I) that the numerical side adds -- which is the assertion
    that keeps the two call sites from drifting by a factor of 2.

    The coherence then takes its noise-only value 1/(1+rho)^2, since the shared
    process gives S_xy = P while S_xx = S_yy = P(1+rho).
    """
    Jt, _, St = _toy_linear_system()
    Lz = torch.zeros_like(St)
    freqs = np.array([1.0, 10.0, 100.0])
    indices = {'x0': 0, 'x1': 1, 'x2': 2}
    ana = analytical_spectra(Jt, Lz, St, freqs, indices, pairs=(('x0', 'x1'),),
                             noise_sigma=sigma, noise_tau=tau,
                             low_pass_add=True, rho=rho, n_check=3)
    expected = 2.0 * low_pass_matrix(freqs, sigma, tau, rho, len(indices))
    err = np.abs(ana['block'] - expected).max() / np.abs(expected).max()
    coh_err = np.abs(ana['coherence'][('x0', 'x1')] - 1.0 / (1 + rho) ** 2).max()
    if err > 1e-12 or coh_err > 1e-12:
        raise ValueError(f"low_pass_add term mismatch: block rel {err:.2e}, "
                         f"coherence abs {coh_err:.2e}")
    if verbose:
        print(f"[selftest_low_pass] one-sided additive term exact to {err:.2e}; "
              f"noise-only coherence = 1/(1+rho)^2 to {coh_err:.2e}")
    return float(err), float(coh_err)


if __name__ == '__main__':
    selftest_ou()
    selftest_linear_system()
    selftest_coherence()
    selftest_low_pass()
