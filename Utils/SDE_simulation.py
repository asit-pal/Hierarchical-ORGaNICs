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
from scipy.signal import welch

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
# analytical reference
# --------------------------------------------------------------------------- #
def _psd_diagonal_reference(mat, freqs, indices, chunk=8):
    """Diagonal of `matrix_solution.spectral_matrix`, walked in frequency chunks.

    This is the untouched reference implementation. It materialises an
    (n_freq, n, n) complex array (~6 GB at n=864 for 500 frequencies) and
    inverts two n x n complex matrices per frequency, so it is only practical
    for spot checks -- `analytical_psd` uses it to validate the fast path.
    """
    out = {label: np.empty(len(freqs)) for label in indices}
    freqs = np.asarray(freqs, dtype=np.float64)
    for start in range(0, freqs.size, chunk):
        stop = min(start + chunk, freqs.size)
        spec = mat.spectral_matrix(freq=torch.as_tensor(freqs[start:stop]))
        for label, idx in indices.items():
            out[label][start:stop] = torch.real(spec[:, idx, idx]).cpu().numpy()
        del spec
    return out


def _psd_diagonal_fast(mat, freqs, indices):
    """Same diagonal, without forming the full spectral matrix.

    `spectral_matrix` computes  S(w) = M^-1 Q M^-H  with  M = J + i w I. Only a
    few diagonal entries are needed, and

        S_kk = e_k^T M^-1 Q M^-H e_k = a^T Q conj(a),   M^T a = e_k ,

    because M^-H e_k = conj(M^-T e_k) for real J. So one LU factorisation of M^T
    per frequency plus one back-substitution per requested index replaces two
    full n x n complex inversions -- roughly an order of magnitude less work,
    and O(n) instead of O(n_freq * n^2) memory.
    """
    n = mat.N
    Q = mat.noise_mat
    eye = torch.eye(n, dtype=torch.cdouble)
    E = torch.zeros((n, len(indices)), dtype=torch.cdouble)
    for col, idx in enumerate(indices.values()):
        E[idx, col] = 1.0
    Jc = mat.J.to(torch.cdouble)

    out = {label: np.empty(len(freqs)) for label in indices}
    labels = list(indices.keys())
    with torch.no_grad():
        for f_i, f in enumerate(np.asarray(freqs, dtype=np.float64)):
            om = 2 * np.pi * f
            A = torch.linalg.solve((Jc + 1j * om * eye).transpose(0, 1), E)  # (n, n_idx)
            # S_kk = a_k^T Q conj(a_k); imaginary part is round-off since Q is Hermitian.
            vals = torch.einsum('ak,ab,bk->k', A, Q, torch.conj(A))
            for col, label in enumerate(labels):
                out[label][f_i] = float(vals[col].real)

    if mat.low_pass_add:
        extra = np.array([noise_power_spectrum(2 * np.pi * f, mat.noise_sigma, mat.noise_tau)
                          for f in freqs])
        for label in out:                       # ones*P on every entry, + rho*P on the diagonal
            out[label] = out[label] + extra * (1.0 + mat.rho)
    return out


def analytical_psd(J, L, S, freqs, indices, noise_sigma=None, noise_tau=None,
                   low_pass_add=False, rho=None, n_check=3, verbose=False):
    """One-sided analytical PSD (welch convention) at `freqs`, for `indices`.

    The values come from the fast path, which is verified against the untouched
    `matrix_solution.spectral_matrix` at `n_check` frequencies spread over the
    grid -- so the curve being compared against the SDE is still the one the
    published pipeline produces, just computed without the 6 GB intermediate.

    Args:
        J, L, S: augmented Jacobian and noise matrices (torch tensors).
        freqs: 1-D array of frequencies in Hz.
        indices: dict {label: reduced-state index}.
        n_check: number of frequencies at which to cross-check the fast path.
    Returns:
        dict {label: array of one-sided PSD values}.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    mat = matrix_solution(J, L, S, noise_sigma, noise_tau,
                          low_pass_add=low_pass_add, rho=rho)
    fast = _psd_diagonal_fast(mat, freqs, indices)

    if n_check:
        probe_at = np.unique(np.linspace(0, freqs.size - 1, n_check).astype(int))
        ref = _psd_diagonal_reference(mat, freqs[probe_at], indices)
        for label in indices:
            a, b = fast[label][probe_at], ref[label]
            rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
            if rel.max() > 1e-6:
                raise ValueError(
                    f"fast analytical PSD disagrees with matrix_solution for '{label}' "
                    f"(max relative difference {rel.max():.3e})")
            if verbose:
                print(f"  [analytical_psd] '{label}' matches matrix_solution to "
                      f"{rel.max():.2e} at {probe_at.size} probe frequencies")

    return {label: to_onesided(v) for label, v in fast.items()}


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


def observation_noise(shape, dt, sigma, tau, rho, rng):
    """Simulate the additive measurement noise implied by `low_pass_add`.

    `matrix_spectrum` adds  ones * P(w) + I * rho * P(w)  to the spectral
    matrix, with P(w) = sigma^2 / (1 + (tau w)^2)^2. That is a *shared* scalar
    process plus an independent per-channel process, each the output of two
    cascaded one-pole filters driven by white noise:

        tau dg1 = -g1 dt + amp dW ,   tau dg2 = (-g2 + g1) dt
        =>  S_g2(w) = amp^2 / (1 + (tau w)^2)^2 ,

    with amp = sigma for the shared term and sigma*sqrt(rho) for the private
    one. Both are simulated here so the SDE traces carry the same additive
    term the analytical curve does.

    Args:
        shape: (..., n_channels, n_samples). The shared process is common to the
            channels of a given leading index but independent across them.
    """
    n_samples = shape[-1]
    n_channels = shape[-2]
    lead = tuple(shape[:-2])
    a = dt / tau
    n_warm = int(np.ceil(10 * tau / dt))   # discard the start-from-zero transient

    def cascade(batch_shape, amp):
        g1 = np.zeros(batch_shape)
        g2 = np.zeros(batch_shape)
        out = np.empty(batch_shape + (n_samples,))
        drive = (amp / tau) * np.sqrt(dt)
        for k in range(-n_warm, n_samples):
            g1 = g1 - a * g1 + drive * rng.standard_normal(batch_shape)
            g2 = g2 + a * (-g2 + g1)
            if k >= 0:
                out[..., k] = g2
        return out

    shared = cascade(lead + (1,), sigma)
    private = cascade(lead + (n_channels,), sigma * np.sqrt(rho))
    return shared + private


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


def selftest_linear_system(dt=2e-4, T=60.0, n_trials=8, seed=1, verbose=True):
    """End-to-end check on a 3-D linear SDE with a resonance.

    Exercises exactly the path used for the real model -- `matrix_solution`
    through `analytical_psd`, `simulate_linear_batch` (both integrators),
    `welch_psd` and `average_psd` -- on a system small enough that the whole
    thing runs in seconds. Returns {integrator: mean PSD ratio}.
    """
    # Damped oscillator (~30 Hz) coupled to a slow leak; nothing special about
    # the numbers beyond giving a spectrum with a peak and a roll-off.
    w0, zeta = 2 * np.pi * 30.0, 0.15
    J = np.array([[0.0, 1.0, 0.0],
                  [-w0 ** 2, -2 * zeta * w0, 20.0],
                  [0.0, 0.0, -50.0]])
    L = np.diag([0.0, 1.0, 0.7])
    S = np.diag([1.0, 3.0, 2.0])
    Jt = torch.as_tensor(J)
    Lt = torch.as_tensor(L)
    St = torch.as_tensor(S)
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


if __name__ == '__main__':
    selftest_ou()
    selftest_linear_system()
