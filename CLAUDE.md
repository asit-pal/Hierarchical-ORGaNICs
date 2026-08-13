# CLAUDE.md

## Project Overview

Hierarchical ORGaNICs (Oscillatory Recurrent Gated Neural Integrator Circuits) — a computational neuroscience model of inter-area communication in visual cortex. This codebase accompanies a paper and implements a linearized matrix-based analysis of neural dynamics, coherence, power spectra, and communication subspaces.

## Architecture

### Core Model (`Models/Model.py`)
- `RingModel` class: two-area (V1, V4) ring network with excitatory neurons
- Key methods: `get_Jacobian()`, `get_Jacobian_augmented()` (adds filtered noise variables), `get_dynamics()`, `analytical_solution()`
- `get_steady_states()`: integrates the ODE to find fixed points
- `relu()`: shared activation function (also imported by Three_area model)
- Uses `autograd` for automatic Jacobian computation

### Three-Area Variant (`Three_area/`)
- `Three_area_model.py`: extends to V1, V4, V5 with its own `RingModel` class
- `Parameters_3.py`: imports shared weight matrix utilities from `Utils.Create_weight_matrices`
- `Communication_3.py`: three-area versions of `create_S_matrix` and `create_L_matrix` (NOT duplicates — they handle 3 areas vs 2)
- `Communication_analysis_3.py`: entry point script

### Utilities (`Utils/`)
- `Create_weight_matrices.py`: `setup_parameters()` builds the full parameter dict from a YAML config. Shared functions: `Norm_matrix`, `Recurrence_matrix`, `RescaleEigenvalues`, `ReceptiveFields`
- `Coherence.py`: `Calculate_coherence()`, `Calculate_power_spectra()`, `create_L_matrix()`, `create_S_matrix()`, noise variance helpers
- `Communication.py`: `correlation()` (Lyapunov equation), `performance()` (reduced-rank regression), `Calculate_Pred_perf_Dim()`, `Calculate_Alignment()`, frequency-wise analysis functions
- `matrix_spectrum.py`: `matrix_solution` class — computes spectral matrix, auto/cross spectra, coherence from Jacobian + noise matrices
- `SDE_simulation.py`: direct stochastic integration used to *check* `matrix_spectrum`. `simulate_linear_batch()` (linearised system, exact Van Loan or Euler-Maruyama stepping), `simulate_paired_trial()` (full nonlinear model, optionally driven by the same Wiener path as the linear one), `analytical_spectra()` / `analytical_psd()`, `welch_psd()` / `cross_spectral_matrix()`, `pool_cross_spectra()`, `coherence_from_block()`, `low_pass_matrix()`, and four self-tests (`selftest_ou`, `selftest_linear_system`, `selftest_coherence`, `selftest_low_pass`)
- `SDE_contrast_index.py`: the published normalisation `(P-P_bg)/(P+P_bg)` and the metrics for comparing two families of spectra under it — `contrast_index()`, `contrast_index_sem()` / `..._bootstrap()`, `peak()` (with an at-edge flag), `index_curves()`, `index_metrics()`, `lp_fraction()`

### Analysis Scripts (`Analysis/`)
Each script is a standalone entry point: `python Analysis/<script>.py path/to/config.yaml`
- `Power_spectra_analysis.py`, `Coherence_analysis.py`, `Communication_analysis.py`
- `Gain_modulation.py`, `Stability_analysis.py`, `Stability_analysis_beta.py`
- `Freq_wise_dim_freq.py`, `Freq_wise_pred_perf_freq.py`
- `Alignment_analysis.py`, `Plot_correlation_matrix.py`
- `SDE_validation.py`: validates the analytical PSD against direct stochastic simulation (see below). `--c-index i` runs one contrast into its own shard, for SLURM arrays
- `merge_sde_validation.py`: merges the per-contrast shards and reports the published-normalisation comparison

### Plotting (`Plotting/`)
- `Plotting.py`: `setup_plot_params()` — shared matplotlib rcParams for journal figures. Exported via `Plotting/__init__.py`
- Individual `Plot_*.py` scripts: take a results directory as argument, load `.npy` data, generate PDF plots

### Job Scripts (`Job_Scripts/`)
- `submit_analysis.sh`: master submission script — takes config number, sets up Results dir, exports env vars, calls sbatch
- `*.sbatch`: SLURM job scripts using singularity container
- All sbatch files use `BASE_DIR`, `CONFIG_NUM`, `NEW_CONFIG`, `RESULTS_DIR` env vars

## Key Patterns

### Data Flow
1. `configs/config.yaml` → `setup_parameters()` → `RingModel`
2. `RingModel.get_Jacobian_augmented()` → Jacobian `J` + steady state `ss`
3. `create_L_matrix()` + `create_S_matrix()` → noise matrices
4. `matrix_solution(J, L, S)` → power spectra / coherence / spectral matrix
5. Results saved as `.npy` in `Results_*/config_N/Data/`
6. Plotting scripts read `.npy` and write `.pdf` to `Results_*/config_N/Plots/`

### Configuration
- All analysis parameters come from YAML config files
- Config sections: `model_params`, `noise_params`, `Gain_modulation`, `Communication`, `Power_spectra`, `Coherence`, `SDE_validation`
- Each analysis script checks `config[section]['enabled']` to decide what to run

### Running Jobs
```bash
# Submit all enabled analyses for a config
bash Job_Scripts/submit_analysis.sh <config_number>

# Or run a single analysis directly
python Analysis/Power_spectra_analysis.py Results_5/config_1/config_1.yaml --area V1
```

### Validating the analytical power spectra and coherence
The published spectra come from a linearisation about the deterministic fixed point.
`Analysis/SDE_validation.py` checks that approximation by simulating the SDE directly:

```bash
# Whole published contrast sweep: one SLURM array task per c_val, then merge + plots
bash Job_Scripts/submit_sde_validation.sh 79

# Or a single condition, in the foreground
python Analysis/SDE_validation.py Results_5/config_1/config_1.yaml --n-jobs 32
python Plotting/Plot_SDE_validation.py Results_5/config_1        # --show-linear for controls
```

It can run three things and reports them side by side:
1. **Linear SDE** — `dX = J_aug X dt + L S dW` with the *exact* (Van Loan) discretisation.
   Tests only the spectral-matrix formula and the PSD normalisation, not the linearisation.
2. **Full nonlinear SDE** — the real model with the same noise sources injected.
   Disagreement here, when (1) passes, is the linearisation error.
3. **Noise-matched pair** — (1) and (2) advanced in one loop consuming *identical*
   Wiener increments, giving a trajectory-level error `rms(nonlinear - linear)`.

The figures overlay only (2) on the analytical curve; the linear controls are always computed
and printed, and `--show-linear` draws them. Only `y1`/`y4` (raw membrane potentials) are
plotted — those are the LFP quantities the paper reports; the `Plus` firing rates stay in the
printed table only.

**Conventions.** `matrix_solution` returns the two-sided PSD in angular frequency;
`scipy.signal.welch` returns the one-sided PSD in Hz, so the analytical curve is multiplied by
2 before overlaying (verified by `selftest_ou`). That factor *cancels* in magnitude-squared
coherence, so the coherence comparison is convention-free.

**Coherence.** `cross_spectral_matrix()` estimates the full complex cross-spectral block using
`scipy.signal.csd` with kwargs identical to `welch_psd`, so the normalisations cancel.
Coherence must be pooled **spectra-first** — average the cross- and auto-spectra over all
trials, *then* form `|Sxy|^2 / (Sxx Syy)`. Averaging per-trial coherences leaves the
`(1-C)^2/n_segments` upward bias in place (it is exactly 1 for a single segment);
`selftest_coherence` asserts that difference is measurable.

**`low_pass_add`.** `spectral_matrix` adds `ones*P(w) + rho*I*P(w)` — a *shared* low-pass
process, which is what creates cross-channel power and hence coherence, plus a private one.
Off-diagonal entries get `P`, not `P(1+rho)`. The term is deterministic, so the analytical and
simulated spectra both take it from `low_pass_matrix()` rather than the simulation paying
estimator noise for a closed-form quantity: analytical adds it two-sided before the x2,
numerical adds `2 *` it to the already-one-sided Welch output, after pooling and before
forming coherence. `selftest_low_pass` pins that factor exactly. The `SDE_validation` config
section overrides `noise_params.low_pass_add` so the other analyses are unaffected.

Sizing matters: `burn_in` must outlast the slowest mode of `J_aug` and `dt` must be well under
the fastest. Both move with the taus: at `tau = 0.01` the modes are 553 ms / 1.00 ms, at
`tau = 0.001` they are 56 ms / **0.48 ms**. The `dt > 0.05 * tau_fast` warning does *not* fire
at `dt = 2e-5` with 1 ms taus (ratio 0.042), so `dt` was dropped to `1e-5` to keep the same
discretisation quality the c = 0.5 run was validated at; Euler bias does not cancel against
the analytical curve. `welch_nperseg_sec` (0.25 Hz resolution) and `T` (7 segments/trial, and
hence the error band) must not be traded away for runtime.

### Comparing the sweep under the published normalisation
The paper does not plot power — `Plot_PS_fixed_gamma_with_power_decay.py` plots the *contrast
index* `(P - P_bg)/(P + P_bg)` against `P_bg` at the lowest contrast, with a log-log inset
normalised by the global maximum over all `(gamma, c)` keys. Validating the figure therefore
means running the whole `c_vals` sweep and pushing both families through that normalisation:

- `Analysis/SDE_validation.py --c-index i --out-name NAME` runs one contrast (selected by
  *position* in `SDE_validation.c_vals`, never by re-typed float) into its own shard.
- `Job_Scripts/submit_sde_validation.sh N` submits the array plus an `afterok` merge job.
  `SDE_EXTRA_ARGS` reaches every task (use it for cheap preflights). The c = 0 task alone
  keeps `paired: True`.
- `Analysis/merge_sde_validation.py` collects the shards, refuses to merge an incomplete or
  inconsistent set, and prints the absolute-PSD, clipping, `lp_fraction` and contrast-index
  tables.
- `Utils/SDE_contrast_index.py` holds the single definition of the normalisation plus the
  metrics; `Plotting/Plot_SDE_validation_contrast.py` draws the overlay + residual grid and a
  published-rcParams single panel.

Three things make this comparison different from the absolute-PSD one:

- **The index is a difference in [-1, 1], not a ratio**, so every metric is an absolute
  difference. Relative error is meaningless where `I` crosses zero.
- **Each series is normalised against its own background.** That is what makes the one-sided
  x2 cancel; crossing them would reintroduce it.
- **All curves divide by one `P_bg` realisation.** A residual whose sign is the same at every
  contrast is a background artefact, not linearisation error — the merge report flags it.

Two traps worth knowing. `low_pass_add` is 93% of the y1 PSD at 20 Hz, added identically to
both sides, so raw agreement partly compares a constant with itself — read `lp_fraction`
beside any error metric. And at `c = 0` the rectified variables sit *on* their floor
(`y1Plus` steady state 9e-11 against a noise sd of 3e-8), so `clip_fraction` there is ~0.25-0.5
rather than the 5e-4 seen at c = 0.5; that is expected, and the unclipped noise-matched linear
shadow is what bounds its effect on the plotted `y1`/`y4` spectra.

## HPC Environment

- **SLURM account**: `--account=torch_pr_239_general`
- **Singularity**: `singularity exec --nv --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
- **Conda env**: sourced via `/ext3/env.sh` inside container
- **Results**: `Results_5/` is the current active results directory

## Dependencies

Core: `numpy`, `scipy`, `torch`, `autograd`, `matplotlib`, `control` (for Lyapunov solver), `slycot`, `PyYAML`, `tqdm`, `mpmath`, `sympy`, `distinctipy`

## Common Tasks

- **Add a new analysis**: Create script in `Analysis/`, add sbatch in `Job_Scripts/`, add config section in YAML
- **Change model parameters**: Edit `configs/config.yaml` or per-config YAML
- **Add a new plot**: Create `Plot_*.py` in `Plotting/`, import `setup_plot_params` from `Plotting`
- **Modify weight matrices**: Edit `Utils/Create_weight_matrices.py` — changes propagate to both two-area and three-area models
