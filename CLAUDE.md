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
- `SDE_simulation.py`: direct stochastic integration used to *check* `matrix_spectrum`. `simulate_linear_batch()` (linearised system, exact Van Loan or Euler-Maruyama stepping), `simulate_paired_trial()` (full nonlinear model, optionally driven by the same Wiener path as the linear one), `analytical_psd()`, `welch_psd()`, and two self-tests (`selftest_ou`, `selftest_linear_system`)

### Analysis Scripts (`Analysis/`)
Each script is a standalone entry point: `python Analysis/<script>.py path/to/config.yaml`
- `Power_spectra_analysis.py`, `Coherence_analysis.py`, `Communication_analysis.py`
- `Gain_modulation.py`, `Stability_analysis.py`, `Stability_analysis_beta.py`
- `Freq_wise_dim_freq.py`, `Freq_wise_pred_perf_freq.py`
- `Alignment_analysis.py`, `Plot_correlation_matrix.py`
- `SDE_validation.py`: validates the analytical PSD against direct stochastic simulation (see below)

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

### Validating the analytical power spectra
The published spectra come from a linearisation about the deterministic fixed point.
`Analysis/SDE_validation.py` checks that approximation by simulating the SDE directly:

```bash
python Analysis/SDE_validation.py Results_5/config_1/config_1.yaml --n-jobs 16
python Plotting/Plot_SDE_validation.py Results_5/config_1
```

It runs three things and reports them side by side:
1. **Linear SDE** — `dX = J_aug X dt + L S dW` with the *exact* (Van Loan) discretisation.
   Tests only the spectral-matrix formula and the PSD normalisation, not the linearisation.
2. **Full nonlinear SDE** — the real model with the same noise sources injected.
   Disagreement here, when (1) passes, is the linearisation error.
3. **Noise-matched pair** — (1) and (2) advanced in one loop consuming *identical*
   Wiener increments, giving a trajectory-level error `rms(nonlinear - linear)`.

Per-trial Welch PSDs are averaged across trials. Conventions: `matrix_solution` returns the
two-sided PSD in angular frequency; `scipy.signal.welch` returns the one-sided PSD in Hz, so
the analytical curve is multiplied by 2 before overlaying (verified by `selftest_ou`).

Sizing matters: `burn_in` must outlast the slowest mode of `J_aug` (~550 ms for the default
config) and `dt` must be well under the fastest (`tau_f`, 1 ms). The script warns when either
is violated.

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
