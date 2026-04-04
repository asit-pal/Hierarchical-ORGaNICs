# Hierarchical ORGaNICs Model

This repository implements a hierarchical version of the ORGaNICs (Oscillatory Recurrent Gated Neural Integrator Circuits) model, focusing on inter-area neural dynamics, communication subspaces, and information processing in visual cortex.

## Project Structure

```
├── Models/             # Core two-area model (RingModel class)
├── Three_area/         # Three-area model variant
├── Utils/              # Shared utilities
│   ├── Coherence.py            # Coherence and power spectra calculations
│   ├── Communication.py        # Communication subspace analysis (prediction performance, dimensionality)
│   ├── Create_weight_matrices.py  # Weight matrix construction and parameter setup
│   └── matrix_spectrum.py      # Matrix-based spectral analysis (PSD, cross-spectrum, coherence)
├── Analysis/           # Analysis scripts (each runnable from command line)
├── Plotting/           # Visualization scripts and shared plot utilities
├── Job_Scripts/        # SLURM job submission scripts for HPC clusters
├── configs/            # YAML configuration files
└── requirements.txt    # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Note: `slycot` requires a Fortran compiler (e.g., `gfortran`). On some systems you may need to install it separately.

## Usage

Each analysis script takes a YAML config file as input:

```bash
python Analysis/Coherence_analysis.py configs/config.yaml
python Analysis/Power_spectra_analysis.py configs/config.yaml
python Analysis/Communication_analysis.py configs/config.yaml
```

### Running on an HPC cluster

The `Job_Scripts/` directory contains SLURM batch scripts. To submit analyses:

```bash
bash Job_Scripts/submit_analysis.sh <config_number>
```

**Important:** The job scripts contain hardcoded paths specific to the original development environment. Update `BASE_DIR` in `submit_analysis.sh` and the singularity/conda paths in each `.sbatch` file to match your cluster setup.

## Configuration

Model and analysis parameters are defined in `configs/config.yaml`:

- `model_params`: Neural circuit parameters (gains, time constants, connectivity strengths)
- `noise_params`: Noise model configuration (filtered noise, low-pass parameters)
- `Communication`, `Power_spectra`, `Coherence`: Analysis-specific settings (contrast values, gain sweeps)

## License

Please contact the authors for licensing information.
