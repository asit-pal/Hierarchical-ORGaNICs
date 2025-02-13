#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Power_Spectra
#SBATCH --output=job.%j.out

module purge

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis"
RESULTS_DIR="${BASE_DIR}/Results_3/config_54"  # Directly specify the results directory

# Export variables for use in the container
export BASE_DIR
export RESULTS_DIR

# Run the singularity container
singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
cd "${BASE_DIR}"

# Debug: Print paths inside container
echo "Current directory: $(pwd)"
echo "Results directory: ${RESULTS_DIR}"

# Run power spectra analysis
python ${BASE_DIR}/Plotting/Plot_PS_fixed_gamma_remove_background_noise.py "${RESULTS_DIR}"
EOF