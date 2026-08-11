#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=SDE_Validation
#SBATCH --output=job.%j.out
#SBATCH --account=torch_pr_239_general

module purge

# Define base paths
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
RESULTS_DIR="${BASE_DIR}/Results_5/config_${CONFIG_NUM}"

singularity exec --nv \
    --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
source /ext3/env.sh
cd "${BASE_DIR}"

echo "Running SDE validation of the analytical power spectra"
python -u ${BASE_DIR}/Analysis/SDE_validation.py "${NEW_CONFIG}" --n-jobs 32

echo "Plotting SDE validation"
python -u ${BASE_DIR}/Plotting/Plot_SDE_validation.py "${RESULTS_DIR}"
EOF
