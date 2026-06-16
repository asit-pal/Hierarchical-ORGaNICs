#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=Pred_Perf_vs_Freq  # Changed job name
#SBATCH --output=job.%j.out
#SBATCH --account=torch_pr_239_general

module purge

# Define base paths (assuming these are the same)
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
RESULTS_DIR="${BASE_DIR}/Results_4/config_${CONFIG_NUM}" 

singularity exec --nv \
    --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
source /ext3/env.sh
cd "${BASE_DIR}"

# echo "Running Frequency-wise Decomposition Analysis"
python ${BASE_DIR}/Analysis/Freq_wise_pred_perf_freq.py "${NEW_CONFIG}" 

echo "Plotting Frequency Decomposition Analysis" # Commented out plotting for now
python ${BASE_DIR}/Plotting/Plot_freq_wise_decom.py "${RESULTS_DIR}" 

EOF