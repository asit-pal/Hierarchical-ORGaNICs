#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Comm_3_Analysis
#SBATCH --output=job.%j.out
#SBATCH --account=torch_pr_239_general

module purge

# Define base paths
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
RESULTS_DIR="${BASE_DIR}/Results_4/config_${CONFIG_NUM}"

singularity exec --nv \
    --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
source /ext3/env.sh
cd "${BASE_DIR}"

# Check if CONFIG_NUM is set
if [ -z "${CONFIG_NUM}" ]; then
  echo "Error: CONFIG_NUM environment variable is not set."
  exit 1
fi

echo "Running Three-Area Communication Analysis for Config ${CONFIG_NUM}"
# Pass CONFIG_NUM as an argument to the analysis script
python ${BASE_DIR}/Three_area/Communication_analysis_3.py "${CONFIG_NUM}"

echo "Plotting Three-Area Communication Analysis for Config ${CONFIG_NUM}"
# The plotting script already accepts RESULTS_DIR, which depends on CONFIG_NUM
python ${BASE_DIR}/Plotting/Plot_communication_analysis_3.py "${RESULTS_DIR}"

EOF