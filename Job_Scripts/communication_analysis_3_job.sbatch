#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Comm_3_Analysis
#SBATCH --output=job.%j.out

module purge

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis"
RESULTS_DIR="${BASE_DIR}/Results_4/config_${CONFIG_NUM}"

singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
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