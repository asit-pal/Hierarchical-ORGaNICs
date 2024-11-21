#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=Plot_Power_Spectra
#SBATCH --output=plot_job.%j.out

# Check if config number is provided
if [ $# -ne 1 ]; then
    echo "Usage: sbatch plot_script.sbatch <config_number>"
    exit 1
fi

module purge

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis"
CONFIG_NUM=$1
RESULTS_DIR="${BASE_DIR}/Results/config_${CONFIG_NUM}"

# Check if directory exists
if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Directory ${RESULTS_DIR} does not exist"
    exit 1
fi

echo "Plotting results from: ${RESULTS_DIR}"

# Run only the plotting script in the singularity container
singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
cd "${BASE_DIR}"
echo "Current directory: $(pwd)"
echo "Results directory: ${RESULTS_DIR}"
python Plotting_power_spectra.py "${RESULTS_DIR}"
EOF