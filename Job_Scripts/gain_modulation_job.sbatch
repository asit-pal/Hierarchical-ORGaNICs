#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=Gain_Mod
#SBATCH --output=job.%j.out
#SBATCH --account=torch_pr_239_general

module purge

# Define base paths
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
RESULTS_DIR="${BASE_DIR}/Results_4/config_${CONFIG_NUM}"

singularity exec --nv --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
cd '${BASE_DIR}'

echo 'Running Gain Modulation Analysis'
python '${BASE_DIR}/Analysis/Gain_modulation.py' '${NEW_CONFIG}'

echo 'Plotting Gain Modulation Analysis'
python '${BASE_DIR}/Plotting/Plot_gain_modulation.py' '${RESULTS_DIR}'
" 