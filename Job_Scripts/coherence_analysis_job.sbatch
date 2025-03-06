#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Coherence_Analysis
#SBATCH --output=job.%j.out

module purge

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis"
RESULTS_DIR="${BASE_DIR}/Results_3/config_${CONFIG_NUM}"

singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
cd "${BASE_DIR}"

echo "Running Coherence Analysis"
python ${BASE_DIR}/Analysis/Coherence_analysis.py "${NEW_CONFIG}"

echo "Plotting Coherence Analysis"
# python ${BASE_DIR}/Plotting/Plot_coherence_fixed_contrast.py "${RESULTS_DIR}"
python ${BASE_DIR}/Plotting/Plot_coherence_fixed_gamma.py "${RESULTS_DIR}"
EOF 