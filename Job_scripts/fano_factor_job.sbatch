#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Fano_Factor
#SBATCH --output=job.%j.out

module purge

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis"
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml"

# Find the next available config number
n=1
while [ -d "${BASE_DIR}/Results/config_${n}" ]; do
    ((n++))
done

# Create new results directory structure
RESULTS_DIR="${BASE_DIR}/Results/config_${n}"
mkdir -p "${RESULTS_DIR}/Data/Fano_factor_data"
mkdir -p "${RESULTS_DIR}/Plots"

# Copy and modify config file
NEW_CONFIG="${RESULTS_DIR}/config_${n}.yaml"
cp "${SOURCE_CONFIG}" "${NEW_CONFIG}"

# Modify Communication/Feedback_gain parameters in the copied config
sed -i '/Communication:/,/Input_gain_beta1:/ s/gamma_vals: XXXX/gamma_vals: [0.0,0.25,0.5,0.75,1.0]/' "${NEW_CONFIG}"
sed -i '/Communication:/,/Input_gain_beta1:/ s/c_vals: XXXX/c_vals: [0.032,0.064,0.125,0.25,0.5,1.0]/' "${NEW_CONFIG}"

# Debug: Print paths
echo "Base directory: ${BASE_DIR}"
echo "Config file path: ${NEW_CONFIG}"
echo "Results directory: ${RESULTS_DIR}"
echo "Config number: ${n}"

# Export variables for use in the container
export BASE_DIR
export NEW_CONFIG
export CONFIG_NUM="${n}"

# Run the singularity container
singularity exec --overlay /home/ap6603/Pytorch/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash << EOF
source /ext3/env.sh
conda activate torch-nn
cd "${BASE_DIR}"

# Debug: Print paths inside container
echo "Current directory: $(pwd)"
echo "Config file: ${NEW_CONFIG}"
echo "Config number: ${CONFIG_NUM}"

# Run analysis
python Fano_factor.py "${NEW_CONFIG}"

# Run plotting with config number (if you have a plotting script)
# python Plotting_fano_factor.py "${RESULTS_DIR}"
EOF 