#!/bin/bash
# vim: syntax=bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Power_Spectra
#SBATCH --output=job.%j.out

module purge # Clear any existing modules

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis" # Base directory
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml" # Source config file

# Find the next available config number
n=1
while [ -d "${BASE_DIR}/Results_3/config_${n}" ]; do # Check if results directory exists
    ((n++))
done

# specify the config number
# n=53 #Change this to specify the config number (for plotting)

# Create new results directory structure
RESULTS_DIR="${BASE_DIR}/Results_3/config_${n}"
mkdir -p "${RESULTS_DIR}/Data" # Create data directory
mkdir -p "${RESULTS_DIR}/Plots" # Create plots directory

# Copy and modify config file
NEW_CONFIG="${RESULTS_DIR}/config_${n}.yaml"
cp "${SOURCE_CONFIG}" "${NEW_CONFIG}" # Copy source config to new config



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

######################### Power spectra analysis #########################
python ${BASE_DIR}/Analysis/Power_spectra_analysis.py "${NEW_CONFIG}"

# python ${BASE_DIR}/Plotting/Plot_power_spectra_fixed_gamma.py "${RESULTS_DIR}"

# python ${BASE_DIR}/Plotting/Plotting_power_spectra.py "${RESULTS_DIR}"
python ${BASE_DIR}/Plotting/Plot_PS_fixed_gamma_remove_background_noise.py "${RESULTS_DIR}"
# python ${BASE_DIR}/Plotting/Plot_PS_fixed_gamma_remove_background_noise_V2.py "${RESULTS_DIR}"
######################### Coherence analysis #########################

# python ${BASE_DIR}/Analysis/Coherence_analysis.py "${NEW_CONFIG}"

# # python ${BASE_DIR}/Plotting/Plotting_coherence.py "${RESULTS_DIR}"

# python ${BASE_DIR}/Plotting/Plot_coherence_fixed_gamma.py "${RESULTS_DIR}"

######################### Communication analysis #########################

python ${BASE_DIR}/Analysis/Communication_analysis.py "${NEW_CONFIG}"

python ${BASE_DIR}/Plotting/Plotting_communication_analysis.py "${RESULTS_DIR}"

######################### Gain modulation analysis #########################

# python ${BASE_DIR}/Analysis/Gain_modulation.py "${NEW_CONFIG}"

# python ${BASE_DIR}/Plotting/Plotting_gain_modulation.py "${RESULTS_DIR}"
EOF
