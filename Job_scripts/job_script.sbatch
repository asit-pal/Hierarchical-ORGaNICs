#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Power_Spectra
#SBATCH --output=job.%j.out

module purge # Clear any existing modules

# Define base paths
BASE_DIR="/home/ap6603/Script_analysis" # Base directory
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml" # Source config file

# Find the next available config number
n=1
while [ -d "${BASE_DIR}/Results/config_${n}" ]; do # Check if results directory exists
    ((n++))
done

# specify the config number
# n=191 #Change this to specify the config number (for plotting)

# Create new results directory structure
RESULTS_DIR="${BASE_DIR}/Results/config_${n}"
mkdir -p "${RESULTS_DIR}/Data" # Create data directory
mkdir -p "${RESULTS_DIR}/Plots" # Create plots directory

# Copy and modify config file
NEW_CONFIG="${RESULTS_DIR}/config_${n}.yaml"
cp "${SOURCE_CONFIG}" "${NEW_CONFIG}" # Copy source config to new config

# Modify only Power_spectra/Feedback_gain parameters in the copied config
sed -i '/Power_spectra:/,/Input_gain_beta1:/ s/gamma_vals: XXXX/gamma_vals: [1.0]/' "${NEW_CONFIG}" # Modify gamma_vals
sed -i '/Power_spectra:/,/Input_gain_beta1:/ s/c_vals: XXXX/c_vals: [0.032,0.064,0.125,0.25,0.5,1.0]/' "${NEW_CONFIG}" # Modify c_vals

# # #Modify alpha1 and alpha4 in the copied config
# sed -i '/model_params:/ s/alpha1: 10.0/alpha1: 0.0/' "${NEW_CONFIG}" # Modify alpha1
# sed -i '/model_params:/ s/alpha4: 10.0/alpha4: 0.0/' "${NEW_CONFIG}" # Modify alpha4

# Modify noise parameters in the copied config
sed -i 's/delta_tau: XXXX/delta_tau: 3.0/' "${NEW_CONFIG}" # Modify delta_tau
sed -i 's/low_pass_add: XXXX/low_pass_add: true/' "${NEW_CONFIG}" # Modify low_pass_add
sed -i 's/get_simulated_taus: XXXX/get_simulated_taus: false/' "${NEW_CONFIG}" # Modify get_simulated_taus
sed -i 's/poisson: XXXX/poisson: false/' "${NEW_CONFIG}" # Modify poisson
sed -i 's/noise_sigma: XXXX/noise_sigma: 0.01/' "${NEW_CONFIG}" # Modify noise_sigma
sed -i 's/noise_tau: XXXX/noise_tau: 0.3/' "${NEW_CONFIG}" # Modify noise_tau
sed -i 's/noise: XXXX/noise: 0.01/' "${NEW_CONFIG}" # Modify noise

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


# python ${BASE_DIR}/Analysis/Power_spectra_analysis.py "${NEW_CONFIG}"

# python ${BASE_DIR}/Plotting/Plot_power_spectra_fixed_gamma.py "${RESULTS_DIR}"

# python ${BASE_DIR}/Plotting/Plotting_power_spectra.py "${RESULTS_DIR}"

# python ${BASE_DIR}/Analysis/Coherence_analysis.py "${NEW_CONFIG}"

# python ${BASE_DIR}/Plotting/Plotting_coherence.py "${RESULTS_DIR}"

python ${BASE_DIR}/Analysis/Communication_analysis.py "${NEW_CONFIG}"

python ${BASE_DIR}/Plotting/Plotting_communication_analysis.py "${RESULTS_DIR}"
EOF